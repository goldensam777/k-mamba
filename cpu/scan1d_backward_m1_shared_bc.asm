; scan1d_backward_m1_shared_bc.asm — AVX2 vectorized backward pass
;
; M=1 with shared B/C (one value per dimension, shared across time).
; Requires A_diag precomputed (exp(delta*A) per timestep+dim).
; If A_diag is NULL, falls back to Taylor exp approximation.
;
; Target: x86-64 System V ABI, AVX2+FMA
;
; Register allocation (callee-saved, survive function calls):
;   r15 = p            (struct pointer — single source of truth)
;   r14 = adj_h        (calloc'd buffer, [D] floats)
;   r13 = D            (dimension count)
;   r12 = t            (time counter, L-1 downto 0)
;   rbx = free/temp
;
; ScanBackwardSharedParams layout:
;   [p+0]   x       [L*D]       [p+64]  dy       [L*D]
;   [p+8]   A       [D]         [p+72]  dx       [L*D]
;   [p+16]  A_diag  [L*D]/NULL  [p+80]  dA       [D]
;   [p+24]  B       [D]         [p+88]  dB       [D]
;   [p+32]  C       [D]         [p+96]  dC       [D]
;   [p+40]  delta   [L]         [p+104] ddelta   [L]
;   [p+48]  h0      [D]/NULL    [p+112] L
;   [p+56]  h       [L*D]       [p+120] D

%include "types.inc"

extern calloc
extern memset
extern free

; ── Stack frame layout (local variables) ──────────────────────────
%define LOCALS      48
%define LOC_HROW     0          ; &h[t*D]
%define LOC_XROW     8          ; &x[t*D]
%define LOC_DYROW   16          ; &dy[t*D]
%define LOC_HPREV   24          ; &h[(t-1)*D], or h0, or NULL
%define LOC_DXROW   32          ; &dx[t*D]
%define LOC_ADIAG   40          ; &A_diag[t*D] or NULL

section .rodata
align 32
bwd_half_vec:   times 8 dd 0.5
bwd_third_vec:  times 8 dd 0.33333334   ; 1/3 (for x^3/6 = x^2/2 * x * 1/3)
bwd_one_vec:    times 8 dd 1.0

section .text
global scan1d_backward_m1_shared_bc_asm

scan1d_backward_m1_shared_bc_asm:
    ; ── Prologue ──────────────────────────────────────────────────
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, LOCALS

    ; ── Validate ──────────────────────────────────────────────────
    test    rdi, rdi
    jz      .epilogue

    mov     r15, rdi                    ; r15 = p
    mov     r13, [r15 + 120]            ; r13 = D

    ; ── calloc(D, sizeof(float)) for adj_h ────────────────────────
    mov     rdi, r13
    mov     rsi, 4
    call    calloc
    test    rax, rax
    jz      .epilogue
    mov     r14, rax                    ; r14 = adj_h

    ; ── Zero output buffers ───────────────────────────────────────
    ; dx [L*D]
    mov     rdi, [r15 + 72]
    xor     esi, esi
    mov     rdx, [r15 + 112]           ; L
    imul    rdx, r13                   ; L*D
    shl     rdx, 2                     ; bytes
    call    memset

    ; dA [D]
    mov     rdi, [r15 + 80]
    xor     esi, esi
    mov     rdx, r13
    shl     rdx, 2
    call    memset

    ; dB [D]
    mov     rdi, [r15 + 88]
    xor     esi, esi
    mov     rdx, r13
    shl     rdx, 2
    call    memset

    ; dC [D]
    mov     rdi, [r15 + 96]
    xor     esi, esi
    mov     rdx, r13
    shl     rdx, 2
    call    memset

    ; ddelta [L]
    mov     rdi, [r15 + 104]
    xor     esi, esi
    mov     rdx, [r15 + 112]           ; L
    shl     rdx, 2
    call    memset

    ; ── Main loop: t = L-1 downto 0 ──────────────────────────────
    mov     r12, [r15 + 112]            ; L
    dec     r12                         ; t = L - 1

.time_loop:
    test    r12, r12
    jl      .free_adj_h

    ; ── Compute byte offset: t * D * 4 ───────────────────────────
    mov     rax, r12
    imul    rax, r13                    ; t * D
    shl     rax, 2                      ; bytes

    ; ── Precompute row pointers ───────────────────────────────────
    mov     rdi, [r15 + 56]             ; p->h
    add     rdi, rax
    mov     [rsp + LOC_HROW], rdi

    mov     rdi, [r15 + 0]              ; p->x
    add     rdi, rax
    mov     [rsp + LOC_XROW], rdi

    mov     rdi, [r15 + 64]             ; p->dy
    add     rdi, rax
    mov     [rsp + LOC_DYROW], rdi

    mov     rdi, [r15 + 72]             ; p->dx
    add     rdi, rax
    mov     [rsp + LOC_DXROW], rdi

    ; A_diag row
    mov     rdi, [r15 + 16]             ; p->A_diag
    test    rdi, rdi
    jz      .adiag_null
    add     rdi, rax
.adiag_null:
    mov     [rsp + LOC_ADIAG], rdi

    ; h_prev row
    test    r12, r12
    jnz     .has_prev
    mov     rdi, [r15 + 48]             ; t == 0: use h0 (may be NULL)
    jmp     .hprev_done
.has_prev:
    mov     rdi, [r15 + 56]             ; p->h
    lea     rbx, [r12 - 1]
    imul    rbx, r13
    shl     rbx, 2                      ; (t-1) * D * 4
    add     rdi, rbx
.hprev_done:
    mov     [rsp + LOC_HPREV], rdi

    ; ── Broadcast delta[t] ────────────────────────────────────────
    mov     rax, [r15 + 40]             ; p->delta
    vbroadcastss ymm0, [rax + r12*4]    ; ymm0 = delta[t]

    ; ── ddt accumulator ───────────────────────────────────────────
    vxorps  ymm1, ymm1, ymm1           ; ymm1 = ddt vector accumulator
    xorps   xmm15, xmm15               ; xmm15 = ddt scalar accumulator

    ; ── Vectorized loop: 8 elements at a time ─────────────────────
    xor     ecx, ecx                    ; d = 0

.vec_loop:
    lea     rax, [r13 - 8]
    cmp     rcx, rax
    jg      .scalar_tail

    ; Load inputs from precomputed row pointers
    mov     rax, [rsp + LOC_DYROW]
    vmovups ymm2, [rax + rcx*4]        ; dy[d..d+7]

    mov     rax, [rsp + LOC_HROW]
    vmovups ymm3, [rax + rcx*4]        ; h[t*D + d]

    mov     rax, [rsp + LOC_XROW]
    vmovups ymm4, [rax + rcx*4]        ; x[t*D + d]

    mov     rax, [r15 + 8]             ; p->A
    vmovups ymm5, [rax + rcx*4]        ; A[d]

    mov     rax, [r15 + 24]            ; p->B
    vmovups ymm6, [rax + rcx*4]        ; B[d]

    mov     rax, [r15 + 32]            ; p->C
    vmovups ymm7, [rax + rcx*4]        ; C[d]

    vmovups ymm8, [r14 + rcx*4]        ; adj_h[d]

    ; h_prev (0 if NULL)
    mov     rax, [rsp + LOC_HPREV]
    test    rax, rax
    jz      .zero_hprev_vec
    vmovups ymm9, [rax + rcx*4]
    jmp     .hprev_vec_ok
.zero_hprev_vec:
    vxorps  ymm9, ymm9, ymm9
.hprev_vec_ok:

    ; dA_val = A_diag[t*D+d] or Taylor exp(delta*A)
    mov     rax, [rsp + LOC_ADIAG]
    test    rax, rax
    jz      .taylor_exp_vec
    vmovups ymm10, [rax + rcx*4]
    jmp     .da_vec_ok
.taylor_exp_vec:
    ; exp(x) ≈ 1 + x + x²/2 + x³/6 where x = delta*A (small negative)
    vmulps  ymm10, ymm0, ymm5          ; x = delta * A
    vmulps  ymm11, ymm10, ymm10        ; x²
    vmovaps ymm12, ymm10               ; accumulator = x
    vmulps  ymm13, ymm11, [rel bwd_half_vec]   ; x²/2
    vaddps  ymm12, ymm12, ymm13        ; x + x²/2
    vmulps  ymm13, ymm11, ymm10        ; x³
    vmulps  ymm13, ymm13, [rel bwd_half_vec]
    vmulps  ymm13, ymm13, [rel bwd_third_vec]  ; x³/6
    vaddps  ymm12, ymm12, ymm13        ; x + x²/2 + x³/6
    vaddps  ymm10, ymm12, [rel bwd_one_vec]    ; 1 + x + x²/2 + x³/6
.da_vec_ok:

    ; ── ah = adj_h + dy * C ──────────────────────────────────────
    vmulps  ymm11, ymm2, ymm7          ; dy * C
    vaddps  ymm11, ymm11, ymm8         ; ah = adj_h + dy * C

    ; ── dC[d] += dy * h ──────────────────────────────────────────
    mov     rax, [r15 + 96]            ; p->dC
    vmovups ymm12, [rax + rcx*4]
    vfmadd231ps ymm12, ymm2, ymm3     ; dC += dy * h
    vmovups [rax + rcx*4], ymm12

    ; ── dB[d] += ah * delta * x ──────────────────────────────────
    mov     rax, [r15 + 88]            ; p->dB
    vmovups ymm12, [rax + rcx*4]
    vmulps  ymm13, ymm11, ymm0        ; ah * delta
    vfmadd231ps ymm12, ymm13, ymm4    ; dB += (ah*delta) * x
    vmovups [rax + rcx*4], ymm12

    ; ── dA[d] += ah * delta * dA_val * h_prev ────────────────────
    mov     rax, [r15 + 80]            ; p->dA
    vmovups ymm12, [rax + rcx*4]
    vmulps  ymm13, ymm11, ymm0        ; ah * delta
    vmulps  ymm13, ymm13, ymm10       ; * dA_val
    vfmadd231ps ymm12, ymm13, ymm9    ; dA += (ah*delta*dA_val) * h_prev
    vmovups [rax + rcx*4], ymm12

    ; ── dx[t*D+d] = ah * delta * B ───────────────────────────────
    mov     rax, [rsp + LOC_DXROW]
    vmulps  ymm12, ymm11, ymm0        ; ah * delta
    vmulps  ymm12, ymm12, ymm6        ; * B
    vmovups [rax + rcx*4], ymm12

    ; ── ddt += ah * (A * dA_val * h_prev + B * x) ────────────────
    vmulps  ymm12, ymm5, ymm10        ; A * dA_val
    vmulps  ymm12, ymm12, ymm9        ; * h_prev
    vfmadd231ps ymm12, ymm6, ymm4     ; + B * x
    vfmadd231ps ymm1, ymm11, ymm12    ; ddt += ah * (...)

    ; ── adj_h[d] = ah * dA_val ────────────────────────────────────
    vmulps  ymm12, ymm11, ymm10
    vmovups [r14 + rcx*4], ymm12

    add     ecx, 8
    jmp     .vec_loop

    ; ── Scalar tail for remaining d ───────────────────────────────
.scalar_tail:
    cmp     rcx, r13
    jge     .store_ddt

    ; Load scalars
    mov     rax, [rsp + LOC_DYROW]
    movss   xmm2, [rax + rcx*4]        ; dy

    mov     rax, [rsp + LOC_HROW]
    movss   xmm3, [rax + rcx*4]        ; h

    mov     rax, [rsp + LOC_XROW]
    movss   xmm4, [rax + rcx*4]        ; x

    mov     rax, [r15 + 8]
    movss   xmm5, [rax + rcx*4]        ; A

    mov     rax, [r15 + 24]
    movss   xmm6, [rax + rcx*4]        ; B

    mov     rax, [r15 + 32]
    movss   xmm7, [rax + rcx*4]        ; C

    movss   xmm8, [r14 + rcx*4]        ; adj_h

    ; h_prev
    mov     rax, [rsp + LOC_HPREV]
    test    rax, rax
    jz      .zero_hprev_scl
    movss   xmm9, [rax + rcx*4]
    jmp     .hprev_scl_ok
.zero_hprev_scl:
    xorps   xmm9, xmm9
.hprev_scl_ok:

    ; dA_val
    mov     rax, [rsp + LOC_ADIAG]
    test    rax, rax
    jz      .taylor_exp_scl
    movss   xmm10, [rax + rcx*4]
    jmp     .da_scl_ok
.taylor_exp_scl:
    vmulss  xmm10, xmm0, xmm5          ; x = delta * A
    vmulss  xmm11, xmm10, xmm10        ; x²
    vmovss  xmm12, xmm10               ; accumulator = x
    vmulss  xmm13, xmm11, [rel bwd_half_vec]
    vaddss  xmm12, xmm12, xmm13        ; x + x²/2
    vmulss  xmm13, xmm11, xmm10        ; x³
    vmulss  xmm13, xmm13, [rel bwd_half_vec]
    vmulss  xmm13, xmm13, [rel bwd_third_vec]
    vaddss  xmm12, xmm12, xmm13        ; x + x²/2 + x³/6
    vaddss  xmm10, xmm12, [rel bwd_one_vec]
.da_scl_ok:

    ; ah = adj_h + dy * C
    vmulss  xmm11, xmm2, xmm7
    vaddss  xmm11, xmm11, xmm8

    ; dC[d] += dy * h
    mov     rax, [r15 + 96]
    movss   xmm12, [rax + rcx*4]
    vfmadd231ss xmm12, xmm2, xmm3
    movss   [rax + rcx*4], xmm12

    ; dB[d] += ah * delta * x
    mov     rax, [r15 + 88]
    movss   xmm12, [rax + rcx*4]
    vmulss  xmm13, xmm11, xmm0
    vfmadd231ss xmm12, xmm13, xmm4
    movss   [rax + rcx*4], xmm12

    ; dA[d] += ah * delta * dA_val * h_prev
    mov     rax, [r15 + 80]
    movss   xmm12, [rax + rcx*4]
    vmulss  xmm13, xmm11, xmm0
    vmulss  xmm13, xmm13, xmm10
    vfmadd231ss xmm12, xmm13, xmm9
    movss   [rax + rcx*4], xmm12

    ; dx[t*D+d] = ah * delta * B
    mov     rax, [rsp + LOC_DXROW]
    vmulss  xmm12, xmm11, xmm0
    vmulss  xmm12, xmm12, xmm6
    movss   [rax + rcx*4], xmm12

    ; ddt += ah * (A * dA_val * h_prev + B * x)
    vmulss  xmm12, xmm5, xmm10        ; A * dA_val
    vmulss  xmm12, xmm12, xmm9        ; * h_prev
    vfmadd231ss xmm12, xmm6, xmm4     ; + B * x
    vmulss  xmm13, xmm11, xmm12       ; ah * (...)
    vaddss  xmm15, xmm15, xmm13       ; ddt scalar accum

    ; adj_h[d] = ah * dA_val
    vmulss  xmm12, xmm11, xmm10
    movss   [r14 + rcx*4], xmm12

    inc     ecx
    jmp     .scalar_tail

    ; ── Horizontal sum of ymm1 + scalar accum → ddelta[t] ────────
.store_ddt:
    vextractf128 xmm2, ymm1, 1
    vaddps  xmm2, xmm2, xmm1          ; 4 floats
    vhaddps xmm2, xmm2, xmm2          ; [a+b, c+d, ...]
    vhaddps xmm2, xmm2, xmm2          ; [a+b+c+d, ...]
    vaddss  xmm2, xmm2, xmm15         ; + scalar tail

    mov     rax, [r15 + 104]           ; p->ddelta
    addss   xmm2, [rax + r12*4]       ; += existing value
    movss   [rax + r12*4], xmm2

    dec     r12
    jmp     .time_loop

    ; ── Cleanup ───────────────────────────────────────────────────
.free_adj_h:
    mov     rdi, r14
    call    free

    ; ── Epilogue ──────────────────────────────────────────────────
.epilogue:
    add     rsp, LOCALS
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    vzeroupper
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
