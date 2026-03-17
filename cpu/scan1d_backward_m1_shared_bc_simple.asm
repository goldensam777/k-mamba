; scan1d_backward_m1_shared_bc_simple.asm — Scalar backward pass
;
; Pure scalar version (no AVX2 inner loop).
; Same logic as the C scalar implementation, just in ASM.
; Requires A_diag precomputed; falls back to Taylor exp if NULL.
;
; Target: x86-64 System V ABI
;
; Register allocation (callee-saved):
;   r15 = p            (struct pointer)
;   r14 = adj_h        (calloc'd buffer, [D] floats)
;   r13 = D
;   r12 = t            (time counter, L-1 downto 0)
;   rbx = d            (dimension counter in inner loop)
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

; Stack frame locals
%define LOCALS      48
%define LOC_HROW     0
%define LOC_XROW     8
%define LOC_DYROW   16
%define LOC_HPREV   24
%define LOC_DXROW   32
%define LOC_ADIAG   40

section .text
global scan1d_backward_m1_shared_bc_simple_asm

scan1d_backward_m1_shared_bc_simple_asm:
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

    ; ── calloc(D, sizeof(float)) ──────────────────────────────────
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
    mov     rdx, [r15 + 112]
    imul    rdx, r13
    shl     rdx, 2
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
    mov     rdx, [r15 + 112]
    shl     rdx, 2
    call    memset

    ; ── Main loop: t = L-1 downto 0 ──────────────────────────────
    mov     r12, [r15 + 112]
    dec     r12

.time_loop:
    test    r12, r12
    jl      .free_adj_h

    ; ── Compute byte offset: t * D * 4 ───────────────────────────
    mov     rax, r12
    imul    rax, r13
    shl     rax, 2

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
    mov     rdi, [r15 + 16]
    test    rdi, rdi
    jz      .adiag_null
    add     rdi, rax
.adiag_null:
    mov     [rsp + LOC_ADIAG], rdi

    ; h_prev row
    test    r12, r12
    jnz     .has_prev
    mov     rdi, [r15 + 48]             ; h0 (may be NULL)
    jmp     .hprev_done
.has_prev:
    mov     rdi, [r15 + 56]
    lea     rbx, [r12 - 1]
    imul    rbx, r13
    shl     rbx, 2
    add     rdi, rbx
.hprev_done:
    mov     [rsp + LOC_HPREV], rdi

    ; Load delta[t]
    mov     rax, [r15 + 40]
    movss   xmm0, [rax + r12*4]        ; xmm0 = delta[t]

    ; ddt accumulator
    xorps   xmm1, xmm1                 ; xmm1 = ddt

    ; ── Inner loop: d = 0 to D-1 ─────────────────────────────────
    xor     ebx, ebx                    ; d = 0

.dim_loop:
    cmp     rbx, r13
    jge     .store_ddt

    ; Load scalars
    mov     rax, [rsp + LOC_DYROW]
    movss   xmm2, [rax + rbx*4]        ; dy

    mov     rax, [rsp + LOC_HROW]
    movss   xmm3, [rax + rbx*4]        ; h[t*D+d]

    mov     rax, [rsp + LOC_XROW]
    movss   xmm4, [rax + rbx*4]        ; x[t*D+d]

    mov     rax, [r15 + 8]
    movss   xmm5, [rax + rbx*4]        ; A[d]

    mov     rax, [r15 + 24]
    movss   xmm6, [rax + rbx*4]        ; B[d]

    mov     rax, [r15 + 32]
    movss   xmm7, [rax + rbx*4]        ; C[d]

    movss   xmm8, [r14 + rbx*4]        ; adj_h[d]

    ; h_prev[d]
    mov     rax, [rsp + LOC_HPREV]
    test    rax, rax
    jz      .zero_hp
    movss   xmm9, [rax + rbx*4]
    jmp     .hp_ok
.zero_hp:
    xorps   xmm9, xmm9
.hp_ok:

    ; dA_val = A_diag[t*D+d] or Taylor exp
    mov     rax, [rsp + LOC_ADIAG]
    test    rax, rax
    jz      .taylor_scl
    movss   xmm10, [rax + rbx*4]
    jmp     .da_ok
.taylor_scl:
    ; exp(x) ≈ 1 + x + x²/2 + x³/6 where x = delta*A
    vmulss  xmm10, xmm0, xmm5          ; x = delta * A
    vmovss  xmm11, xmm10               ; save x
    vmulss  xmm12, xmm10, xmm10        ; x²
    movss   xmm13, [rel bwd_s_half]
    vmulss  xmm13, xmm12, xmm13        ; x²/2
    vaddss  xmm10, xmm10, xmm13        ; x + x²/2
    vmulss  xmm13, xmm12, xmm11        ; x³
    movss   xmm14, [rel bwd_s_sixth]
    vmulss  xmm13, xmm13, xmm14        ; x³/6
    vaddss  xmm10, xmm10, xmm13        ; x + x²/2 + x³/6
    movss   xmm13, [rel bwd_s_one]
    vaddss  xmm10, xmm10, xmm13        ; 1 + ...
.da_ok:

    ; ah = adj_h[d] + dy * C[d]
    vmulss  xmm11, xmm2, xmm7          ; dy * C
    vaddss  xmm11, xmm11, xmm8         ; ah

    ; dC[d] += dy * h
    mov     rax, [r15 + 96]
    movss   xmm12, [rax + rbx*4]
    vfmadd231ss xmm12, xmm2, xmm3
    movss   [rax + rbx*4], xmm12

    ; dB[d] += ah * delta * x
    mov     rax, [r15 + 88]
    movss   xmm12, [rax + rbx*4]
    vmulss  xmm13, xmm11, xmm0         ; ah * delta
    vfmadd231ss xmm12, xmm13, xmm4
    movss   [rax + rbx*4], xmm12

    ; dA[d] += ah * delta * dA_val * h_prev
    mov     rax, [r15 + 80]
    movss   xmm12, [rax + rbx*4]
    vmulss  xmm13, xmm11, xmm0         ; ah * delta
    vmulss  xmm13, xmm13, xmm10        ; * dA_val
    vfmadd231ss xmm12, xmm13, xmm9
    movss   [rax + rbx*4], xmm12

    ; dx[t*D+d] = ah * delta * B
    mov     rax, [rsp + LOC_DXROW]
    vmulss  xmm12, xmm11, xmm0
    vmulss  xmm12, xmm12, xmm6
    movss   [rax + rbx*4], xmm12

    ; ddt += ah * (A * dA_val * h_prev + B * x)
    vmulss  xmm12, xmm5, xmm10         ; A * dA_val
    vmulss  xmm12, xmm12, xmm9         ; * h_prev
    vfmadd231ss xmm12, xmm6, xmm4      ; + B * x
    vmulss  xmm13, xmm11, xmm12        ; ah * (...)
    vaddss  xmm1, xmm1, xmm13

    ; adj_h[d] = ah * dA_val
    vmulss  xmm12, xmm11, xmm10
    movss   [r14 + rbx*4], xmm12

    inc     ebx
    jmp     .dim_loop

    ; ── Store ddelta[t] ───────────────────────────────────────────
.store_ddt:
    mov     rax, [r15 + 104]
    addss   xmm1, [rax + r12*4]
    movss   [rax + r12*4], xmm1

    dec     r12
    jmp     .time_loop

    ; ── Cleanup ───────────────────────────────────────────────────
.free_adj_h:
    mov     rdi, r14
    call    free

.epilogue:
    add     rsp, LOCALS
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

section .rodata
bwd_s_half:   dd 0.5
bwd_s_sixth:  dd 0.16666667            ; 1/6
bwd_s_one:    dd 1.0

section .note.GNU-stack noalloc noexec nowrite progbits
