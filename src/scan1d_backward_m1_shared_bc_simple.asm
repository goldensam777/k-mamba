; scan1d_backward_m1_shared_bc.asm - Version simplifiée et corrigée
; 
; NASM implementation of the critical backward pass kernel
; Optimized for M=1 with shared B/C computation
; Target: x86-64 System V ABI, AVX2

%include "types.inc"

extern calloc
extern memset  
extern free

section .text
global scan1d_backward_m1_shared_bc_simple_asm

; extern void scan1d_backward_m1_shared_bc_simple_asm(ScanBackwardSharedParams *p);
; Arguments:
;   rdi = pointer to ScanBackwardSharedParams structure

scan1d_backward_m1_shared_bc_simple_asm:
    ; Prologue
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    
    ; Save parameter pointer
    mov     rbx, rdi                    ; rbx = p
    
    ; Validate parameters
    test    rdi, rdi
    jz      .cleanup
    
    ; Load structure fields
    mov     r8,  [rdi]                  ; r8  = p->x
    mov     r9,  [rdi+8]                ; r9  = p->A  
    mov     r10, [rdi+16]               ; r10 = p->A_diag
    mov     r11, [rdi+24]               ; r11 = p->B
    mov     r12, [rdi+32]               ; r12 = p->C
    push    qword [rdi+40]              ; p->delta
    push    qword [rdi+48]              ; p->h0
    push    qword [rdi+56]              ; p->h
    push    qword [rdi+64]              ; p->dy
    push    qword [rdi+72]              ; p->dx
    push    qword [rdi+80]              ; p->dA
    push    qword [rdi+88]              ; p->dB
    push    qword [rdi+96]              ; p->dC
    push    qword [rdi+104]             ; p->ddelta
    
    ; Load dimensions
    mov     rbp, [rdi+112]             ; rbp = p->L
    mov     rdx, [rdi+120]             ; rdx = p->D
    
    ; Allocate adj_h array
    mov     rdi, rdx                    ; count = D
    mov     rsi, 4                      ; size = 4 (float)
    call    calloc
    mov     r14, rax                    ; r14 = adj_h
    test    rax, rax
    jz      .cleanup
    
    ; Zero output arrays
    push    rdx                         ; save D
    push    rbp                         ; save L
    
    ; Zero dx
    mov     rdi, [rbx+64]               ; p->dx
    mov     rcx, rbp
    imul    rcx, rdx                    ; L * D
    shl     rcx, 2                      ; * 4 bytes
    xor     esi, esi
    call    memset
    
    ; Zero dA
    mov     rdi, [rbx+72]               ; p->dA
    mov     rcx, [rsp+8]                ; restore D
    shl     rcx, 2
    call    memset
    
    ; Zero dB
    mov     rdi, [rbx+80]               ; p->dB
    mov     rcx, [rsp+8]                ; D
    shl     rcx, 2
    call    memset
    
    ; Zero dC
    mov     rdi, [rbx+88]               ; p->dC
    mov     rcx, [rsp+8]                ; D
    shl     rcx, 2
    call    memset
    
    ; Zero ddelta
    mov     rdi, [rbx+96]               ; p->ddelta
    mov     rcx, [rsp+16]               ; restore L
    shl     rcx, 2
    call    memset
    
    pop     rbp                         ; restore L
    pop     rdx                         ; restore D
    
    ; Main backward loop: for t = L-1 downto 0
    mov     rcx, rbp                    ; rcx = t counter
    dec     rcx                         ; start at L-1
    
.time_loop:
    test    rcx, rcx
    jl      .cleanup
    
    ; Load delta[t] and broadcast
    mov     r13, [rsp+40]             ; r13 = p->delta
    movss   xmm0, [r13 + rcx*4]         ; delta[t]
    
    ; Calculate base offset
    mov     rax, rcx                    ; t
    imul    rax, rdx                    ; t * D
    shl     rax, 2                      ; bytes
    
    ; Pointers to current row
    mov     r15, [rsp+32]             ; r15 = p->h
    lea     rsi, [r15 + rax]            ; &h[t*D]
    lea     rdi, [r8 + rax]             ; &x[t*D]
    mov     r13, [rsp+48]              ; r13 = p->dy
    lea     r13, [r13 + rax]            ; &dy[t*D]
    
    ; Previous h pointer
    test    rcx, rcx
    jnz     .has_prev_h
    test    r13, r13                    ; h0
    jz      .zero_prev_h
    mov     r13, r14                    ; use h0
    jmp     .prev_h_ready
.has_prev_h:
    mov     r13, rax                    ; t*D
    sub     r13, rdx                    ; (t-1)*D
    shl     r13, 2                      ; *4 bytes
    lea     r13, [r15 + r13]            ; &h[(t-1)*D]
    jmp     .prev_h_ready
.zero_prev_h:
    xor     r13, r13
.prev_h_ready:

    ; Scalar loop (for correctness)
    mov     rbx, 0                      ; d counter
.scalar_loop:
    cmp     rbx, rdx
    jge     .scalar_done
    
    ; Load values
    movss   xmm1, [rsi + rbx*4]         ; h[t,d]
    movss   xmm2, [rdi + rbx*4]         ; x[t,d]
    movss   xmm3, [r9 + rbx*4]          ; A[d]
    movss   xmm4, [r11 + rbx*4]         ; B[d]
    movss   xmm5, [r12 + rbx*4]         ; C[d]
    movss   xmm6, [r14 + rbx*4]         ; adj_h[d]
    movss   xmm7, [r13 + rbx*4]         ; dy[t,d] (correction!)
    
    ; Compute dA = exp(delta * A) (simplified for now)
    vmulss  xmm8, xmm0, xmm3            ; delta * A
    vaddss  xmm8, xmm8, xmm3            ; A + delta*A (approx exp)
    
    ; Compute ah = adj_h + dy * c
    vmulss  xmm9, xmm7, xmm5            ; dy * c
    vaddss  xmm9, xmm9, xmm6            ; ah
    
    ; dC += dy * h
    vmulss  xmm10, xmm7, xmm1           ; dy * h
    mov     r15, [rsp+8]                ; p->dC
    movss   xmm11, [r15 + rbx*4]        ; load dC
    vaddss  xmm10, xmm11, xmm10          ; dC + dy*h
    movss   [r15 + rbx*4], xmm10         ; store dC
    
    ; dB += ah * delta * x
    vmulss  xmm10, xmm9, xmm0           ; ah * delta
    vmulss  xmm10, xmm10, xmm2          ; * x
    mov     r15, [rsp]                  ; p->dB
    movss   xmm11, [r15 + rbx*4]        ; load dB
    vaddss  xmm10, xmm11, xmm10          ; dB + ah*delta*x
    movss   [r15 + rbx*4], xmm10         ; store dB
    
    ; dA += ah * delta * dA * h_prev
    vmulss  xmm10, xmm9, xmm0           ; ah * delta
    vmulss  xmm10, xmm10, xmm8          ; * dA
    vmulss  xmm10, xmm10, xmm1          ; * h_prev
    mov     r15, [rsp-8]               ; p->dA
    movss   xmm11, [r15 + rbx*4]        ; load dA
    vaddss  xmm10, xmm11, xmm10          ; dA + ah*delta*dA*h_prev
    movss   [r15 + rbx*4], xmm10         ; store dA
    
    ; dx += ah * delta * b
    vmulss  xmm10, xmm9, xmm0           ; ah * delta
    vmulss  xmm10, xmm10, xmm4          ; * b
    mov     r15, [rsp-16]              ; p->dx
    lea     r15, [r15 + rax]            ; r15 = &dx[t*D]
    movss   xmm11, [r15 + rbx*4]        ; load dx
    vaddss  xmm10, xmm11, xmm10          ; dx + ah*delta*b
    movss   [r15 + rbx*4], xmm10         ; store dx
    
    ; Update adj_h = ah * dA
    vmulss  xmm6, xmm9, xmm8            ; adj_h = ah * dA
    movss   [r14 + rbx*4], xmm6         ; store adj_h
    
    inc     rbx
    jmp     .scalar_loop
    
.scalar_done:
    dec     rcx
    jmp     .time_loop
    
.cleanup:
    ; Epilogue
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
