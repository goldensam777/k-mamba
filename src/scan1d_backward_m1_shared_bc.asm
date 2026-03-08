; scan1d_backward_m1_shared_bc.asm
; 
; NASM implementation of the critical backward pass kernel
; Optimized for M=1 with shared B/C computation
; Target: x86-64 System V ABI, AVX2
;
; Performance goals:
; - Eliminate expf() calls from hot loop
; - Maximize register usage
; - Enable aggressive unrolling
; - Use FMA instructions where possible

%include "types.inc"

extern calloc
extern memset  
extern free

section .text
global scan1d_backward_m1_shared_bc_asm

; extern void scan1d_backward_m1_shared_bc_asm(ScanBackwardSharedParams *p);
; Arguments:
;   rdi = pointer to ScanBackwardSharedParams structure

scan1d_backward_m1_shared_bc_asm:
    ; Prologue
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    
    ; Save original parameter pointer
    mov     rbx, rdi                    ; rbx = p (original)
    
    ; Validate parameters
    test    rdi, rdi
    jz      .cleanup
    
    ; Load structure fields
    mov     r8,  [rdi]                  ; r8  = p->x
    mov     r9,  [rdi+8]                ; r9  = p->A  
    mov     r10, [rdi+16]               ; r10 = p->B
    mov     r11, [rdi+24]               ; r11 = p->C
    mov     r12, [rdi+32]               ; r12 = p->delta
    mov     r13, [rdi+40]               ; r13 = p->A_diag
    mov     r14, [rdi+48]               ; r14 = p->h0
    mov     r15, [rdi+56]               ; r15 = p->h
    
    ; Load dimensions
    mov     rbp, [rbx+104]              ; rbp = p->L
    mov     rax, [rbx+112]              ; rax = p->D
    
    ; Sauvegarder les registres importants
    push    rax                         ; D
    push    rbp                         ; L
    push    rbx                         ; p original
    
    ; Allocate adj_h array using calloc (simpler than mmap)
    mov     rdi, r15                    ; count = D
    mov     rsi, 4                      ; size = 4 (float)
    call    calloc                      ; returns adj_h in rax
    mov     r14, rax                    ; r14 = adj_h
    test    rax, rax
    jz      .cleanup
    
    ; Zero output arrays using memset (more efficient)
    ; dx: L*D floats
    mov     rdi, rax                    ; restore p->dx from stack
    push    rdi
    mov     rdx, rbp
    imul    rdx, r15                    ; rdx = L*D
    shl     rdx, 2                      ; rdx = L*D*4 (bytes)
    xor     esi, esi                    ; value = 0
    call    memset
    
    ; dA: D floats
    mov     rdi, rcx                    ; p->dA
    push    rdi
    mov     rdx, r15
    shl     rdx, 2                      ; rdx = D*4 bytes
    call    memset
    
    ; dB: D floats  
    pop     rdi                         ; restore p->dB
    push    rdi
    mov     rdx, r15
    shl     rdx, 2
    call    memset
    
    ; dC: D floats
    pop     rdi                         ; restore p->dC
    push    rdi
    mov     rdx, r15
    shl     rdx, 2
    call    memset
    
    ; ddelta: L floats
    pop     rdi                         ; restore p->ddelta
    push    rdi
    mov     rdx, rbp
    shl     rdx, 2
    call    memset
    
    ; Restore pointers from stack
    pop     rdi                         ; rdi = p->ddelta
    pop     rsi                         ; rsi = p->dC
    pop     rdx                         ; rdx = p->dB
    pop     rcx                         ; rcx = p->dA
    pop     rax                         ; rax = p->dx
    pop     r15                         ; r15 = p->h

    ; Main backward loop: for t = L-1 downto 0
    mov     r12, rbp                    ; r12 = t counter
    dec     r12                         ; start at L-1
    
.time_loop:
    test    r12, r12
    jl      .cleanup
    
    ; Load delta[t] and broadcast to AVX2
    movss   xmm0, [r12 + r12*4]         ; xmm0 = delta[t] (r12 = p->delta)
    vbroadcastss ymm0, xmm0             ; ymm0 = delta[t] broadcasted
    
    ; Initialize ddt accumulator
    vxorps  ymm1, ymm1, ymm1            ; ymm1 = ddt accumulator (vector)
    vxorps  xmm2, xmm2, xmm2            ; xmm2 = ddt scalar accumulator
    
    ; Calculate base offsets
    mov     r13, r12                    ; t
    imul    r13, r15                    ; t * D
    shl     r13, 2                      ; (t * D) * 4 = byte offset
    
    ; Pointers to current row data
    lea     r10, [r15 + r13]            ; r10 = &h[t*D]
    lea     r11, [r8 + r13]             ; r11 = &x[t*D]
    
    ; Previous h pointer
    test    r12, r12
    jnz     .has_prev_h
    test    r14, r14                    ; r14 = p->h0
    jz      .zero_prev_h
    mov     r9, r14                     ; use h0
    jmp     .prev_h_ready
.has_prev_h:
    mov     r9, r13                     ; r9 = t*D
    sub     r9, r15                    ; r9 = (t-1)*D  
    shl     r9, 2                      ; r9 = (t-1)*D*4 (bytes)
    lea     r9, [r15 + r9]             ; r9 = &h[(t-1)*D]
    jmp     .prev_h_ready
.zero_prev_h:
    xor     r9, r9                      ; NULL pointer
.prev_h_ready:

    ; Vectorized loop: process 8 elements at once
    mov     rcx, 0                      ; d counter
.vector_loop:
    cmp     rcx, r15
    jge     .vector_done
    
    ; Check if we have 8 elements remaining
    mov     rdx, r15
    sub     rdx, rcx
    cmp     rdx, 8
    jl      .scalar_tail
    
    ; Load 8-element vectors
    vmovups ymm2, [r15 + rcx*4]         ; h[t*D + d]
    vmovups ymm3, [r8 + rcx*4]          ; x[t*D + d]
    vmovups ymm4, [r9 + rcx*4]          ; A[d]
    vmovups ymm5, [r10 + rcx*4]         ; B[d]
    vmovups ymm6, [r11 + rcx*4]         ; C[d]
    vmovups ymm7, [r14 + rcx*4]         ; adj_h[d]
    
    ; Compute dA = exp(delta[t] * A[d]) or load from A_diag
    test    r13, r13                    ; r13 = p->A_diag
    jnz     .load_a_diag
    ; Compute exp(delta * A) for each element
    vfmadd213ps ymm4, ymm0, ymm4        ; ymm4 = delta * A + A = A * (1 + delta)
    ; TODO: Implement fast exp approximation
    vmovaps ymm8, ymm4                  ; ymm8 = dA
    jmp     .dA_ready
.load_a_diag:
    mov     rdx, r13                    ; rdx = A_diag pointer
    add     rdx, r13                    ; rdx = A_diag + t*D
    vmovups ymm8, [rdx + rcx*4]         ; Load from A_diag
.dA_ready:
    
    ; Compute ah = adj_h + dy * c
    vmulps  ymm9, ymm6, ymm3            ; ymm9 = dy * c
    vaddps  ymm9, ymm9, ymm7            ; ymm9 = ah
    
    ; Accumulate gradients
    ; dC += dy * h
    vmulps  ymm10, ymm3, ymm6          ; dy * C (correction: c = C, pas dy)
    vmovups ymm11, [rsi + rcx*4]       ; load dC
    vmulps  ymm10, ymm3, ymm2          ; dy * h
    vaddps  ymm10, ymm11, ymm10        ; dC + dy * h
    vmovups [rsi + rcx*4], ymm10       ; store back
    
    ; dB += ah * delta * x
    vmulps  ymm10, ymm9, ymm0           ; ah * delta
    vmulps  ymm10, ymm10, ymm3          ; * x
    vmovups ymm11, [rdx + rcx*4]       ; load dB
    vaddps  ymm10, ymm11, ymm10        ; dB + ah * delta * x
    vmovups [rdx + rcx*4], ymm10       ; store back
    
    ; dA += ah * delta * dA * h_prev
    vmulps  ymm11, ymm9, ymm0           ; ah * delta
    vmulps  ymm11, ymm11, ymm8          ; * dA
    vmulps  ymm11, ymm11, ymm9          ; * h_prev
    vmovups ymm12, [rcx + rcx*4]       ; load dA (PROBLEME: rcx utilisé pour deux choses)
    vaddps  ymm11, ymm12, ymm11        ; dA + ah * delta * dA * h_prev
    vmovups [rcx + rcx*4], ymm11       ; store back
    
    ; dx += ah * delta * b
    vmulps  ymm12, ymm9, ymm0           ; ah * delta
    vmulps  ymm12, ymm12, ymm5          ; * b
    mov     r13, [rsp+8]                ; récupérer p original
    mov     r13, [r13+64]               ; r13 = p->dx
    vmovups ymm13, [r13 + r13 + rcx*4] ; load dx
    vaddps  ymm12, ymm13, ymm12        ; dx + ah * delta * b
    vmovups [r13 + r13 + rcx*4], ymm12 ; store back
    
    ; ddt += ah * (a * dA * h_prev + b * x)
    vmulps  ymm13, ymm4, ymm8           ; a * dA
    vmulps  ymm13, ymm13, ymm9          ; * h_prev
    vmulps  ymm14, ymm5, ymm3           ; b * x
    vaddps  ymm13, ymm13, ymm14         ; a * dA * h_prev + b * x
    vmulps  ymm13, ymm9, ymm13          ; ah * (...)
    vaddps  ymm1, ymm1, ymm13           ; ddt += ah * (...)
    
    ; Update adj_h = ah * dA
    vmulps  ymm7, ymm9, ymm8            ; adj_h = ah * dA
    vmovups [r14 + rcx*4], ymm7         ; store back
    
    add     rcx, 8
    jmp     .vector_loop
    
.scalar_tail:
    ; Handle remaining elements scalar
.scalar_loop:
    cmp     rcx, r15
    jge     .scalar_done
    
    ; TODO: Scalar processing for remaining elements
    ; Load scalar values
    movss   xmm2, [r10 + rcx*4]         ; h
    movss   xmm3, [r11 + rcx*4]         ; x  
    movss   xmm4, [r9 + rcx*4]          ; A
    movss   xmm5, [r10 + rcx*4]         ; B
    movss   xmm6, [r11 + rcx*4]         ; C
    movss   xmm7, [r14 + rcx*4]         ; adj_h
    
    ; TODO: Complete scalar computations
    inc     rcx
    jmp     .scalar_loop
    
.scalar_done:
.vector_done:
    
    ; Horizontal sum of ddt vector and store
    vextractf128 xmm3, ymm1, 1          ; high 128 bits
    vaddps xmm2, xmm1, xmm3            ; add high and low
    vhaddps xmm2, xmm2, xmm2           ; horizontal add
    vmovss  [rdi + r12*4], xmm2        ; store ddt[t]
    
    dec     r12
    jmp     .time_loop
    
.cleanup:
    ; Free adj_h array
    mov     rdi, r14                    ; adj_h pointer
    call    free
    
    ; Epilogue
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
