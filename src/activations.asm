; activations.asm — Fonctions d'activation (float32)
; relu_f32, sigmoid_f32, silu_f32, softplus_f32
; rdi=x, rsi=y, rdx=n

%include "types.inc"

section .data
    ONE_F   dd  1.0
    NEG_ONE dd -1.0
    ZERO_F  dd  0.0

section .text
    global relu_f32
    global sigmoid_f32
    global silu_f32
    global softplus_f32
    extern expf
    extern logf

; ---- relu_f32 : y[i] = max(0, x[i]) ----
relu_f32:
    push rbp
    mov  rbp, rsp
    push r12
    push r13
    push r14
    push r15
    sub  rsp, 8
    mov  r12, rdi
    mov  r13, rsi
    mov  r14, rdx
    xor  r15, r15

.relu_loop:
    cmp  r15, r14
    jge  .relu_done
    movss xmm0, [r12 + r15*FLOAT32_SIZE]
    maxss xmm0, [ZERO_F]
    movss [r13 + r15*FLOAT32_SIZE], xmm0
    inc  r15
    jmp  .relu_loop

.relu_done:
    add  rsp, 8
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbp
    ret

; ---- sigmoid_f32 : y[i] = 1/(1+expf(-x[i])) ----
sigmoid_f32:
    push rbp
    mov  rbp, rsp
    push r12
    push r13
    push r14
    push r15
    sub  rsp, 8
    mov  r12, rdi
    mov  r13, rsi
    mov  r14, rdx
    xor  r15, r15

.sig_loop:
    cmp  r15, r14
    jge  .sig_done
    movss xmm0, [r12 + r15*FLOAT32_SIZE]
    mulss xmm0, [NEG_ONE]
    call expf                   ; xmm0 = expf(-x[i])
    addss xmm0, [ONE_F]
    movss xmm1, [ONE_F]
    divss xmm1, xmm0
    movss [r13 + r15*FLOAT32_SIZE], xmm1
    inc  r15
    jmp  .sig_loop

.sig_done:
    add  rsp, 8
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbp
    ret

; ---- silu_f32 : y[i] = x[i]*sigmoid(x[i]) ----
silu_f32:
    push rbp
    mov  rbp, rsp
    push r12
    push r13
    push r14
    push r15
    sub  rsp, 16
    mov  r12, rdi
    mov  r13, rsi
    mov  r14, rdx
    xor  r15, r15

.silu_loop:
    cmp  r15, r14
    jge  .silu_done
    movss xmm0, [r12 + r15*FLOAT32_SIZE]
    movss [rbp-48], xmm0        ; sauvegarder x[i]
    mulss xmm0, [NEG_ONE]
    call expf
    addss xmm0, [ONE_F]
    movss xmm1, [ONE_F]
    divss xmm1, xmm0            ; sigmoid
    movss xmm0, [rbp-48]        ; x[i]
    mulss xmm0, xmm1
    movss [r13 + r15*FLOAT32_SIZE], xmm0
    inc  r15
    jmp  .silu_loop

.silu_done:
    add  rsp, 16
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbp
    ret

; ---- softplus_f32 : y[i] = logf(1+expf(x[i])) ----
softplus_f32:
    push rbp
    mov  rbp, rsp
    push r12
    push r13
    push r14
    push r15
    sub  rsp, 8
    mov  r12, rdi
    mov  r13, rsi
    mov  r14, rdx
    xor  r15, r15

.sfp_loop:
    cmp  r15, r14
    jge  .sfp_done
    movss xmm0, [r12 + r15*FLOAT32_SIZE]
    call expf
    addss xmm0, [ONE_F]
    call logf
    movss [r13 + r15*FLOAT32_SIZE], xmm0
    inc  r15
    jmp  .sfp_loop

.sfp_done:
    add  rsp, 8
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
