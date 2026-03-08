; hadamard.asm — Produit de Hadamard (float32)
; z[i] = x[i] * y[i]
; rdi=x, rsi=y, rdx=z, rcx=n

%include "types.inc"

section .text
    global hadamard
    global hadamard_avx2

hadamard:
    push rbp
    mov  rbp, rsp
    xor  rax, rax

.loop:
    cmp  rax, rcx
    jge  .fin
    movss xmm0, [rdi + rax*FLOAT32_SIZE]
    mulss xmm0, [rsi + rax*FLOAT32_SIZE]
    movss [rdx + rax*FLOAT32_SIZE], xmm0
    inc  rax
    jmp  .loop

.fin:
    pop  rbp
    ret

hadamard_avx2:
    push rbp
    mov  rbp, rsp
    push rbx
    mov  rbx, rcx
    and  rbx, ~7        ; n8
    xor  rax, rax

.vec:
    cmp  rax, rbx
    jge  .scal
    vmovups ymm0, [rdi + rax*FLOAT32_SIZE]
    vmulps  ymm0, ymm0, [rsi + rax*FLOAT32_SIZE]
    vmovups [rdx + rax*FLOAT32_SIZE], ymm0
    add  rax, 8
    jmp  .vec

.scal:
    cmp  rax, rcx
    jge  .fin2
    movss xmm0, [rdi + rax*FLOAT32_SIZE]
    mulss xmm0, [rsi + rax*FLOAT32_SIZE]
    movss [rdx + rax*FLOAT32_SIZE], xmm0
    inc  rax
    jmp  .scal

.fin2:
    vzeroupper
    pop  rbx
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
