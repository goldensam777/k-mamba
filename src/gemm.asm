; gemm.asm — Produit Matrice × Matrice (scalaire, float32)
; C = A · B
; rdi=A (m×k), rsi=B (k×n), rdx=C (m×n), rcx=m, r8=k, r9=n

%include "types.inc"

section .text
    global gemm

gemm:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov  r12, rdi
    mov  r13, rsi
    mov  r14, rdx
    push rcx            ; m  [rbp-48]
    push r8             ; k  [rbp-56]
    push r9             ; n  [rbp-64]
    xor  rbx, rbx       ; i = 0

.boucle_i:
    cmp  rbx, [rbp-48]
    jge  .fin
    xor  r15, r15       ; j = 0

.boucle_j:
    cmp  r15, [rbp-64]
    jge  .fin_j
    xorps xmm0, xmm0   ; accum = 0.0f
    xor  rcx, rcx      ; p = 0

.boucle_p:
    cmp  rcx, [rbp-56]
    jge  .ecrire
    mov  rax, rbx
    imul rax, [rbp-56]
    add  rax, rcx
    movss xmm1, [r12 + rax*FLOAT32_SIZE]
    mov  rax, rcx
    imul rax, [rbp-64]
    add  rax, r15
    movss xmm2, [r13 + rax*FLOAT32_SIZE]
    mulss xmm1, xmm2
    addss xmm0, xmm1
    inc  rcx
    jmp  .boucle_p

.ecrire:
    mov  rax, rbx
    imul rax, [rbp-64]
    add  rax, r15
    movss [r14 + rax*FLOAT32_SIZE], xmm0
    inc  r15
    jmp  .boucle_j

.fin_j:
    inc  rbx
    jmp  .boucle_i

.fin:
    pop  r9
    pop  r8
    pop  rcx
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
