; gemv.asm — Produit Matrice × Vecteur (scalaire, float32)
; y = A · x
; rdi=A (m×n float32), rsi=x (n), rdx=y (m), rcx=m, r8=n

%include "types.inc"

section .text
    global gemv

gemv:
    push rbp
    mov  rbp, rsp
    xor  rax, rax           ; i = 0

.boucle_i:
    cmp  rax, rcx
    jge  .fin
    xorps xmm0, xmm0        ; accum = 0.0f
    xor  r9, r9             ; j = 0

.boucle_j:
    cmp  r9, r8
    jge  .ecrire
    mov  r10, rax
    imul r10, r8
    add  r10, r9
    movss xmm1, [rdi + r10*FLOAT32_SIZE]
    movss xmm2, [rsi + r9*FLOAT32_SIZE]
    mulss xmm1, xmm2
    addss xmm0, xmm1
    inc  r9
    jmp  .boucle_j

.ecrire:
    movss [rdx + rax*FLOAT32_SIZE], xmm0
    inc  rax
    jmp  .boucle_i

.fin:
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
