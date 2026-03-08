; gemm_avx2.asm — GEMM vectorisé AVX2 (8 float32 en parallèle)
; C = A · B  (stratégie outer product)
; rdi=A (m×k), rsi=B (k×n), rdx=C (m×n), rcx=m, r8=k, r9=n

%include "types.inc"

section .text
    global gemm_avx2

gemm_avx2:
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
    ; n8 = n & ~7
    mov  rax, r9
    and  rax, ~7
    push rax            ; n8 [rbp-72]
    xor  rbx, rbx       ; i = 0

.boucle_i:
    cmp  rbx, [rbp-48]
    jge  .fin
    xor  r15, r15       ; p = 0

.boucle_p:
    cmp  r15, [rbp-56]
    jge  .fin_p
    ; broadcast A[i][p]
    mov  rax, rbx
    imul rax, [rbp-56]
    add  rax, r15
    vbroadcastss ymm3, [r12 + rax*FLOAT32_SIZE]
    ; &C[i][0]
    mov  rax, rbx
    imul rax, [rbp-64]
    lea  r10, [r14 + rax*FLOAT32_SIZE]
    ; &B[p][0]
    mov  rax, r15
    imul rax, [rbp-64]
    lea  r11, [r13 + rax*FLOAT32_SIZE]
    xor  rcx, rcx       ; j = 0

.vec:
    cmp  rcx, [rbp-72]
    jge  .scal
    vmovups ymm1, [r10 + rcx*FLOAT32_SIZE]
    vmovups ymm2, [r11 + rcx*FLOAT32_SIZE]
    vfmadd231ps ymm1, ymm3, ymm2
    vmovups [r10 + rcx*FLOAT32_SIZE], ymm1
    add  rcx, 8
    jmp  .vec

.scal:
    cmp  rcx, [rbp-64]
    jge  .fin_j
    movss xmm1, [r10 + rcx*FLOAT32_SIZE]
    movss xmm2, [r11 + rcx*FLOAT32_SIZE]
    mulss xmm2, xmm3
    addss xmm1, xmm2
    movss [r10 + rcx*FLOAT32_SIZE], xmm1
    inc  rcx
    jmp  .scal

.fin_j:
    inc  r15
    jmp  .boucle_p

.fin_p:
    inc  rbx
    jmp  .boucle_i

.fin:
    vzeroupper
    pop  rax
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
