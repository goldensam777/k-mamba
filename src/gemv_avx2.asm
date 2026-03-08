; gemv_avx2.asm — GEMV vectorisé AVX2 (8 float32 en parallèle)
; y = A · x
; rdi=A, rsi=x, rdx=y, rcx=m, r8=n

%include "types.inc"

section .text
    global gemv_avx2

gemv_avx2:
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
    mov  r15, rcx
    ; n8 = n & ~7
    mov  rbx, r8
    and  rbx, ~7
    xor  rcx, rcx       ; i = 0

.boucle_i:
    cmp  rcx, r15
    jge  .fin
    vxorps ymm0, ymm0, ymm0    ; accum = 0.0f (8 floats)
    ; &A[i][0]
    mov  rax, rcx
    imul rax, r8
    lea  r9, [r12 + rax*FLOAT32_SIZE]
    xor  r10, r10       ; j = 0

.vec:
    cmp  r10, rbx
    jge  .scal
    vmovups ymm1, [r9 + r10*FLOAT32_SIZE]
    vmovups ymm2, [r13 + r10*FLOAT32_SIZE]
    vfmadd231ps ymm0, ymm1, ymm2
    add  r10, 8
    jmp  .vec

.scal:
    cmp  r10, r8
    jge  .reduire
    movss xmm1, [r9 + r10*FLOAT32_SIZE]
    movss xmm2, [r13 + r10*FLOAT32_SIZE]
    mulss xmm1, xmm2
    vaddss xmm0, xmm0, xmm1
    inc  r10
    jmp  .scal

.reduire:
    ; ymm0 = [a,b,c,d,e,f,g,h] → xmm0 = a+b+c+d+e+f+g+h
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    movss [r14 + rcx*FLOAT32_SIZE], xmm0
    inc  rcx
    jmp  .boucle_i

.fin:
    vzeroupper
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
