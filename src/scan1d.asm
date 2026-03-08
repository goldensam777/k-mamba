; ============================================================
; scan1d.asm — Scan sélectif 1D (scalaire, float32)
;
; Pour chaque position t et canal d :
;   dt      = delta[t, d]
;   dA[m]   = expf(dt * A[d, m])          — discrétisation
;   dB[m]   = dt * B[t, d, m]
;   h[d,m]  = dA[m] * h[d,m] + dB[m] * x[t,d]   — mise à jour état
;   y[t,d]  = Σ_m C[t,d,m] * h[d,m]              — sortie
;
; Interface :
;   rdi = pointeur ScanParams*
;
; Registres callee-saved (survivent aux appels expf) :
;   r12 = ScanParams*
;   r13 = t
;   r14 = d
;   r15 = m
;   rbx = dm_idx = d*M + m
;
; Variables locales (offset depuis rbp après 5 pushes) :
;   [rbp-48] : y_accum (float)
;   [rbp-56] : td_idx  (long)
;   [rbp-64] : dm_base (long, d*M)
; ============================================================

%include "types.inc"
%include "scan.inc"

section .text
    global scan1d
    extern expf

scan1d:
    push rbp
    mov  rbp, rsp
    push rbx                ; [rbp - 8]
    push r12                ; [rbp - 16]
    push r13                ; [rbp - 24]
    push r14                ; [rbp - 32]
    push r15                ; [rbp - 40]
    sub  rsp, 56            ; [rbp-48]..[rbp-96], aligné 16 pour appels expf

    mov  r12, rdi           ; ScanParams*
    xor  r13, r13           ; t = 0

.loop_t:
    cmp  r13, [r12 + ScanParams.L]
    jge  .done

    xor  r14, r14           ; d = 0

.loop_d:
    cmp  r14, [r12 + ScanParams.D]
    jge  .next_t

    ; td_idx = t * D + d
    mov  rax, r13
    imul rax, [r12 + ScanParams.D]
    add  rax, r14
    mov  [rbp-56], rax      ; sauvegarder td_idx

    ; dm_base = d * M
    mov  rbx, r14
    imul rbx, [r12 + ScanParams.M]
    mov  [rbp-64], rbx      ; sauvegarder dm_base

    ; y_accum = 0.0
    xorps xmm0, xmm0
    movss [rbp-48], xmm0

    xor  r15, r15           ; m = 0

.loop_m:
    cmp  r15, [r12 + ScanParams.M]
    jge  .store_y

    ; dm_idx = dm_base + m  (survit à l'appel expf car rbx callee-saved)
    mov  rbx, [rbp-64]
    add  rbx, r15

    ; ---- Préparer et appeler expf(dt * A[dm_idx]) ----

    ; dt = delta[td_idx]
    mov  rax, [rbp-56]
    mov  rdx, [r12 + ScanParams.delta]
    movss xmm0, [rdx + rax*FLOAT32_SIZE]   ; xmm0 = dt

    ; a_val = A[dm_idx]
    mov  rdx, [r12 + ScanParams.A]
    movss xmm1, [rdx + rbx*FLOAT32_SIZE]   ; xmm1 = a_val

    mulss xmm0, xmm1                        ; xmm0 = dt * a_val

    call expf                               ; xmm0 = dA = expf(dt * a_val)
    ; Après expf : r12, r13, r14, r15, rbx sont intacts (callee-saved)
    ; Les xmm sont potentiellement clobbered — on recharge depuis mémoire

    ; ---- Recharger les valeurs nécessaires ----

    ; td_idx
    mov  rax, [rbp-56]

    ; dt = delta[td_idx]
    mov  rdx, [r12 + ScanParams.delta]
    movss xmm1, [rdx + rax*FLOAT32_SIZE]   ; xmm1 = dt

    ; b_val = B[td_idx * M + m]
    mov  rcx, rax
    imul rcx, [r12 + ScanParams.M]
    add  rcx, r15
    mov  rdx, [r12 + ScanParams.B]
    movss xmm2, [rdx + rcx*FLOAT32_SIZE]   ; xmm2 = b_val

    mulss xmm2, xmm1                        ; xmm2 = dB = dt * b_val

    ; xt = x[td_idx]
    mov  rdx, [r12 + ScanParams.x]
    movss xmm3, [rdx + rax*FLOAT32_SIZE]   ; xmm3 = xt

    ; h_old = h[dm_idx]
    mov  rdx, [r12 + ScanParams.h]
    movss xmm4, [rdx + rbx*FLOAT32_SIZE]   ; xmm4 = h_old

    ; h_new = dA * h_old + dB * xt
    mulss xmm0, xmm4                        ; xmm0 = dA * h_old
    mulss xmm2, xmm3                        ; xmm2 = dB * xt
    addss xmm0, xmm2                        ; xmm0 = h_new

    ; stocker h[dm_idx] = h_new
    movss [rdx + rbx*FLOAT32_SIZE], xmm0   ; rdx = h pointer

    ; y_accum += C[td_idx * M + m] * h_new
    ; rcx = td_idx * M + m (déjà calculé ci-dessus)
    mov  rdx, [r12 + ScanParams.C]
    movss xmm1, [rdx + rcx*FLOAT32_SIZE]   ; xmm1 = c_val
    mulss xmm1, xmm0                        ; xmm1 = c_val * h_new
    movss xmm2, [rbp-48]
    addss xmm2, xmm1
    movss [rbp-48], xmm2                    ; y_accum mis à jour

    inc  r15
    jmp  .loop_m

.store_y:
    ; y[td_idx] = y_accum
    mov  rax, [rbp-56]
    mov  rdx, [r12 + ScanParams.y]
    movss xmm0, [rbp-48]
    movss [rdx + rax*FLOAT32_SIZE], xmm0

    inc  r14
    jmp  .loop_d

.next_t:
    inc  r13
    jmp  .loop_t

.done:
    add  rsp, 56
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
