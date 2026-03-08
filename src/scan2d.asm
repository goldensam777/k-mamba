; ============================================================
; scan2d.asm — Scan sélectif 2D avec ordonnancement Wavefront (float32)
;
; Récurrence :
;   h(i,j,d,m) = dA1 * h(i-1,j,d,m)
;              + dA2 * h(i,j-1,d,m)
;              + dB  * x(i,j,d)
;   y(i,j,d)   = Σ_m C(i,j,d,m) * h(i,j,d,m)
;
; Ordonnancement Wavefront (diagonal k = i+j) :
;   k=0 : (0,0)
;   k=1 : (1,0), (0,1)        ← indépendantes
;   k=2 : (2,0), (1,1), (0,2) ← indépendantes
;   ...
;
; Interface :
;   rdi = Scan2DParams*
;
; Allocation des registres callee-saved :
;   r12 = Scan2DParams*
;   r13 = d (canal)
;   r14 = m (état)
;   r15 = pos_idx = i*d2 + j
;   rbx = dm_idx = d*M + m  (dans la boucle M)
;
; Variables locales sur la pile :
;   [rbp-48] : k        (diagonal courant)
;   [rbp-56] : i        (courant dans la diagonale)
;   [rbp-64] : i_max
;   [rbp-72] : j        (= k - i)
;   [rbp-80] : y_accum  (float)
;   [rbp-88] : h_prev1  (float, sauvegarde xmm5)
;   [rbp-96] : h_prev2  (float, sauvegarde xmm6)
; ============================================================

%include "types.inc"
%include "scan.inc"

section .data
    HALF    dd 0.5          ; constante 0.5 pour le calcul de dB

section .text
    global scan2d
    extern expf

scan2d:
    push rbp
    mov  rbp, rsp
    push rbx                ; [rbp-8]
    push r12                ; [rbp-16]
    push r13                ; [rbp-24]
    push r14                ; [rbp-32]
    push r15                ; [rbp-40]
    sub  rsp, 64            ; locaux [rbp-48]..[rbp-96], stack alignée 16

    mov  r12, rdi           ; Scan2DParams*

    ; k = 0 — stocker sur pile (rbx réservé pour dm_idx)
    mov  qword [rbp-48], 0

; ===========================================================
; Boucle externe Wavefront : diagonale k = 0 .. d1+d2-2
; ===========================================================
.loop_k:
    ; limite = d1 + d2 - 1
    mov  rax, [r12 + Scan2DParams.d1]
    add  rax, [r12 + Scan2DParams.d2]
    dec  rax
    cmp  [rbp-48], rax
    jge  .done

    ; ---- i_min = max(0, k - d2 + 1) ----
    mov  rax, [rbp-48]      ; k
    sub  rax, [r12 + Scan2DParams.d2]
    inc  rax
    test rax, rax
    jns  .i_min_ok
    xor  rax, rax
.i_min_ok:
    mov  [rbp-56], rax      ; i = i_min

    ; ---- i_max = min(k, d1-1) ----
    mov  rax, [rbp-48]      ; k
    mov  rcx, [r12 + Scan2DParams.d1]
    dec  rcx                ; d1-1
    cmp  rax, rcx
    jle  .i_max_ok
    mov  rax, rcx
.i_max_ok:
    mov  [rbp-64], rax      ; i_max

; ===========================================================
; Boucle sur les positions de la diagonale
; ===========================================================
.loop_pos:
    mov  rax, [rbp-56]      ; i
    cmp  rax, [rbp-64]      ; i > i_max ?
    jg   .next_k

    ; j = k - i
    mov  rcx, [rbp-48]      ; k
    sub  rcx, rax           ; j = k - i
    mov  [rbp-72], rcx      ; sauvegarder j

    ; pos_idx = i * d2 + j
    mov  r15, rax
    imul r15, [r12 + Scan2DParams.d2]
    add  r15, rcx           ; r15 = pos_idx

    ; ---- Boucle D (canaux) ----
    xor  r13, r13           ; d = 0

.loop_d:
    cmp  r13, [r12 + Scan2DParams.D]
    jge  .next_pos

    xorps xmm0, xmm0
    movss [rbp-80], xmm0    ; y_accum = 0

    ; ---- Boucle M (dimension état) ----
    xor  r14, r14           ; m = 0

.loop_m:
    cmp  r14, [r12 + Scan2DParams.M]
    jge  .store_y_2d

    ; dm_idx = d*M + m
    mov  rbx, r13
    imul rbx, [r12 + Scan2DParams.M]
    add  rbx, r14           ; rbx = dm_idx

    ; pd_idx = pos_idx * D + d
    mov  rax, r15
    imul rax, [r12 + Scan2DParams.D]
    add  rax, r13           ; rax = pd_idx

    ; DM = D * M
    mov  rcx, [r12 + Scan2DParams.D]
    imul rcx, [r12 + Scan2DParams.M]   ; rcx = DM

    ; pDM_idx = pos_idx * DM + dm_idx
    mov  rdx, r15
    imul rdx, rcx
    add  rdx, rbx           ; rdx = pDM_idx

    ; ---- h_prev1 : h(i-1, j, d, m) — si i > 0 ----
    xorps xmm5, xmm5
    cmp  qword [rbp-56], 0  ; i == 0 ?
    je   .no_prev1
    mov  rax, r15
    sub  rax, [r12 + Scan2DParams.d2]
    imul rax, rcx
    add  rax, rbx                       ; rax = pDM_prev1
    mov  r8, [r12 + Scan2DParams.h]
    movss xmm5, [r8 + rax*FLOAT32_SIZE]
.no_prev1:
    movss [rbp-88], xmm5    ; sauvegarder h_prev1

    ; ---- h_prev2 : h(i, j-1, d, m) — si j > 0 ----
    xorps xmm6, xmm6
    cmp  qword [rbp-72], 0  ; j == 0 ?
    je   .no_prev2
    mov  rax, r15
    dec  rax                ; pos_idx - 1
    imul rax, rcx
    add  rax, rbx           ; pDM_prev2
    mov  r8, [r12 + Scan2DParams.h]
    movss xmm6, [r8 + rax*FLOAT32_SIZE]
.no_prev2:
    movss [rbp-96], xmm6    ; sauvegarder h_prev2

    ; pd_idx (recalcul propre pour éviter conflits)
    mov  rax, r15
    imul rax, [r12 + Scan2DParams.D]
    add  rax, r13           ; rax = pd_idx

    ; ---- expf(dt1 * A1[dm_idx]) → dA1 ----
    mov  r8, [r12 + Scan2DParams.delta1]
    movss xmm0, [r8 + rax*FLOAT32_SIZE]    ; dt1
    mov  r8, [r12 + Scan2DParams.A1]
    movss xmm1, [r8 + rbx*FLOAT32_SIZE]    ; a1_val
    mulss xmm0, xmm1
    call expf               ; xmm0 = dA1
    sub  rsp, 16            ; réserver espace pour dA1
    ; rbx, r12-r15 intacts
    movss [rsp], xmm0       ; sauvegarder dA1

    ; ---- expf(dt2 * A2[dm_idx]) → dA2 ----
    mov  rax, r15
    imul rax, [r12 + Scan2DParams.D]
    add  rax, r13           ; pd_idx
    mov  r8, [r12 + Scan2DParams.delta2]
    movss xmm0, [r8 + rax*FLOAT32_SIZE]    ; dt2
    mov  r8, [r12 + Scan2DParams.A2]
    movss xmm1, [r8 + rbx*FLOAT32_SIZE]    ; a2_val
    mulss xmm0, xmm1
    call expf               ; xmm0 = dA2

    movss xmm7, [rsp]       ; xmm7 = dA1
    add  rsp, 16            ; restaurer la pile

    ; ---- Calculer dB ----
    ; pd_idx
    mov  rax, r15
    imul rax, [r12 + Scan2DParams.D]
    add  rax, r13
    ; dt1 + dt2
    mov  r8, [r12 + Scan2DParams.delta1]
    movss xmm1, [r8 + rax*FLOAT32_SIZE]
    mov  r8, [r12 + Scan2DParams.delta2]
    movss xmm2, [r8 + rax*FLOAT32_SIZE]
    addss xmm1, xmm2
    mulss xmm1, [HALF]      ; (dt1+dt2)/2
    ; pDM_idx (recalcul)
    mov  rcx, [r12 + Scan2DParams.D]
    imul rcx, [r12 + Scan2DParams.M]
    mov  rdx, r15
    imul rdx, rcx
    add  rdx, rbx           ; pDM_idx
    mov  r8, [r12 + Scan2DParams.B]
    movss xmm2, [r8 + rdx*FLOAT32_SIZE]
    mulss xmm2, xmm1        ; xmm2 = dB
    ; x_val
    mov  r8, [r12 + Scan2DParams.x]
    mov  rax, r15
    imul rax, [r12 + Scan2DParams.D]
    add  rax, r13
    movss xmm3, [r8 + rax*FLOAT32_SIZE]    ; x_val

    ; ---- h_new = dA1*h_prev1 + dA2*h_prev2 + dB*x_val ----
    movss xmm4, [rbp-88]    ; h_prev1
    movss xmm5, [rbp-96]    ; h_prev2
    mulss xmm7, xmm4        ; dA1 * h_prev1
    mulss xmm0, xmm5        ; dA2 * h_prev2
    mulss xmm2, xmm3        ; dB * x_val
    addss xmm7, xmm0
    addss xmm7, xmm2        ; xmm7 = h_new

    ; stocker h[pDM_idx] = h_new
    mov  r8, [r12 + Scan2DParams.h]
    movss [r8 + rdx*FLOAT32_SIZE], xmm7

    ; y_accum += C[pDM_idx] * h_new
    mov  r8, [r12 + Scan2DParams.C]
    movss xmm1, [r8 + rdx*FLOAT32_SIZE]
    mulss xmm1, xmm7
    movss xmm2, [rbp-80]
    addss xmm2, xmm1
    movss [rbp-80], xmm2    ; y_accum

    inc  r14
    jmp  .loop_m

.store_y_2d:
    ; y[pd_idx] = y_accum
    mov  rax, r15
    imul rax, [r12 + Scan2DParams.D]
    add  rax, r13
    mov  r8, [r12 + Scan2DParams.y]
    movss xmm0, [rbp-80]
    movss [r8 + rax*FLOAT32_SIZE], xmm0

    inc  r13
    jmp  .loop_d

.next_pos:
    ; i++
    inc  qword [rbp-56]
    jmp  .loop_pos

.next_k:
    inc  qword [rbp-48]     ; k++
    jmp  .loop_k

.done:
    add  rsp, 64
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
