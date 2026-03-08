; ============================================================
; scan.inc — Structures de paramètres pour les scans sélectifs
; ============================================================

%ifndef SCAN_INC
%define SCAN_INC

; ------------------------------------------------------------
; ScanParams — Paramètres du scan sélectif 1D
;
; x     : [L, D]       — entrée
; A     : [D, M]       — log diagonal de A (valeurs négatives)
; B     : [L, D, M]    — matrice d'entrée sélective
; C     : [L, D, M]    — matrice de sortie sélective
; delta : [L, D]       — pas de discrétisation (positif)
; h     : [D, M]       — état caché (entrée/sortie, init à 0)
; y     : [L, D]       — sortie
; ------------------------------------------------------------
struc ScanParams
    .x      resq 1
    .A      resq 1
    .B      resq 1
    .C      resq 1
    .delta  resq 1
    .h      resq 1
    .y      resq 1
    .L      resq 1
    .D      resq 1
    .M      resq 1
endstruc

; ------------------------------------------------------------
; ScanBackwardParams — Paramètres du backward du scan 1D
;
; x       : [L, D]       — entrée du scan
; A       : [D, M]       — paramètres de transition
; B       : [L, D, M]    — matrice d'entrée sélective
; C       : [L, D, M]    — matrice de sortie sélective
; delta   : [L, D]       — pas de discrétisation
; h0      : [D, M]       — état initial optionnel (NULL => zéro)
; h       : [L, D, M]    — états cachés forward après mise à jour
; dy      : [L, D]       — gradient amont sur y
; dx      : [L, D]       — gradient sur x
; dA      : [D, M]       — gradient sur A
; dB      : [L, D, M]    — gradient sur B
; dC      : [L, D, M]    — gradient sur C
; ddelta  : [L, D]       — gradient sur delta
; ------------------------------------------------------------
struc ScanBackwardParams
    .x       resq 1
    .A       resq 1
    .B       resq 1
    .C       resq 1
    .delta   resq 1
    .h0      resq 1
    .h       resq 1
    .dy      resq 1
    .dx      resq 1
    .dA      resq 1
    .dB      resq 1
    .dC      resq 1
    .ddelta  resq 1
    .L       resq 1
    .D       resq 1
    .M       resq 1
endstruc

; ------------------------------------------------------------
; ScanBackwardSharedParams — backward 1D specialise
; Cas M=1 avec B/C partages par canal et delta[t] scalaire.
; ------------------------------------------------------------
struc ScanBackwardSharedParams
    .x       resq 1
    .A       resq 1
    .A_diag   resq 1
    .B       resq 1
    .C       resq 1
    .delta   resq 1
    .h0      resq 1
    .h       resq 1
    .dy      resq 1
    .dx      resq 1
    .dA      resq 1
    .dB      resq 1
    .dC      resq 1
    .ddelta  resq 1
    .L       resq 1
    .D       resq 1
endstruc

; ------------------------------------------------------------
; Scan2DParams — Paramètres du scan sélectif 2D
;
; x      : [d1, d2, D]       — entrée
; A1     : [D, M]            — transition sur l'axe 1
; A2     : [D, M]            — transition sur l'axe 2
; B      : [d1, d2, D, M]   — matrice d'entrée sélective
; C      : [d1, d2, D, M]   — matrice de sortie sélective
; delta1 : [d1, d2, D]      — pas sur l'axe 1
; delta2 : [d1, d2, D]      — pas sur l'axe 2
; h      : [d1, d2, D, M]   — états (toutes positions, init à 0)
; y      : [d1, d2, D]      — sortie
; d1, d2 : dimensions spatiales
; D, M   : dimension modèle, dimension état
; ------------------------------------------------------------
struc Scan2DParams
    .x      resq 1
    .A1     resq 1
    .A2     resq 1
    .B      resq 1
    .C      resq 1
    .delta1 resq 1
    .delta2 resq 1
    .h      resq 1
    .y      resq 1
    .d1     resq 1
    .d2     resq 1
    .D      resq 1
    .M      resq 1
endstruc

%endif
