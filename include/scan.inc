; ============================================================
; scan.inc — Structures NASM pour les scans sélectifs k-mamba
; ============================================================

%ifndef SCAN_INC
%define SCAN_INC

; ------------------------------------------------------------
; ScanParams — Paramètres du scan sélectif 1D
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
; ScanBackwardSharedParams — backward 1D M=1 avec B/C partagés
; ------------------------------------------------------------
struc ScanBackwardSharedParams
    .x       resq 1
    .A       resq 1
    .A_diag  resq 1
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
