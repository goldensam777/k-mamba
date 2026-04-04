; ============================================================
; types.inc — Types fondamentaux d'optimatrix (float32)
; ============================================================

%ifndef TYPES_INC
%define TYPES_INC

FLOAT32_SIZE  equ 4     ; float simple précision — 32 bits

section .data
    ZERO_F32    dd 0.0
    ONE_F32     dd 1.0

%endif
