#include "mamba_scan.h"
#include <string.h>

/* ============================================================
 * Mamba Scan 1D CPU Implementation
 * ============================================================ */

void mamba_scan1d_forward(MambaScan1DParams *p) {
    // Externally defined in scan1d.asm
    extern void scan1d(void *params);
    scan1d(p);
}

/* Implémentations placeholder pour éviter les erreurs de compilation */
void mamba_scan1d_backward(void *p) {
    /* Placeholder */
    (void)p;
}

void mamba_scan1d_backward_m1_shared_bc_impl(void *p) {
    /* Placeholder */
    (void)p;
}
