#include <stdlib.h>
#include <string.h>
#include "optimatrix.h"

void scan1d_backward_m_generic(ScanBackwardMParams *p) {
    if (!p || p->M <= 0) return;
    
    // Pour M=1, utiliser directement la fonction optimisée
    if (p->M == 1) {
        ScanBackwardSharedParams params_m1 = {
            .x = p->x,
            .A = p->A,
            .B = p->B,
            .C = p->C,
            .delta = p->delta,
            .A_diag = NULL,
            .h0 = p->h0,
            .h = p->h,
            .dy = p->dy,
            .dx = p->dx,
            .dA = p->dA,
            .dB = p->dB,
            .dC = p->dC,
            .ddelta = p->ddelta,
            .L = p->L,
            .D = p->D
        };
        scan1d_backward_m1_shared_bc(&params_m1);
        return;
    }
    
    // Pour M>1, boucler sur chaque dimension de l'état
    long stride = p->L * p->D;  // Nombre d'éléments par dimension M
    
    for (long m = 0; m < p->M; m++) {
        // Calculer les offsets pour cette dimension
        long offset = m * stride;
        
        // Créer les paramètres pour M=1
        ScanBackwardSharedParams params_m1 = {
            .x = p->x + offset,
            .A = p->A + offset,
            .B = p->B + offset,
            .C = p->C + offset,
            .delta = p->delta + offset,
            .A_diag = NULL,
            .h0 = p->h0 ? p->h0 + offset : NULL,
            .h = p->h + offset,
            .dy = p->dy + offset,
            .dx = p->dx + offset,
            .dA = p->dA + offset,
            .dB = p->dB + offset,
            .dC = p->dC + offset,
            .ddelta = p->ddelta + offset,
            .L = p->L,
            .D = p->D
        };
        
        // Appeler la fonction M=1 optimisée
        scan1d_backward_m1_shared_bc(&params_m1);
    }
}
