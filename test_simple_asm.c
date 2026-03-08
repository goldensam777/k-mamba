#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "optimatrix.h"

int main() {
    printf("Test comparaison C vs ASM\n");
    
    // Test avec les mêmes paramètres que les tests existants
    long L = 4, D = 3;
    
    float *x = malloc(L * D * sizeof(float));
    float *A = malloc(D * sizeof(float));
    float *B = malloc(D * sizeof(float));
    float *C = malloc(D * sizeof(float));
    float *delta = malloc(L * sizeof(float));
    float *h = malloc(L * D * sizeof(float));
    float *dy = malloc(L * D * sizeof(float));
    
    // Résultats C
    float *dx_c = malloc(L * D * sizeof(float));
    float *dA_c = malloc(D * sizeof(float));
    float *dB_c = malloc(D * sizeof(float));
    float *dC_c = malloc(D * sizeof(float));
    float *ddelta_c = malloc(L * sizeof(float));
    
    // Résultats ASM
    float *dx_asm = malloc(L * D * sizeof(float));
    float *dA_asm = malloc(D * sizeof(float));
    float *dB_asm = malloc(D * sizeof(float));
    float *dC_asm = malloc(D * sizeof(float));
    float *ddelta_asm = malloc(L * sizeof(float));
    
    // Initialiser avec les valeurs du test
    for (int i = 0; i < L * D; i++) {
        x[i] = 0.1f * i;
        h[i] = 0.2f * i;
        dy[i] = 0.01f * i;
    }
    
    for (int i = 0; i < L; i++) {
        delta[i] = 0.1f + 0.01f * i;
    }
    
    for (int i = 0; i < D; i++) {
        A[i] = -0.1f + 0.05f * i;
        B[i] = 0.3f + 0.1f * i;
        C[i] = 0.5f + 0.1f * i;
    }
    
    // Paramètres pour version C
    ScanBackwardSharedParams params_c = {
        .x = x, .A = A, .B = B, .C = C, .delta = delta,
        .A_diag = NULL, .h0 = NULL, .h = h, .dy = dy,
        .dx = dx_c, .dA = dA_c, .dB = dB_c, .dC = dC_c, .ddelta = ddelta_c,
        .L = L, .D = D
    };
    
    // Paramètres pour version ASM
    ScanBackwardSharedParams params_asm = {
        .x = x, .A = A, .B = B, .C = C, .delta = delta,
        .A_diag = NULL, .h0 = NULL, .h = h, .dy = dy,
        .dx = dx_asm, .dA = dA_asm, .dB = dB_asm, .dC = dC_asm, .ddelta = ddelta_asm,
        .L = L, .D = D
    };
    
    printf("Appel version C...\n");
    scan1d_backward_m1_shared_bc(&params_c);
    printf("Version C terminée\n");
    
    printf("Appel version ASM...\n");
    scan1d_backward_m1_shared_bc_simple_asm(&params_asm);
    printf("Version ASM terminée\n");
    
    // Comparer les résultats
    printf("\nComparaison dC:\n");
    for (int i = 0; i < D; i++) {
        float diff = dC_c[i] - dC_asm[i];
        printf("  dC[%d] C=%.6f ASM=%.6f diff=%.6f\n", i, dC_c[i], dC_asm[i], diff);
    }
    
    printf("\nComparaison dB:\n");
    for (int i = 0; i < D; i++) {
        float diff = dB_c[i] - dB_asm[i];
        printf("  dB[%d] C=%.6f ASM=%.6f diff=%.6f\n", i, dB_c[i], dB_asm[i], diff);
    }
    
    printf("\nComparaison dA:\n");
    for (int i = 0; i < D; i++) {
        float diff = dA_c[i] - dA_asm[i];
        printf("  dA[%d] C=%.6f ASM=%.6f diff=%.6f\n", i, dA_c[i], dA_asm[i], diff);
    }
    
    // Nettoyer
    free(x); free(A); free(B); free(C); free(delta); free(h); free(dy);
    free(dx_c); free(dA_c); free(dB_c); free(dC_c); free(ddelta_c);
    free(dx_asm); free(dA_asm); free(dB_asm); free(dC_asm); free(ddelta_asm);
    
    return 0;
}
