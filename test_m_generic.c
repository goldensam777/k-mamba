#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "optimatrix.h"

int main() {
    printf("Test M>1 générique\n");
    
    // Test avec M=2, L=4, D=3
    long L = 4, D = 3, M = 2;
    long stride = L * D;  // 12 éléments par dimension M
    
    // Allouer les tableaux
    float *x = malloc(M * stride * sizeof(float));
    float *A = malloc(M * stride * sizeof(float));
    float *B = malloc(M * stride * sizeof(float));
    float *C = malloc(M * stride * sizeof(float));
    float *delta = malloc(M * stride * sizeof(float));
    float *h = malloc(M * stride * sizeof(float));
    float *dy = malloc(M * stride * sizeof(float));
    
    float *dx = malloc(M * stride * sizeof(float));
    float *dA = malloc(M * stride * sizeof(float));
    float *dB = malloc(M * stride * sizeof(float));
    float *dC = malloc(M * stride * sizeof(float));
    float *ddelta = malloc(M * stride * sizeof(float));
    
    // Initialiser avec des valeurs différentes pour chaque dimension M
    for (long m = 0; m < M; m++) {
        for (long i = 0; i < stride; i++) {
            long idx = m * stride + i;
            x[idx] = 0.1f * idx + 0.01f * m;
            h[idx] = 0.2f * idx + 0.02f * m;
            dy[idx] = 0.01f * idx + 0.001f * m;
            A[idx] = -0.1f + 0.05f * idx + 0.01f * m;
            B[idx] = 0.3f + 0.1f * idx + 0.02f * m;
            C[idx] = 0.5f + 0.1f * idx + 0.03f * m;
            delta[idx] = 0.1f + 0.01f * idx + 0.005f * m;
        }
    }
    
    // Paramètres pour M>1
    ScanBackwardMParams params = {
        .x = x, .A = A, .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h, .dy = dy,
        .dx = dx, .dA = dA, .dB = dB, .dC = dC, .ddelta = ddelta,
        .L = L, .D = D, .M = M
    };
    
    printf("Appel scan1d_backward_m_generic avec M=%ld, L=%ld, D=%ld...\n", M, L, D);
    scan1d_backward_m_generic(&params);
    printf("Terminé\n");
    
    // Afficher quelques résultats pour vérifier
    printf("\nRésultats dC (première dimension M=0):\n");
    for (int i = 0; i < 3; i++) {
        printf("  dC[%d] = %.6f\n", i, dC[i]);
    }
    
    printf("\nRésultats dC (deuxième dimension M=1):\n");
    for (int i = 0; i < 3; i++) {
        printf("  dC[%ld] = %.6f\n", stride + i, dC[stride + i]);
    }
    
    // Test avec M=1 pour comparaison
    printf("\n--- Test avec M=1 pour comparaison ---\n");
    ScanBackwardMParams params_m1 = {
        .x = x, .A = A, .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h, .dy = dy,
        .dx = dx, .dA = dA, .dB = dB, .dC = dC, .ddelta = ddelta,
        .L = L, .D = D, .M = 1
    };
    
    scan1d_backward_m_generic(&params_m1);
    
    printf("Résultats dC avec M=1:\n");
    for (int i = 0; i < 3; i++) {
        printf("  dC[%d] = %.6f\n", i, dC[i]);
    }
    
    // Nettoyer
    free(x); free(A); free(B); free(C); free(delta); free(h); free(dy);
    free(dx); free(dA); free(dB); free(dC); free(ddelta);
    
    return 0;
}
