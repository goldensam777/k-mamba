#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "optimatrix.h"

// Test simple pour comparer les versions C et ASM du noyau M=1 shared_bc
int main() {
    const long L = 128, D = 64;
    
    // Allouer les tableaux
    float *x = malloc(L * D * sizeof(float));
    float *A = malloc(D * sizeof(float));
    float *B = malloc(D * sizeof(float));
    float *C = malloc(D * sizeof(float));
    float *delta = malloc(L * sizeof(float));
    float *h = malloc(L * D * sizeof(float));
    float *dy = malloc(L * D * sizeof(float));
    
    float *dx_c = malloc(L * D * sizeof(float));
    float *dA_c = malloc(D * sizeof(float));
    float *dB_c = malloc(D * sizeof(float));
    float *dC_c = malloc(D * sizeof(float));
    float *ddelta_c = malloc(L * sizeof(float));
    
    float *dx_asm = malloc(L * D * sizeof(float));
    float *dA_asm = malloc(D * sizeof(float));
    float *dB_asm = malloc(D * sizeof(float));
    float *dC_asm = malloc(D * sizeof(float));
    float *ddelta_asm = malloc(L * sizeof(float));
    
    // Initialiser avec des valeurs aléatoires
    srand(42);
    for (long i = 0; i < L * D; i++) {
        x[i] = (float)rand() / RAND_MAX;
        h[i] = (float)rand() / RAND_MAX;
        dy[i] = (float)rand() / RAND_MAX;
    }
    
    for (long i = 0; i < L; i++) {
        delta[i] = 0.1f + (float)rand() / RAND_MAX;
    }
    
    for (long i = 0; i < D; i++) {
        A[i] = -0.1f + (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
        C[i] = (float)rand() / RAND_MAX;
    }
    
    // Préparer les structures
    ScanBackwardSharedParams params_c = {
        .x = x, .A = A, .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h, .dy = dy,
        .dx = dx_c, .dA = dA_c, .dB = dB_c, .dC = dC_c, .ddelta = ddelta_c,
        .L = L, .D = D
    };
    
    ScanBackwardSharedParams params_asm = {
        .x = x, .A = A, .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h, .dy = dy,
        .dx = dx_asm, .dA = dA_asm, .dB = dB_asm, .dC = dC_asm, .ddelta = ddelta_asm,
        .L = L, .D = D
    };
    
    // Benchmark version C
    clock_t start = clock();
    for (int iter = 0; iter < 100; iter++) {
        memset(dx_c, 0, L * D * sizeof(float));
        memset(dA_c, 0, D * sizeof(float));
        memset(dB_c, 0, D * sizeof(float));
        memset(dC_c, 0, D * sizeof(float));
        memset(ddelta_c, 0, L * sizeof(float));
        
        scan1d_backward_m1_shared_bc(&params_c);
    }
    clock_t end = clock();
    double time_c = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Benchmark version ASM (quand elle sera prête)
    printf("Version C: %.3f seconds\n", time_c);
    
    // TODO: Appeler la version ASM et comparer
    // scan1d_backward_m1_shared_bc_asm(&params_asm);
    
    // Vérifier quelques résultats
    printf("\nPremiers résultats C:\n");
    for (int i = 0; i < 5; i++) {
        printf("dx[%d] = %.6f\n", i, dx_c[i]);
    }
    
    // Nettoyer
    free(x); free(A); free(B); free(C); free(delta); free(h); free(dy);
    free(dx_c); free(dA_c); free(dB_c); free(dC_c); free(ddelta_c);
    free(dx_asm); free(dA_asm); free(dB_asm); free(dC_asm); free(ddelta_asm);
    
    return 0;
}
