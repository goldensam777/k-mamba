#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "optimatrix.h"

int main() {
    printf("Test ConvND 1D et 2D\n");
    
    // Test 1D : N=4, D=3, M=2
    printf("\n=== Test ConvND 1D ===\n");
    long dims_1d[] = {4};  // N=4
    long L = 4, D = 3, M = 2;
    long stride = L * D;  // 12
    
    // Allouer les tableaux
    float *input_1d = malloc(M * stride * sizeof(float));
    float *A = malloc(M * M * D * sizeof(float));
    float *B = malloc(M * M * D * sizeof(float));
    float *C = malloc(M * M * D * sizeof(float));
    float *delta = malloc(M * stride * sizeof(float));
    float *h0 = malloc(M * D * sizeof(float));
    float *output_1d = malloc(M * stride * sizeof(float));
    
    // Initialiser
    for (long m = 0; m < M; m++) {
        for (long i = 0; i < stride; i++) {
            long idx = m * stride + i;
            input_1d[idx] = 0.1f * idx + 0.01f * m;
            delta[idx] = 0.1f + 0.01f * idx + 0.005f * m;
        }
        for (long d = 0; d < D; d++) {
            h0[m * D + d] = 0.05f * m + 0.01f * d;
        }
    }
    
    // Initialiser A, B, C (identité pour l'instant)
    memset(A, 0, M * M * D * sizeof(float));
    memset(B, 0, M * M * D * sizeof(float));
    memset(C, 0, M * M * D * sizeof(float));
    for (long m = 0; m < M; m++) {
        for (long d = 0; d < D; d++) {
            A[m * M * D + m * D + d] = -0.1f + 0.05f * d;  // A[m,m,d]
            B[m * M * D + m * D + d] = 0.3f + 0.1f * d;   // B[m,m,d]
            C[m * M * D + m * D + d] = 0.5f + 0.1f * d;   // C[m,m,d]
        }
    }
    
    // Paramètres ConvND 1D
    ConvNDParams params_1d = {
        .input = input_1d,
        .A = A,
        .B = B,
        .C = C,
        .delta = delta,
        .h0 = h0,
        .output = output_1d,
        .dims = dims_1d,
        .ndims = 1,
        .D = D,
        .M = M
    };
    
    printf("Forward pass 1D...\n");
    convnd_forward(&params_1d);
    
    printf("Résultats output 1D (première dimension M=0):\n");
    for (int i = 0; i < 3; i++) {
        printf("  output[%d] = %.6f\n", i, output_1d[i]);
    }
    
    printf("Résultats output 1D (deuxième dimension M=1):\n");
    for (int i = 0; i < 3; i++) {
        printf("  output[%ld] = %.6f\n", stride + i, output_1d[stride + i]);
    }
    
    // Test 2D : N1=3, N2=4, D=2, M=2
    printf("\n=== Test ConvND 2D ===\n");
    long dims_2d[] = {3, 4};  // 3x4 = 12 positions spatiales
    long total_2d = 3 * 4;  // 12
    long stride_2d = total_2d * D;  // 24
    
    float *input_2d = malloc(M * stride_2d * sizeof(float));
    float *output_2d = malloc(M * stride_2d * sizeof(float));
    
    // Initialiser
    for (long m = 0; m < M; m++) {
        for (long i = 0; i < stride_2d; i++) {
            long idx = m * stride_2d + i;
            input_2d[idx] = 0.05f * idx + 0.02f * m;
        }
    }
    
    // Paramètres ConvND 2D
    ConvNDParams params_2d = {
        .input = input_2d,
        .A = A,
        .B = B,
        .C = C,
        .delta = delta,  // Réutiliser pour simplifier
        .h0 = h0,
        .output = output_2d,
        .dims = dims_2d,
        .ndims = 2,
        .D = D,
        .M = M
    };
    
    printf("Forward pass 2D...\n");
    convnd_forward(&params_2d);
    
    printf("Résultats output 2D (position [0,0], M=0):\n");
    for (int d = 0; d < 2; d++) {
        printf("  output[0][0][%d] = %.6f\n", d, output_2d[d]);
    }
    
    printf("Résultats output 2D (position [1,1], M=1):\n");
    long pos_11 = 1 * 4 + 1;  // [1,1] en row-major
    for (int d = 0; d < 2; d++) {
        printf("  output[1][1][%d] = %.6f\n", d, output_2d[stride_2d + pos_11 * 2 + d]);
    }
    
    // Test backward 1D
    printf("\n=== Test Backward 1D ===\n");
    printf("Backward pass 1D...\n");
    convnd_backward(&params_1d);
    printf("Backward terminé\n");
    
    // Nettoyer
    free(input_1d); free(A); free(B); free(C); free(delta); free(h0); free(output_1d);
    free(input_2d); free(output_2d);
    
    return 0;
}
