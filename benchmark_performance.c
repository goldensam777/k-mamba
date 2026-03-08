#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "optimatrix.h"

// Benchmark simple pour comparer les performances
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main() {
    printf("Benchmark Performance Mamba ND\n");
    
    // Paramètres de test
    long L = 1024, D = 512, M = 4;
    long stride = L * D;
    
    printf("Configuration: L=%ld, D=%ld, M=%ld\n", L, D, M);
    printf("Taille totale: %.2f MB\n", (double)(M * stride * sizeof(float)) / (1024*1024));
    
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
    
    // Initialiser avec des valeurs aléatoires
    srand(42);
    for (long i = 0; i < M * stride; i++) {
        x[i] = (float)rand() / RAND_MAX;
        A[i] = -0.1f + (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
        C[i] = (float)rand() / RAND_MAX;
        delta[i] = 0.1f + (float)rand() / RAND_MAX;
        h[i] = (float)rand() / RAND_MAX;
        dy[i] = (float)rand() / RAND_MAX;
    }
    
    // Paramètres pour M>1
    ScanBackwardMParams params = {
        .x = x, .A = A, .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h, .dy = dy,
        .dx = dx, .dA = dA, .dB = dB, .dC = dC, .ddelta = ddelta,
        .L = L, .D = D, .M = M
    };
    
    // Benchmark M>1
    printf("\nBenchmark scan1d_backward_m_generic:\n");
    
    double start = get_time();
    for (int iter = 0; iter < 10; iter++) {
        scan1d_backward_m_generic(&params);
    }
    double end = get_time();
    
    double avg_time = (end - start) / 10.0;
    double throughput = (double)(M * stride) / (avg_time * 1e9);  // GFLOPS approximation
    
    printf("  Temps moyen: %.3f ms\n", avg_time * 1000);
    printf("  Débit: %.2f GB/s\n", (double)(M * stride * sizeof(float)) / (avg_time * 1e9));
    printf("  Premiers résultats dC: %.6f, %.6f, %.6f\n", dC[0], dC[1], dC[2]);
    
    // Benchmark ConvND 1D
    printf("\nBenchmark ConvND 1D:\n");
    long dims_1d[] = {L};
    
    float *input = malloc(M * stride * sizeof(float));
    float *output = malloc(M * stride * sizeof(float));
    memcpy(input, x, M * stride * sizeof(float));
    
    ConvNDParams conv_params = {
        .input = input,
        .A = A,
        .B = B,
        .C = C,
        .delta = delta,
        .h0 = NULL,
        .output = output,
        .dims = dims_1d,
        .ndims = 1,
        .D = D,
        .M = M
    };
    
    start = get_time();
    for (int iter = 0; iter < 10; iter++) {
        convnd_forward(&conv_params);
    }
    end = get_time();
    
    avg_time = (end - start) / 10.0;
    
    printf("  Temps moyen: %.3f ms\n", avg_time * 1000);
    printf("  Débit: %.2f GB/s\n", (double)(M * stride * sizeof(float)) / (avg_time * 1e9));
    printf("  Premiers résultats output: %.6f, %.6f, %.6f\n", output[0], output[1], output[2]);
    
    // Test de scalabilité
    printf("\nTest de scalabilité (M=1 à M=8):\n");
    for (int test_M = 1; test_M <= 8; test_M *= 2) {
        // Allouer la mémoire spécifique pour ce test
        long test_stride = L * D;
        float *test_x = malloc(test_M * test_stride * sizeof(float));
        float *test_A = malloc(test_M * test_stride * sizeof(float));
        float *test_B = malloc(test_M * test_stride * sizeof(float));
        float *test_C = malloc(test_M * test_stride * sizeof(float));
        float *test_delta = malloc(test_M * test_stride * sizeof(float));
        float *test_h = malloc(test_M * test_stride * sizeof(float));
        float *test_dy = malloc(test_M * test_stride * sizeof(float));
        
        float *test_dx = malloc(test_M * test_stride * sizeof(float));
        float *test_dA = malloc(test_M * test_stride * sizeof(float));
        float *test_dB = malloc(test_M * test_stride * sizeof(float));
        float *test_dC = malloc(test_M * test_stride * sizeof(float));
        float *test_ddelta = malloc(test_M * test_stride * sizeof(float));
        
        // Initialiser (copier les premières valeurs des tableaux originaux)
        long copy_size = (test_M <= M) ? test_M * test_stride : M * test_stride;
        if (test_M <= M) {
            memcpy(test_x, x, test_M * test_stride * sizeof(float));
            memcpy(test_A, A, test_M * test_stride * sizeof(float));
            memcpy(test_B, B, test_M * test_stride * sizeof(float));
            memcpy(test_C, C, test_M * test_stride * sizeof(float));
            memcpy(test_delta, delta, test_M * test_stride * sizeof(float));
            memcpy(test_h, h, test_M * test_stride * sizeof(float));
            memcpy(test_dy, dy, test_M * test_stride * sizeof(float));
        } else {
            // Pour M > 4, initialiser avec zéro
            memset(test_x, 0, test_M * test_stride * sizeof(float));
            memset(test_A, 0, test_M * test_stride * sizeof(float));
            memset(test_B, 0, test_M * test_stride * sizeof(float));
            memset(test_C, 0, test_M * test_stride * sizeof(float));
            memset(test_delta, 0, test_M * test_stride * sizeof(float));
            memset(test_h, 0, test_M * test_stride * sizeof(float));
            memset(test_dy, 0, test_M * test_stride * sizeof(float));
        }
        
        ScanBackwardMParams test_params = {
            .x = test_x, .A = test_A, .B = test_B, .C = test_C, .delta = test_delta,
            .h0 = NULL, .h = test_h, .dy = test_dy,
            .dx = test_dx, .dA = test_dA, .dB = test_dB, .dC = test_dC, .ddelta = test_ddelta,
            .L = L, .D = D, .M = test_M
        };
        
        start = get_time();
        for (int iter = 0; iter < 5; iter++) {
            scan1d_backward_m_generic(&test_params);
        }
        end = get_time();
        
        avg_time = (end - start) / 5.0;
        printf("  M=%2d: %.3f ms (%.2f GB/s)\n", test_M, avg_time * 1000, 
               (double)(test_M * test_stride * sizeof(float)) / (avg_time * 1e9));
        
        // Libérer la mémoire
        free(test_x); free(test_A); free(test_B); free(test_C); free(test_delta);
        free(test_h); free(test_dy);
        free(test_dx); free(test_dA); free(test_dB); free(test_dC); free(test_ddelta);
    }
    
    // Nettoyer
    free(x); free(A); free(B); free(C); free(delta); free(h); free(dy);
    free(dx); free(dA); free(dB); free(dC); free(ddelta);
    free(input); free(output);
    
    return 0;
}
