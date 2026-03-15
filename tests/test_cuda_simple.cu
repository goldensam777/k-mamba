#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../optimatrix/include/optimatrix.h"

/* Test simple du gradient clipping CUDA */
int test_cuda_clipping_simple() {
    printf("=== Test CUDA Gradient Clipping Simple ===\n");
    
    size_t n = 1000;
    float max_norm = 5.0f;
    
    // Allouer mémoire GPU
    float *d_grad;
    cudaError_t err = cudaMalloc(&d_grad, n * sizeof(float));
    if (err != cudaSuccess) {
        printf("❌ Erreur allocation GPU: %s\n", cudaGetErrorString(err));
        return 0;
    }
    
    // Créer des gradients très grands
    float *h_grad = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        h_grad[i] = 10.0f + (float)i;
    }
    
    // Copier sur GPU
    cudaMemcpy(d_grad, h_grad, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Calculer norme avant clipping
    float norm_before = gradient_norm_cuda(d_grad, n);
    printf("Norme avant clipping: %.3f (max: %.1f) ", norm_before, max_norm);
    
    // Appliquer clipping
    gradient_clip_inplace_cuda(d_grad, n, max_norm);
    
    // Calculer norme après clipping
    float norm_after = gradient_norm_cuda(d_grad, n);
    printf("→ après: %.3f ", norm_after);
    
    if (norm_after <= max_norm + 1e-6f) {
        printf("✅\n");
    } else {
        printf("❌\n");
        cudaFree(d_grad);
        free(h_grad);
        return 0;
    }
    
    // Nettoyage
    cudaFree(d_grad);
    free(h_grad);
    
    return 1;
}

int main() {
    printf("🚀 Test CUDA Simple\n");
    printf("===================\n");
    
    int success = test_cuda_clipping_simple();
    
    printf("\n=== Résultat ===\n");
    if (success) {
        printf("🎉 Test CUDA clipping réussi !\n");
        printf("✅ GPU allocation fonctionnelle\n");
        printf("✅ Gradient clipping CUDA fonctionnel\n");
    } else {
        printf("❌ Test CUDA échoué\n");
    }
    
    return success ? 0 : 1;
}
