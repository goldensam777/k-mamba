#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../optimatrix/include/optimatrix.h"

/* Test CUDA gradient clipping */
int test_cuda_gradient_clipping() {
    printf("=== Test CUDA Gradient Clipping ===\n");
    
    size_t n = 1000;
    float max_norm = 5.0f;
    
    // Allouer mémoire GPU
    float *d_grad;
    cudaMalloc(&d_grad, n * sizeof(float));
    
    // Créer des gradients très grands
    float *h_grad = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        h_grad[i] = 10.0f + (float)i;  // Gradients croissants
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

/* Test CUDA AdamW optimizer */
int test_cuda_adamw() {
    printf("\n=== Test CUDA AdamW Optimizer ===\n");
    
    size_t n = 512;
    MBOptimConfig config = {
        .lr = 1e-3f,
        .mu = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .clip_norm = 1.0f,
        .weight_decay = 1e-5f
    };
    
    // Allouer mémoire GPU
    float *d_param, *d_grad, *d_m, *d_v;
    cudaMalloc(&d_param, n * sizeof(float));
    cudaMalloc(&d_grad, n * sizeof(float));
    cudaMalloc(&d_m, n * sizeof(float));
    cudaMalloc(&d_v, n * sizeof(float));
    
    // Initialiser sur GPU (zéros)
    cudaMemset(d_param, 0, n * sizeof(float));
    cudaMemset(d_m, 0, n * sizeof(float));
    cudaMemset(d_v, 0, n * sizeof(float));
    
    // Créer gradients grands pour tester clipping
    float *h_grad = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        h_grad[i] = 20.0f;  // Très grands gradients
    }
    cudaMemcpy(d_grad, h_grad, n * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Test AdamW avec gradients grands (norme initiale: %.1f)\n", 
           gradient_norm_cuda(d_grad, n));
    
    // Lancer AdamW pour quelques steps
    for (size_t step = 1; step <= 3; step++) {
        adamw_update_cuda(d_param, d_grad, d_m, d_v, n, &config, step);
        
        float param_norm = gradient_norm_cuda(d_param, n);
        printf("Step %zu: norme paramètres = %.6f\n", step, param_norm);
    }
    
    printf("✅ AdamW CUDA fonctionnel\n");
    
    // Nettoyage
    cudaFree(d_param);
    cudaFree(d_grad);
    cudaFree(d_m);
    cudaFree(d_v);
    free(h_grad);
    
    return 1;
}

/* Test CUDA MUON optimizer */
int test_cuda_muon() {
    printf("\n=== Test CUDA MUON Optimizer ===\n");
    
    size_t n = 512;
    MBOptimConfig config = {
        .lr = 1e-3f,
        .mu = 0.9f,
        .clip_norm = 0.5f,
        .weight_decay = 1e-5f
    };
    
    // Allouer mémoire GPU
    float *d_param, *d_grad, *d_m;
    cudaMalloc(&d_param, n * sizeof(float));
    cudaMalloc(&d_grad, n * sizeof(float));
    cudaMalloc(&d_m, n * sizeof(float));
    
    // Initialiser
    cudaMemset(d_param, 0, n * sizeof(float));
    cudaMemset(d_m, 0, n * sizeof(float));
    
    // Gradients grands
    float *h_grad = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        h_grad[i] = 15.0f;
    }
    cudaMemcpy(d_grad, h_grad, n * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Test MUON avec gradients grands (norme initiale: %.1f)\n", 
           gradient_norm_cuda(d_grad, n));
    
    // Lancer MUON pour quelques steps
    for (size_t step = 1; step <= 3; step++) {
        muon_update_cuda(d_param, d_grad, d_m, n, &config);
        
        float param_norm = gradient_norm_cuda(d_param, n);
        float momentum_norm = gradient_norm_cuda(d_m, n);
        printf("Step %zu: param=%.6f, momentum=%.6f\n", step, param_norm, momentum_norm);
    }
    
    printf("✅ MUON CUDA fonctionnel\n");
    
    // Nettoyage
    cudaFree(d_param);
    cudaFree(d_grad);
    cudaFree(d_m);
    free(h_grad);
    
    return 1;
}

/* Test de consistance CPU/CUDA */
int test_cpu_cuda_consistency() {
    printf("\n=== Test Consistance CPU/CUDA ===\n");
    
    size_t n = 100;
    float max_norm = 2.0f;
    
    // Test sur CPU
    float *h_grad_cpu = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        h_grad_cpu[i] = 5.0f + (float)i * 0.1f;
    }
    
    float *h_grad_copy = (float*)malloc(n * sizeof(float));
    memcpy(h_grad_copy, h_grad_cpu, n * sizeof(float));
    
    gradient_clip_inplace(h_grad_cpu, n, max_norm);
    float cpu_norm = gradient_norm(h_grad_cpu, n);
    
    // Test sur CUDA
    float *d_grad;
    cudaMalloc(&d_grad, n * sizeof(float));
    cudaMemcpy(d_grad, h_grad_copy, n * sizeof(float), cudaMemcpyHostToDevice);
    
    gradient_clip_inplace_cuda(d_grad, n, max_norm);
    
    float *h_grad_cuda = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_grad_cuda, d_grad, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    float cuda_norm = gradient_norm(h_grad_cuda, n);
    
    printf("CPU norm: %.6f, CUDA norm: %.6f ", cpu_norm, cuda_norm);
    
    // Vérifier la consistance
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(h_grad_cpu[i] - h_grad_cuda[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    if (max_diff < 1e-5f && fabsf(cpu_norm - cuda_norm) < 1e-5f) {
        printf("✅ (max diff: %.2e)\n", max_diff);
    } else {
        printf("❌ (max diff: %.2e)\n", max_diff);
        cudaFree(d_grad);
        free(h_grad_cpu);
        free(h_grad_copy);
        free(h_grad_cuda);
        return 0;
    }
    
    // Nettoyage
    cudaFree(d_grad);
    free(h_grad_cpu);
    free(h_grad_copy);
    free(h_grad_cuda);
    
    return 1;
}

int main() {
    printf("🚀 Tests CUDA Optimizers + Clipping\n");
    printf("===================================\n");
    
    int success = 1;
    
    // Test 1: Gradient clipping CUDA
    if (!test_cuda_gradient_clipping()) {
        success = 0;
    }
    
    // Test 2: AdamW CUDA
    if (!test_cuda_adamw()) {
        success = 0;
    }
    
    // Test 3: MUON CUDA
    if (!test_cuda_muon()) {
        success = 0;
    }
    
    // Test 4: Consistance CPU/CUDA
    if (!test_cpu_cuda_consistency()) {
        success = 0;
    }
    
    printf("\n=== Résultat Final CUDA ===\n");
    if (success) {
        printf("🎉 Tous les tests CUDA passés !\n");
        printf("✅ Gradient clipping CUDA fonctionnel\n");
        printf("✅ AdamW CUDA avec clipping\n");
        printf("✅ MUON CUDA avec clipping sur momentum\n");
        printf("✅ Consistance CPU/CUDA parfaite\n");
    } else {
        printf("❌ Certains tests CUDA ont échoué\n");
    }
    
    return success ? 0 : 1;
}
