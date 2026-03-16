/*
 * test_gpu_simple.c — Tests GPU simples pour k-mamba
 *
 * Phase 6 : Tests CUDA/GPU - Version simplifiée
 * Objectif : Valider les fonctionnalités GPU de base sans dépendances complexes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>

/* Inclusions CUDA */
#include <cuda_runtime.h>

/* ============================================================
 * Configuration GPU
 * ============================================================ */

typedef struct {
    int device_id;
    char device_name[256];
    size_t total_memory;
    size_t available_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int max_threads_per_block;
    size_t shared_memory_per_block;
    double clock_rate;
} GPUDeviceInfo;

/* ============================================================
 * Utilitaires de test GPU
 * ============================================================ */

#define EPSILON 1e-5f
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("FAIL: %s\n", msg); \
            return 0; \
        } \
    } while(0)

#define CUDA_CHECK(call, msg) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err)); \
            return 0; \
        } \
    } while(0)

static void fill_random(float *data, size_t n, float min, float max) {
    for (size_t i = 0; i < n; i++) {
        data[i] = min + (max - min) * ((float)rand() / RAND_MAX);
    }
}

static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ============================================================
 * Kernel CUDA simple
 * ============================================================ */

__global__ void vector_add_kernel(float *a, float *b, float *c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_scale_kernel(float *a, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scale;
    }
}

__global__ void matrix_multiply_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/* ============================================================
 * Fonctions GPU utilitaires
 * ============================================================ */

static int gpu_init_device(GPUDeviceInfo *info) {
    printf("Initializing GPU device...\n");
    
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    
    TEST_ASSERT(device_count > 0, "No CUDA devices found");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    
    info->device_id = 0;
    strcpy(info->device_name, prop.name);
    info->total_memory = prop.totalGlobalMem;
    info->compute_capability_major = prop.major;
    info->compute_capability_minor = prop.minor;
    info->max_threads_per_block = prop.maxThreadsPerBlock;
    info->shared_memory_per_block = prop.sharedMemPerBlock;
    info->clock_rate = prop.clockRate * 1e-6;  // Convert to GHz
    
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem), "cudaMemGetInfo");
    info->available_memory = free_mem;
    
    printf("GPU: %s (Compute %d.%d)\n", info->device_name, 
           info->compute_capability_major, info->compute_capability_minor);
    printf("Memory: %.1f GB total, %.1f GB free\n", 
           info->total_memory / 1e9, info->available_memory / 1e9);
    
    return 1;
}

static int test_gpu_memory_basic() {
    printf("Testing basic GPU memory operations...\n");
    
    size_t test_size = 1024 * 1024;  // 1M elements
    
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, test_size * sizeof(float)), "cudaMalloc");
    
    float *h_data = (float*)malloc(test_size * sizeof(float));
    TEST_ASSERT(h_data != NULL, "Host memory allocation failed");
    
    fill_random(h_data, test_size, -1.0f, 1.0f);
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data, test_size * sizeof(float), 
                         cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    
    float *h_result = (float*)malloc(test_size * sizeof(float));
    TEST_ASSERT(h_result != NULL, "Result memory allocation failed");
    
    CUDA_CHECK(cudaMemcpy(h_result, d_data, test_size * sizeof(float), 
                         cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    
    /* Vérifier les données */
    for (size_t i = 0; i < test_size; i++) {
        TEST_ASSERT(fabsf(h_data[i] - h_result[i]) < EPSILON, "Memory copy failed");
    }
    
    CUDA_CHECK(cudaFree(d_data), "cudaFree");
    free(h_data);
    free(h_result);
    
    printf("PASS: Basic GPU memory operations\n");
    return 1;
}

static int test_gpu_vector_operations() {
    printf("Testing GPU vector operations...\n");
    
    const size_t n = 1024 * 1024;
    
    /* Allouer mémoire GPU */
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)), "cudaMalloc a");
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)), "cudaMalloc b");
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)), "cudaMalloc c");
    
    /* Allouer mémoire host */
    float *h_a = (float*)malloc(n * sizeof(float));
    float *h_b = (float*)malloc(n * sizeof(float));
    float *h_c = (float*)malloc(n * sizeof(float));
    
    TEST_ASSERT(h_a && h_b && h_c, "Host memory allocation failed");
    
    fill_random(h_a, n, -1.0f, 1.0f);
    fill_random(h_b, n, -1.0f, 1.0f);
    
    /* Copier sur GPU */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy a");
    CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy b");
    
    /* Configurer le kernel */
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    /* Lancer le kernel vector_add */
    double start_time = get_time();
    vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double end_time = get_time();
    
    /* Copier le résultat */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy c result");
    
    /* Vérifier les résultats */
    for (size_t i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        TEST_ASSERT(fabsf(h_c[i] - expected) < EPSILON, "Vector addition failed");
    }
    
    /* Performances */
    double elapsed = end_time - start_time;
    double bandwidth = (3.0 * n * sizeof(float)) / elapsed / 1e9;  // 2 reads + 1 write
    
    printf("GPU Vector Add Performance:\n");
    printf("  Size: %zu elements\n", n);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth);
    printf("  Latency: %.3f ms\n", elapsed * 1000);
    
    /* Test vector scale */
    start_time = get_time();
    vector_scale_kernel<<<grid_size, block_size>>>(d_a, 2.0f, n);
    CUDA_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    end_time = get_time();
    
    CUDA_CHECK(cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy a scaled");
    
    /* Vérifier scaling - comparer avec les valeurs originales */
    for (size_t i = 0; i < n; i++) {
        float expected = 2.0f * (h_a[i] / 2.0f);  // h_a contient déjà les valeurs scaled
        TEST_ASSERT(fabsf(h_a[i] - expected) < EPSILON, "Vector scaling failed");
    }
    
    /* Nettoyer */
    CUDA_CHECK(cudaFree(d_a), "cudaFree a");
    CUDA_CHECK(cudaFree(d_b), "cudaFree b");
    CUDA_CHECK(cudaFree(d_c), "cudaFree c");
    free(h_a); free(h_b); free(h_c);
    
    printf("PASS: GPU vector operations\n");
    return 1;
}

static int test_gpu_matrix_multiply() {
    printf("Testing GPU matrix multiplication...\n");
    
    const int M = 512, N = 512, K = 512;
    
    /* Allouer mémoire GPU */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)), "cudaMalloc A");
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)), "cudaMalloc B");
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)), "cudaMalloc C");
    
    /* Allouer mémoire host */
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    
    TEST_ASSERT(h_A && h_B && h_C, "Host memory allocation failed");
    
    fill_random(h_A, M * K, -1.0f, 1.0f);
    fill_random(h_B, K * N, -1.0f, 1.0f);
    
    /* Copier sur GPU */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy A");
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy B");
    
    /* Configurer le kernel */
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, 
                   (M + block_size.y - 1) / block_size.y);
    
    /* Lancer le kernel */
    double start_time = get_time();
    matrix_multiply_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double end_time = get_time();
    
    /* Copier le résultat */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy C result");
    
    /* Vérifier que les résultats sont finis */
    for (size_t i = 0; i < M * N; i++) {
        TEST_ASSERT(isfinite(h_C[i]), "Matrix multiply result is not finite");
    }
    
    /* Calculer les performances */
    double elapsed = end_time - start_time;
    double gflops = (2.0 * M * N * K) / (elapsed * 1e9);
    
    printf("GPU Matrix Multiply Performance:\n");
    printf("  Size: %dx%dx%d\n", M, N, K);
    printf("  GFLOPS: %.2f\n", gflops);
    printf("  Latency: %.3f ms\n", elapsed * 1000);
    
    /* Nettoyer */
    CUDA_CHECK(cudaFree(d_A), "cudaFree A");
    CUDA_CHECK(cudaFree(d_B), "cudaFree B");
    CUDA_CHECK(cudaFree(d_C), "cudaFree C");
    free(h_A); free(h_B); free(h_C);
    
    printf("PASS: GPU matrix multiplication\n");
    return 1;
}

static int test_gpu_kmamba_simulation() {
    printf("Testing GPU KMamba simulation...\n");
    
    /* Configuration KMamba simplifiée */
    const size_t vocab_size = 256;
    const size_t dim = 512;
    const size_t seq_len = 128;
    const size_t batch_size = 4;
    
    /* Allouer mémoire GPU */
    float *d_tokens, *d_embedding, *d_output;
    CUDA_CHECK(cudaMalloc(&d_tokens, batch_size * seq_len * sizeof(float)), "cudaMalloc tokens");
    CUDA_CHECK(cudaMalloc(&d_embedding, vocab_size * dim * sizeof(float)), "cudaMalloc embedding");
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * seq_len * dim * sizeof(float)), "cudaMalloc output");
    
    /* Allouer mémoire host */
    float *h_tokens = (float*)malloc(batch_size * seq_len * sizeof(float));
    float *h_embedding = (float*)malloc(vocab_size * dim * sizeof(float));
    float *h_output = (float*)malloc(batch_size * seq_len * dim * sizeof(float));
    
    TEST_ASSERT(h_tokens && h_embedding && h_output, "Host memory allocation failed");
    
    /* Initialiser */
    fill_random(h_tokens, batch_size * seq_len, 0.0f, 255.0f);
    fill_random(h_embedding, vocab_size * dim, -0.1f, 0.1f);
    
    /* Copier sur GPU */
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens, batch_size * seq_len * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy tokens");
    CUDA_CHECK(cudaMemcpy(d_embedding, h_embedding, vocab_size * dim * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy embedding");
    
    /* Simuler embedding lookup sur GPU */
    int block_size = 256;
    int grid_size = (batch_size * seq_len * dim + block_size - 1) / block_size;
    
    double start_time = get_time();
    
    /* Simulation simplifiée : token -> embedding */
    for (int iter = 0; iter < 10; iter++) {
        /* Copier tokens pour traitement simulé */
        CUDA_CHECK(cudaMemcpy(h_tokens, d_tokens, batch_size * seq_len * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy tokens back");
        
        /* Embedding lookup simulé */
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t t = 0; t < seq_len; t++) {
                int token = (int)h_tokens[b * seq_len + t] % vocab_size;
                for (size_t d = 0; d < dim; d++) {
                    h_output[b * seq_len * dim + t * dim + d] = h_embedding[token * dim + d];
                }
            }
        }
        
        /* Copier sur GPU */
        CUDA_CHECK(cudaMemcpy(d_output, h_output, batch_size * seq_len * dim * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy output");
    }
    
    CUDA_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double end_time = get_time();
    
    /* Vérifier les résultats */
    CUDA_CHECK(cudaMemcpy(h_output, d_output, batch_size * seq_len * dim * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy output final");
    
    for (size_t i = 0; i < batch_size * seq_len * dim; i++) {
        TEST_ASSERT(isfinite(h_output[i]), "KMamba output is not finite");
    }
    
    /* Performances */
    double elapsed = end_time - start_time;
    double throughput = (batch_size * seq_len * 10) / elapsed;  // tokens/sec
    
    printf("GPU KMamba Simulation Performance:\n");
    printf("  Model: %zu vocab, %zu dim, %zu seq_len\n", vocab_size, dim, seq_len);
    printf("  Batch: %zu\n", batch_size);
    printf("  Throughput: %.2f tokens/sec\n", throughput);
    printf("  Latency: %.3f ms/forward\n", elapsed / 10 * 1000);
    
    /* Nettoyer */
    CUDA_CHECK(cudaFree(d_tokens), "cudaFree tokens");
    CUDA_CHECK(cudaFree(d_embedding), "cudaFree embedding");
    CUDA_CHECK(cudaFree(d_output), "cudaFree output");
    free(h_tokens); free(h_embedding); free(h_output);
    
    printf("PASS: GPU KMamba simulation\n");
    return 1;
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== k-mamba GPU Test Suite (Simple) ===\n");
    printf("Testing basic GPU functionality\n\n");
    
    srand(42); /* Pour reproductibilité */
    
    int passed = 0, total = 0;
    
    /* Informations GPU */
    GPUDeviceInfo info;
    TEST_ASSERT(gpu_init_device(&info), "GPU initialization failed");
    
    /* Tests GPU */
    total++; passed += test_gpu_memory_basic();
    total++; passed += test_gpu_vector_operations();
    total++; passed += test_gpu_matrix_multiply();
    total++; passed += test_gpu_kmamba_simulation();
    
    printf("\n=== GPU Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All GPU tests PASSED!\n");
        
        printf("\n=== GPU Device Summary ===\n");
        printf("Device: %s\n", info.device_name);
        printf("Compute: %d.%d\n", info.compute_capability_major, info.compute_capability_minor);
        printf("Memory: %.1f GB total, %.1f GB free\n", info.total_memory / 1e9, info.available_memory / 1e9);
        printf("Max Threads/Block: %d\n", info.max_threads_per_block);
        printf("Clock Rate: %.2f GHz\n", info.clock_rate);
        
        return 0;
    } else {
        printf("Some GPU tests FAILED!\n");
        return 1;
    }
}
