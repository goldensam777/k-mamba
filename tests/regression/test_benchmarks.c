/*
 * test_benchmarks.c — Tests de régression et benchmarks performance
 *
 * Phase 4.2 : Benchmarks - Performance benchmarks
 * Objectif : Validation continue et performance de production
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "kmamba.h"
#include "optimatrix.h"

/* ============================================================
 * Utilitaires de benchmark
 * ============================================================ */

#define BENCHMARK_ITERATIONS 100
#define WARMUP_ITERATIONS 10

static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void fill_random(float *data, size_t n, float min, float max) {
    for (size_t i = 0; i < n; i++) {
        float scale = (float)rand() / (float)RAND_MAX;
        data[i] = min + scale * (max - min);
    }
}

/* ============================================================
 * Benchmarks Kernels Optimatrix
 * ============================================================ */

static void benchmark_gemm_kernels() {
    printf("=== GEMM Kernel Benchmarks ===\n");
    
    const size_t sizes[][3] = {
        {64, 64, 64},      // Small
        {256, 256, 256},   // Medium
        {512, 512, 512},   // Large
        {1024, 1024, 1024} // XL
    };
    
    const char *size_names[] = {"Small", "Medium", "Large", "XL"};
    
    for (int s = 0; s < 4; s++) {
        size_t m = sizes[s][0], k = sizes[s][1], n = sizes[s][2];
        
        float *A = (float*)malloc(m * k * sizeof(float));
        float *B = (float*)malloc(k * n * sizeof(float));
        float *C = (float*)malloc(m * n * sizeof(float));
        
        fill_random(A, m * k, -1.0f, 1.0f);
        fill_random(B, k * n, -1.0f, 1.0f);
        
        /* Warmup */
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            gemm_avx2(A, B, C, m, k, n);
        }
        
        /* Benchmark */
        double start = get_time();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            gemm_avx2(A, B, C, m, k, n);
        }
        double end = get_time();
        
        double elapsed = end - start;
        double ops = 2.0 * m * k * n * BENCHMARK_ITERATIONS;
        double gflops = ops / (elapsed * 1e9);
        double throughput = (m * n) / elapsed * BENCHMARK_ITERATIONS / 1e9;  // G elems/sec
        
        printf("GEMM %s (%zux%zux%zu): %.3f sec, %.2f GFLOPS, %.2f G elems/sec\n",
               size_names[s], m, k, n, elapsed, gflops, throughput);
        
        free(A); free(B); free(C);
    }
}

static void benchmark_gemv_kernels() {
    printf("\n=== GEMV Kernel Benchmarks ===\n");
    
    const size_t sizes[][2] = {
        {1024, 1024},    // Small
        {4096, 4096},    // Medium
        {8192, 8192},    // Large
        {16384, 16384}   // XL
    };
    
    const char *size_names[] = {"Small", "Medium", "Large", "XL"};
    
    for (int s = 0; s < 4; s++) {
        size_t m = sizes[s][0], n = sizes[s][1];
        
        float *A = (float*)malloc(m * n * sizeof(float));
        float *x = (float*)malloc(n * sizeof(float));
        float *y = (float*)malloc(m * sizeof(float));
        
        fill_random(A, m * n, -1.0f, 1.0f);
        fill_random(x, n, -1.0f, 1.0f);
        
        /* Warmup */
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            gemv_avx2(A, x, y, m, n);
        }
        
        /* Benchmark */
        double start = get_time();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            gemv_avx2(A, x, y, m, n);
        }
        double end = get_time();
        
        double elapsed = end - start;
        double ops = 2.0 * m * n * BENCHMARK_ITERATIONS;
        double gflops = ops / (elapsed * 1e9);
        double throughput = (m * n) / elapsed * BENCHMARK_ITERATIONS / 1e9;  // G elems/sec
        
        printf("GEMV %s (%zux%zu): %.3f sec, %.2f GFLOPS, %.2f G elems/sec\n",
               size_names[s], m, n, elapsed, gflops, throughput);
        
        free(A); free(x); free(y);
    }
}

static void benchmark_activation_kernels() {
    printf("\n=== Activation Kernel Benchmarks ===\n");
    
    const size_t sizes[] = {1024, 4096, 16384, 65536, 262144};
    const char *size_names[] = {"1K", "4K", "16K", "65K", "262K"};
    
    for (int s = 0; s < 5; s++) {
        size_t n = sizes[s];
        float *input = (float*)malloc(n * sizeof(float));
        float *output = (float*)malloc(n * sizeof(float));
        
        fill_random(input, n, -5.0f, 5.0f);
        
        /* Benchmark SiLU */
        double start = get_time();
        for (int i = 0; i < BENCHMARK_ITERATIONS * 10; i++) {
            silu_avx2(input, output, n);
        }
        double end = get_time();
        
        double elapsed = end - start;
        double throughput = (n * BENCHMARK_ITERATIONS * 10) / elapsed / 1e9;  // G elems/sec
        
        printf("SiLU %s: %.3f sec, %.2f G elems/sec\n", size_names[s], elapsed, throughput);
        
        free(input); free(output);
    }
}

static void benchmark_scan1d_kernels() {
    printf("\n=== Scan1D Kernel Benchmarks ===\n");
    
    const size_t configs[][3] = {
        {128, 64, 16},   // Small
        {256, 128, 32},  // Medium
        {512, 256, 64},  // Large
        {1024, 512, 128} // XL
    };
    
    const char *config_names[] = {"Small", "Medium", "Large", "XL"};
    
    for (int c = 0; c < 4; c++) {
        size_t L = configs[c][0], D = configs[c][1], M = configs[c][2];
        
        float *x = (float*)malloc(L * D * M * sizeof(float));
        float *dt = (float*)malloc(L * D * sizeof(float));
        float *A = (float*)malloc(L * D * M * sizeof(float));
        float *B = (float*)malloc(L * D * M * sizeof(float));
        float *C = (float*)malloc(L * D * M * sizeof(float));
        float *h = (float*)malloc(D * M * sizeof(float));
        
        fill_random(x, L * D * M, -1.0f, 1.0f);
        fill_random(dt, L * D, 0.001f, 0.1f);
        fill_random(A, L * D * M, -1.0f, 1.0f);
        fill_random(B, L * D * M, -1.0f, 1.0f);
        fill_random(C, L * D * M, -1.0f, 1.0f);
        fill_random(h, D * M, -1.0f, 1.0f);
        
        /* Warmup */
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            scan1d_forward(x, dt, A, B, C, h, L, D, M);
        }
        
        /* Benchmark */
        double start = get_time();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            scan1d_forward(x, dt, A, B, C, h, L, D, M);
        }
        double end = get_time();
        
        double elapsed = end - start;
        double ops = L * D * M * 6.0 * BENCHMARK_ITERATIONS;  // ~6 ops per element
        double gflops = ops / (elapsed * 1e9);
        double throughput = (L * D * M) / elapsed * BENCHMARK_ITERATIONS / 1e9;  // G elems/sec
        
        printf("Scan1D %s (L=%zu, D=%zu, M=%zu): %.3f sec, %.2f GFLOPS, %.2f G elems/sec\n",
               config_names[c], L, D, M, elapsed, gflops, throughput);
        
        free(x); free(dt); free(A); free(B); free(C); free(h);
    }
}

/* ============================================================
 * Benchmarks MambaBlock
 * ============================================================ */

static void benchmark_mamba_block() {
    printf("\n=== MambaBlock Benchmarks ===\n");
    
    const size_t configs[][3] = {
        {64, 128, 32},    // Small
        {128, 256, 64},   // Medium
        {256, 512, 128},  // Large
        {384, 1024, 128}  // Config actuelle
    };
    
    const char *config_names[] = {"Small", "Medium", "Large", "Current"};
    
    for (int c = 0; c < 4; c++) {
        size_t dim = configs[c][0], state_size = configs[c][1], seq_len = configs[c][2];
        
        MBConfig config = {
            .dim = dim,
            .state_size = state_size,
            .seq_len = seq_len,
            .dt_scale = 1.0f,
            .dt_min = 0.001f,
            .dt_max = 0.1f,
            .dt_rank = 1.0f,
            .dt_init = 1.0f,
            .use_convnd = 0,
            .convnd_K = 0,
            .convnd_ndims = 0
        };
        
        MambaBlock *block = mamba_block_create(&config);
        mamba_block_init(block);
        
        float *input = (float*)malloc(seq_len * dim * sizeof(float));
        float *output = (float*)malloc(seq_len * dim * sizeof(float));
        
        fill_random(input, seq_len * dim, -1.0f, 1.0f);
        
        /* Warmup */
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            mamba_block_forward(block, input, output);
        }
        
        /* Benchmark Forward */
        double start = get_time();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            mamba_block_forward(block, input, output);
        }
        double end = get_time();
        
        double elapsed = end - start;
        double throughput = (seq_len * dim) / elapsed * BENCHMARK_ITERATIONS / 1e9;  // G elems/sec
        double tokens_per_sec = seq_len / elapsed * BENCHMARK_ITERATIONS;
        
        printf("MambaBlock %s (dim=%zu, state=%zu, seq=%zu): %.3f sec, %.2f G elems/sec, %.0f tokens/sec\n",
               config_names[c], dim, state_size, seq_len, elapsed, throughput, tokens_per_sec);
        
        free(input); free(output);
        mamba_block_free(block);
    }
}

/* ============================================================
 * Benchmarks KMamba Complet
 * ============================================================ */

static void benchmark_kmamba_complete() {
    printf("\n=== KMamba Complete Benchmarks ===\n");
    
    const size_t configs[][4] = {
        {64, 128, 32, 1},    // Small, 1 layer
        {128, 256, 64, 1},   // Medium, 1 layer
        {256, 512, 128, 2},  // Large, 2 layers
        {384, 1024, 128, 2}  // Config actuelle
    };
    
    const char *config_names[] = {"Small", "Medium", "Large", "Current"};
    
    for (int c = 0; c < 4; c++) {
        size_t dim = configs[c][0], state_size = configs[c][1], seq_len = configs[c][2], n_layers = configs[c][3];
        
        KMambaConfig config = {
            .vocab_size = 256,
            .dim = dim,
            .state_size = state_size,
            .seq_len = seq_len,
            .n_layers = n_layers,
            .dt_scale = 1.0f,
            .dt_min = 0.001f,
            .dt_max = 0.1f
        };
        
        KMamba *model = kmamba_create(&config);
        kmamba_init(model, 1234);
        
        uint8_t *tokens = (uint8_t*)malloc(seq_len * sizeof(uint8_t));
        float *logits = (float*)malloc(seq_len * 256 * sizeof(float));
        
        for (size_t i = 0; i < seq_len; i++) {
            tokens[i] = (uint8_t)(rand() % 256);
        }
        
        /* Warmup */
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            kmamba_forward(model, tokens, logits);
        }
        
        /* Benchmark Forward */
        double start = get_time();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            kmamba_forward(model, tokens, logits);
        }
        double end = get_time();
        
        double elapsed = end - start;
        double tokens_per_sec = seq_len / elapsed * BENCHMARK_ITERATIONS;
        double params = 256 * dim + n_layers * (4 * dim * state_size + 2 * dim) + dim * 256;
        
        printf("KMamba %s (dim=%zu, layers=%zu): %.3f sec, %.0f tokens/sec, %.0f params\n",
               config_names[c], dim, n_layers, elapsed, tokens_per_sec, params);
        
        free(tokens); free(logits);
        kmamba_free(model);
    }
}

/* ============================================================
 * Benchmarks Training
 * ============================================================ */

static void benchmark_training() {
    printf("\n=== Training Benchmarks ===\n");
    
    KMambaConfig config = {
        .vocab_size = 256,
        .dim = 384,
        .state_size = 1024,
        .seq_len = 128,
        .n_layers = 2,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    
    KMamba *model = kmamba_create(&config);
    kmamba_init(model, 1234);
    
    MBOptimConfig opt_config = {
        .lr = 1e-3f,
        .mu = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .clip_norm = 1.0f,
        .weight_decay = 1e-5f
    };
    
    kmamba_enable_training(model, &opt_config, 1e-3f, 1e-5f);
    
    uint8_t *tokens = (uint8_t*)malloc((config.seq_len + 1) * sizeof(uint8_t));
    for (size_t i = 0; i < config.seq_len + 1; i++) {
        tokens[i] = (uint8_t)(rand() % 256);
    }
    
    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        kmamba_train_step(model, tokens);
    }
    
    /* Benchmark Training Step */
    double start = get_time();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        kmamba_train_step(model, tokens);
    }
    double end = get_time();
    
    double elapsed = end - start;
    double tokens_per_sec = config.seq_len / elapsed * BENCHMARK_ITERATIONS;
    
    printf("Training Step: %.3f sec, %.0f tokens/sec\n", elapsed, tokens_per_sec);
    
    /* Benchmark Batch Training */
    const size_t batch_sizes[] = {1, 4, 8, 16};
    
    for (int b = 0; b < 4; b++) {
        size_t batch_size = batch_sizes[b];
        uint8_t *batch_tokens = (uint8_t*)malloc(batch_size * (config.seq_len + 1) * sizeof(uint8_t));
        
        for (size_t i = 0; i < batch_size * (config.seq_len + 1); i++) {
            batch_tokens[i] = (uint8_t)(rand() % 256);
        }
        
        start = get_time();
        for (int i = 0; i < BENCHMARK_ITERATIONS / 4; i++) {
            kmamba_train_batch(model, batch_tokens, batch_size);
        }
        end = get_time();
        
        elapsed = end - start;
        tokens_per_sec = (batch_size * config.seq_len) / elapsed * (BENCHMARK_ITERATIONS / 4);
        
        printf("Batch Training (size=%zu): %.3f sec, %.0f tokens/sec\n",
               batch_size, elapsed, tokens_per_sec);
        
        free(batch_tokens);
    }
    
    free(tokens);
    kmamba_free(model);
}

/* ============================================================
 * Benchmarks Memory Usage
 * ============================================================ */

static void benchmark_memory_usage() {
    printf("\n=== Memory Usage Benchmarks ===\n");
    
    const size_t configs[][3] = {
        {64, 128, 32},
        {128, 256, 64},
        {256, 512, 128},
        {384, 1024, 128}
    };
    
    const char *config_names[] = {"Small", "Medium", "Large", "Current"};
    
    for (int c = 0; c < 4; c++) {
        size_t dim = configs[c][0], state_size = configs[c][1], seq_len = configs[c][2];
        
        /* Estimer l'utilisation mémoire */
        size_t embedding_size = 256 * dim * sizeof(float);
        size_t mamba_block_size = (4 * dim * state_size + 2 * dim) * sizeof(float);
        size_t lm_head_size = dim * 256 * sizeof(float);
        size_t activation_size = seq_len * dim * sizeof(float);
        size_t state_size_mem = state_size * dim * sizeof(float);
        
        size_t total_model_memory = embedding_size + mamba_block_size + lm_head_size;
        size_t total_inference_memory = total_model_memory + activation_size + state_size_mem;
        size_t total_training_memory = total_inference_memory * 3;  // Approximation
        
        printf("Memory %s (dim=%zu, seq=%zu):\n", config_names[c], dim, seq_len);
        printf("  Model: %.2f MB\n", total_model_memory / (1024.0 * 1024.0));
        printf("  Inference: %.2f MB\n", total_inference_memory / (1024.0 * 1024.0));
        printf("  Training: %.2f MB\n", total_training_memory / (1024.0 * 1024.0));
    }
}

/* ============================================================
 * Tests de Régression Performance
 * ============================================================ */

static int test_performance_regression() {
    printf("\n=== Performance Regression Tests ===\n");
    
    /* Baseline performance targets */
    struct {
        const char *name;
        double min_gflops;
        double max_latency_ms;
    } targets[] = {
        {"GEMM 512x512x512", 40.0, 5.0},
        {"GEMV 4096x4096", 15.0, 2.0},
        {"SiLU 64K", 30.0, 1.0},
        {"Scan1D Medium", 5.0, 10.0},
        {"MambaBlock Current", 1000.0, 50.0}  // tokens/sec
    };
    
    int passed = 0;
    
    /* Test GEMM performance */
    {
        const size_t m = 512, k = 512, n = 512;
        float *A = (float*)malloc(m * k * sizeof(float));
        float *B = (float*)malloc(k * n * sizeof(float));
        float *C = (float*)malloc(m * n * sizeof(float));
        
        fill_random(A, m * k, -1.0f, 1.0f);
        fill_random(B, k * n, -1.0f, 1.0f);
        
        double start = get_time();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            gemm_avx2(A, B, C, m, k, n);
        }
        double end = get_time();
        
        double elapsed = end - start;
        double ops = 2.0 * m * k * n * BENCHMARK_ITERATIONS;
        double gflops = ops / (elapsed * 1e9);
        double latency = elapsed / BENCHMARK_ITERATIONS * 1000;
        
        printf("GEMM Performance: %.2f GFLOPS, %.3f ms\n", gflops, latency);
        
        if (gflops >= targets[0].min_gflops && latency <= targets[0].max_latency_ms) {
            printf("✅ GEMM performance target met\n");
            passed++;
        } else {
            printf("❌ GEMM performance target missed\n");
        }
        
        free(A); free(B); free(C);
    }
    
    /* Test MambaBlock throughput */
    {
        KMambaConfig config = {
            .vocab_size = 256,
            .dim = 384,
            .state_size = 1024,
            .seq_len = 128,
            .n_layers = 2,
            .dt_scale = 1.0f,
            .dt_min = 0.001f,
            .dt_max = 0.1f
        };
        
        KMamba *model = kmamba_create(&config);
        kmamba_init(model, 1234);
        
        uint8_t *tokens = (uint8_t*)malloc(config.seq_len * sizeof(uint8_t));
        float *logits = (float*)malloc(config.seq_len * 256 * sizeof(float));
        
        for (size_t i = 0; i < config.seq_len; i++) {
            tokens[i] = (uint8_t)(rand() % 256);
        }
        
        double start = get_time();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            kmamba_forward(model, tokens, logits);
        }
        double end = get_time();
        
        double elapsed = end - start;
        double tokens_per_sec = config.seq_len / elapsed * BENCHMARK_ITERATIONS;
        double latency = elapsed / BENCHMARK_ITERATIONS * 1000;
        
        printf("KMamba Performance: %.0f tokens/sec, %.3f ms\n", tokens_per_sec, latency);
        
        if (tokens_per_sec >= targets[4].min_gflops && latency <= targets[4].max_latency_ms) {
            printf("✅ KMamba performance target met\n");
            passed++;
        } else {
            printf("❌ KMamba performance target missed\n");
        }
        
        free(tokens); free(logits);
        kmamba_free(model);
    }
    
    return passed == 2;
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== k-mamba Performance Benchmark Suite ===\n");
    printf("Testing performance regression and benchmarks\n\n");
    
    srand(42); /* Pour reproductibilité */
    
    /* Benchmarks individuels */
    benchmark_gemm_kernels();
    benchmark_gemv_kernels();
    benchmark_activation_kernels();
    benchmark_scan1d_kernels();
    benchmark_mamba_block();
    benchmark_kmamba_complete();
    benchmark_training();
    benchmark_memory_usage();
    
    /* Tests de régression */
    int regression_passed = test_performance_regression();
    
    printf("\n=== Summary ===\n");
    if (regression_passed) {
        printf("✅ All performance regression tests PASSED!\n");
        printf("✅ Performance targets maintained\n");
        return 0;
    } else {
        printf("❌ Some performance regression tests FAILED!\n");
        return 1;
    }
}
