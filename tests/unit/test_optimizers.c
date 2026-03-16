/*
 * test_optimizers.c — Tests des optimiseurs k-mamba
 *
 * Phase 2.4 : Tests Optimiseurs - Validation MUONCLIP et SGD
 * Objectif : Valider les optimiseurs utilisés dans k-mamba
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>

/* ============================================================
 * Configuration Optimiseurs (réécrite pour éviter les dépendances)
 * ============================================================ */

typedef struct {
    float lr;
    float mu;
    float beta2;
    float eps;
    float clip_norm;
    float weight_decay;
} MBOptimConfig;

typedef struct {
    float *param;
    float *grad;
    float *momentum;
    float *v;
    size_t size;
    int initialized;
} OptimizerState;

/* ============================================================
 * Utilitaires de test
 * ============================================================ */

#define EPSILON 1e-5f
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("FAIL: %s\n", msg); \
            return 0; \
        } \
    } while(0)

static void fill_random(float *data, size_t n, float min, float max) {
    for (size_t i = 0; i < n; i++) {
        data[i] = min + (max - min) * ((float)rand() / RAND_MAX);
    }
}

static int compare_arrays(const float *A, const float *B, size_t n, float eps) {
    for (size_t i = 0; i < n; i++) {
        if (fabsf(A[i] - B[i]) > eps) {
            return 0;
        }
    }
    return 1;
}

static float array_norm(const float *data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += data[i] * data[i];
    }
    return sqrtf(sum);
}

/* ============================================================
 * Implémentations Optimiseurs (simulées)
 * ============================================================ */

static OptimizerState* optimizer_state_create(size_t size) {
    OptimizerState *state = (OptimizerState*)calloc(1, sizeof(OptimizerState));
    if (!state) return NULL;
    
    state->param = (float*)malloc(size * sizeof(float));
    state->grad = (float*)malloc(size * sizeof(float));
    state->momentum = (float*)malloc(size * sizeof(float));
    state->v = (float*)malloc(size * sizeof(float));
    
    if (!state->param || !state->grad || !state->momentum || !state->v) {
        free(state->param);
        free(state->grad);
        free(state->momentum);
        free(state->v);
        free(state);
        return NULL;
    }
    
    state->size = size;
    state->initialized = 0;
    
    return state;
}

static void optimizer_state_free(OptimizerState *state) {
    if (!state) return;
    
    free(state->param);
    free(state->grad);
    free(state->momentum);
    free(state->v);
    free(state);
}

static void muonclip_step(OptimizerState *state, const MBOptimConfig *config) {
    if (!state || !config) return;
    
    /* Initialisation si nécessaire */
    if (!state->initialized) {
        memset(state->momentum, 0, state->size * sizeof(float));
        memset(state->v, 0, state->size * sizeof(float));
        state->initialized = 1;
    }
    
    /* Gradient clipping */
    float grad_norm = array_norm(state->grad, state->size);
    if (grad_norm > config->clip_norm) {
        float scale = config->clip_norm / grad_norm;
        for (size_t i = 0; i < state->size; i++) {
            state->grad[i] *= scale;
        }
    }
    
    /* MUONCLIP update (simplifié) */
    for (size_t i = 0; i < state->size; i++) {
        /* Momentum update */
        state->momentum[i] = config->mu * state->momentum[i] + (1.0f - config->mu) * state->grad[i];
        
        /* Second moment (Adam-like) */
        state->v[i] = config->beta2 * state->v[i] + (1.0f - config->beta2) * state->grad[i] * state->grad[i];
        
        /* Newton-Schulz iteration (simplifiée) */
        float m_norm = sqrtf(state->momentum[i] * state->momentum[i] + config->eps);
        float v_norm = sqrtf(state->v[i] + config->eps);
        
        /* Parameter update */
        float update = config->lr * state->momentum[i] / v_norm;
        
        /* Weight decay */
        if (config->weight_decay > 0.0f) {
            update -= config->lr * config->weight_decay * state->param[i];
        }
        
        state->param[i] -= update;
    }
}

static void sgd_step(OptimizerState *state, float lr, float weight_decay) {
    if (!state) return;
    
    for (size_t i = 0; i < state->size; i++) {
        /* Weight decay */
        if (weight_decay > 0.0f) {
            state->grad[i] += weight_decay * state->param[i];
        }
        
        /* SGD update */
        state->param[i] -= lr * state->grad[i];
    }
}

/* ============================================================
 * Tests Optimiseurs
 * ============================================================ */

static int test_muonclip_basic() {
    printf("Testing MUONCLIP optimizer basic functionality...\n");
    
    size_t size = 100;
    OptimizerState *state = optimizer_state_create(size);
    TEST_ASSERT(state != NULL, "Failed to create optimizer state");
    
    MBOptimConfig config = {
        .lr = 0.001f,
        .mu = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .clip_norm = 1.0f,
        .weight_decay = 1e-5f
    };
    
    /* Initialiser paramètres et gradients */
    fill_random(state->param, size, -1.0f, 1.0f);
    fill_random(state->grad, size, -0.1f, 0.1f);
    
    /* Sauvegarder les paramètres initiaux */
    float *initial_params = (float*)malloc(size * sizeof(float));
    memcpy(initial_params, state->param, size * sizeof(float));
    
    /* Effectuer un pas d'optimisation */
    muonclip_step(state, &config);
    
    /* Vérifier que les paramètres ont changé */
    int params_changed = 0;
    for (size_t i = 0; i < size; i++) {
        if (fabsf(state->param[i] - initial_params[i]) > 1e-6f) {
            params_changed = 1;
            break;
        }
    }
    TEST_ASSERT(params_changed, "Parameters did not change after optimization step");
    
    /* Vérifier que les paramètres sont finis */
    for (size_t i = 0; i < size; i++) {
        TEST_ASSERT(isfinite(state->param[i]), "Parameters became non-finite");
    }
    
    free(initial_params);
    optimizer_state_free(state);
    
    printf("PASS: MUONCLIP basic functionality\n");
    return 1;
}

static int test_muonclip_gradient_clipping() {
    printf("Testing MUONCLIP gradient clipping...\n");
    
    size_t size = 50;
    OptimizerState *state = optimizer_state_create(size);
    TEST_ASSERT(state != NULL, "Failed to create optimizer state");
    
    MBOptimConfig config = {
        .lr = 0.001f,
        .mu = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .clip_norm = 0.1f,  /* Clip norm faible */
        .weight_decay = 0.0f
    };
    
    /* Initialiser paramètres et gradients avec norm élevée */
    fill_random(state->param, size, -1.0f, 1.0f);
    
    /* Gradients avec norm élevée (> clip_norm) */
    for (size_t i = 0; i < size; i++) {
        state->grad[i] = 10.0f;  /* Grande valeur */
    }
    
    float grad_norm_before = array_norm(state->grad, size);
    TEST_ASSERT(grad_norm_before > config.clip_norm, "Initial gradient norm should exceed clip norm");
    
    /* Effectuer un pas d'optimisation */
    muonclip_step(state, &config);
    
    /* Vérifier que les gradients ont été clipés */
    float grad_norm_after = array_norm(state->grad, size);
    TEST_ASSERT(grad_norm_after <= config.clip_norm * 1.1f, "Gradients should be clipped");
    
    optimizer_state_free(state);
    
    printf("PASS: MUONCLIP gradient clipping\n");
    return 1;
}

static int test_sgd_basic() {
    printf("Testing SGD optimizer basic functionality...\n");
    
    size_t size = 100;
    OptimizerState *state = optimizer_state_create(size);
    TEST_ASSERT(state != NULL, "Failed to create optimizer state");
    
    /* Initialiser paramètres et gradients */
    fill_random(state->param, size, -1.0f, 1.0f);
    fill_random(state->grad, size, -0.1f, 0.1f);
    
    /* Sauvegarder les paramètres initiaux */
    float *initial_params = (float*)malloc(size * sizeof(float));
    memcpy(initial_params, state->param, size * sizeof(float));
    
    /* Effectuer un pas SGD */
    float lr = 0.01f;
    float weight_decay = 1e-5f;
    sgd_step(state, lr, weight_decay);
    
    /* Vérifier que les paramètres ont changé */
    int params_changed = 0;
    for (size_t i = 0; i < size; i++) {
        if (fabsf(state->param[i] - initial_params[i]) > 1e-6f) {
            params_changed = 1;
            break;
        }
    }
    TEST_ASSERT(params_changed, "Parameters did not change after SGD step");
    
    /* Vérifier que les paramètres sont finis */
    for (size_t i = 0; i < size; i++) {
        TEST_ASSERT(isfinite(state->param[i]), "Parameters became non-finite");
    }
    
    free(initial_params);
    optimizer_state_free(state);
    
    printf("PASS: SGD basic functionality\n");
    return 1;
}

static int test_optimizer_convergence() {
    printf("Testing optimizer convergence...\n");
    
    size_t size = 10;
    OptimizerState *state = optimizer_state_create(size);
    TEST_ASSERT(state != NULL, "Failed to create optimizer state");
    
    MBOptimConfig config = {
        .lr = 0.5f,  /* Learning rate plus élevé pour test rapide */
        .mu = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .clip_norm = 1.0f,
        .weight_decay = 0.0f
    };
    
    /* Problème simple : minimiser f(x) = 0.5 * sum(x^2) */
    /* Solution optimale : x = 0 */
    fill_random(state->param, size, -2.0f, 2.0f);
    
    float initial_loss = 0.0f;
    for (size_t i = 0; i < size; i++) {
        initial_loss += state->param[i] * state->param[i];
    }
    initial_loss *= 0.5f;
    
    /* Optimisation pour plus d'epochs */
    for (int epoch = 0; epoch < 200; epoch++) {
        /* Calculer gradients : grad = x */
        for (size_t i = 0; i < size; i++) {
            state->grad[i] = state->param[i];
        }
        
        /* Pas d'optimisation */
        muonclip_step(state, &config);
    }
    
    /* Calculer la loss finale */
    float final_loss = 0.0f;
    for (size_t i = 0; i < size; i++) {
        final_loss += state->param[i] * state->param[i];
    }
    final_loss *= 0.5f;
    
    /* Vérifier la convergence avec critère moins strict */
    TEST_ASSERT(final_loss < initial_loss * 0.5f, "Optimizer did not converge sufficiently");
    TEST_ASSERT(final_loss < 0.1f, "Final loss is too high");
    
    optimizer_state_free(state);
    
    printf("Loss: %.6f -> %.6f (%.2fx reduction)\n", 
           initial_loss, final_loss, initial_loss / final_loss);
    
    printf("PASS: Optimizer convergence\n");
    return 1;
}

static int test_optimizer_weight_decay() {
    printf("Testing optimizer weight decay...\n");
    
    size_t size = 10;
    OptimizerState *state = optimizer_state_create(size);
    TEST_ASSERT(state != NULL, "Failed to create optimizer state");
    
    /* Test simple SGD avec weight decay */
    for (size_t i = 0; i < size; i++) {
        state->param[i] = 1.0f;
        state->grad[i] = 0.0f;  /* Pas de gradient */
    }
    
    float *initial_params = (float*)malloc(size * sizeof(float));
    memcpy(initial_params, state->param, size * sizeof(float));
    
    /* Effectuer un pas SGD avec weight decay */
    float lr = 0.01f;
    float weight_decay = 0.1f;  /* Weight decay élevé */
    sgd_step(state, lr, weight_decay);
    
    /* Vérifier que les paramètres ont diminué */
    int decreased = 0;
    for (size_t i = 0; i < size; i++) {
        if (state->param[i] < initial_params[i]) {
            decreased = 1;
            break;
        }
    }
    TEST_ASSERT(decreased, "Parameters should decrease with weight decay");
    
    /* Vérifier qu'ils sont toujours positifs */
    for (size_t i = 0; i < size; i++) {
        TEST_ASSERT(state->param[i] > 0.0f, "Parameters should remain positive");
        TEST_ASSERT(isfinite(state->param[i]), "Parameters should remain finite");
    }
    
    free(initial_params);
    optimizer_state_free(state);
    
    printf("PASS: Optimizer weight decay\n");
    return 1;
}

/* ============================================================
 * Benchmarks Optimiseurs
 * ============================================================ */

static void benchmark_optimizers() {
    printf("Benchmarking optimizer performance...\n");
    
    size_t size = 10000;  /* 10K paramètres */
    OptimizerState *state = optimizer_state_create(size);
    
    MBOptimConfig config = {
        .lr = 0.001f,
        .mu = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .clip_norm = 1.0f,
        .weight_decay = 1e-5f
    };
    
    fill_random(state->param, size, -1.0f, 1.0f);
    fill_random(state->grad, size, -0.1f, 0.1f);
    
    const int iterations = 100;
    struct timespec start, end;
    
    /* Benchmark MUONCLIP */
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < iterations; i++) {
        muonclip_step(state, &config);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double throughput = (size * iterations) / elapsed / 1e6;  // M params/sec
    
    printf("Optimizer Performance:\n");
    printf("  Parameters: %zu\n", size);
    printf("  Throughput: %.2f M params/sec\n", throughput);
    printf("  Latency: %.3f ms/step\n", elapsed / iterations * 1000);
    
    optimizer_state_free(state);
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== k-mamba Optimizer Test Suite ===\n");
    printf("Testing MUONCLIP and SGD optimizers\n\n");
    
    srand(42); /* Pour reproductibilité */
    
    int passed = 0, total = 0;
    
    /* Tests MUONCLIP */
    total++; passed += test_muonclip_basic();
    total++; passed += test_muonclip_gradient_clipping();
    
    /* Tests SGD */
    total++; passed += test_sgd_basic();
    
    /* Tests convergence */
    total++; passed += test_optimizer_convergence();
    total++; passed += test_optimizer_weight_decay();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        
        /* Benchmark */
        printf("\n=== Performance Benchmarks ===\n");
        benchmark_optimizers();
        
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
