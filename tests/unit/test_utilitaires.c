/*
 * test_utilitaires.c — Tests unitaires utilitaires optimatrix
 *
 * Phase 1.5 : Tests Utilitaires - Hadamard, Gradient Clipping, Vector Ops
 * Objectif : Valider les opérations utilitaires de base disponibles
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "optimatrix.h"

/* ============================================================
 * Utilitaires de test
 * ============================================================ */

#define EPSILON 1e-6f
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("FAIL: %s\n", msg); \
            return 0; \
        } \
    } while(0)

#define TEST_ASSERT_FLOAT_EQ(a, b, eps, msg) \
    TEST_ASSERT(fabsf((a) - (b)) < (eps), msg)

static float frand_range(float min, float max) {
    float scale = (float)rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

static void fill_random(float *data, size_t n, float min, float max) {
    for (size_t i = 0; i < n; i++) {
        data[i] = frand_range(min, max);
    }
}

static int compare_arrays(const float *A, const float *B, size_t n, float eps) {
    for (size_t i = 0; i < n; i++) {
        if (fabsf(A[i] - B[i]) > eps) {
            printf("Mismatch at [%zu]: %f vs %f (diff: %f)\n", 
                   i, A[i], B[i], fabsf(A[i] - B[i]));
            return 0;
        }
    }
    return 1;
}

/* ============================================================
 * Références C pures
 * ============================================================ */

static void hadamard_reference(const float *A, const float *B, float *C, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] * B[i];
    }
}

static void vector_add_reference(const float *A, const float *B, float *C, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

static void vector_scale_reference(const float *A, float alpha, float *C, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] * alpha;
    }
}

static float gradient_norm_reference(const float *grad, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += grad[i] * grad[i];
    }
    return sqrtf(sum);
}

static void gradient_clip_reference(float *grad, size_t n, float max_norm) {
    float norm = gradient_norm_reference(grad, n);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (size_t i = 0; i < n; i++) {
            grad[i] *= scale;
        }
    }
}

/* ============================================================
 * Tests Gradient Clipping (disponible dans optimatrix)
 * ============================================================ */

static int test_gradient_norm() {
    printf("Testing gradient norm computation...\n");
    
    /* Test simple */
    {
        float grad[4] = {3.0f, 4.0f, 0.0f, 0.0f};  // norm = 5
        float norm_test = gradient_norm(grad, 4);
        float expected = 5.0f;
        
        TEST_ASSERT_FLOAT_EQ(norm_test, expected, EPSILON, "Gradient norm simple failed");
    }
    
    /* Test vecteur aléatoire */
    {
        const size_t n = 100;
        float *grad = (float*)malloc(n * sizeof(float));
        fill_random(grad, n, -5.0f, 5.0f);
        
        float norm_test = gradient_norm(grad, n);
        
        TEST_ASSERT(isfinite(norm_test), "Gradient norm not finite");
        TEST_ASSERT(norm_test >= 0.0f, "Gradient norm negative");
        
        free(grad);
    }
    
    /* Test vecteur vide */
    {
        float norm_test = gradient_norm(NULL, 0);
        TEST_ASSERT_FLOAT_EQ(norm_test, 0.0f, EPSILON, "Gradient norm empty failed");
    }
    
    printf("PASS: Gradient norm computation\n");
    return 1;
}

static int test_gradient_clipping() {
    printf("Testing gradient clipping...\n");
    
    /* Test 1: Pas de clipping nécessaire */
    {
        float grad[4] = {0.1f, 0.2f, 0.3f, 0.4f};
        float max_norm = 2.0f;
        
        float norm_before = gradient_norm(grad, 4);
        gradient_clip_inplace(grad, 4, max_norm);
        float norm_after = gradient_norm(grad, 4);
        
        TEST_ASSERT_FLOAT_EQ(norm_before, norm_after, EPSILON, "No clipping should change norm");
    }
    
    /* Test 2: Clipping nécessaire */
    {
        float grad[4] = {3.0f, 4.0f, 0.0f, 0.0f};  // norm = 5
        float max_norm = 2.0f;
        
        gradient_clip_inplace(grad, 4, max_norm);
        float norm_after = gradient_norm(grad, 4);
        
        TEST_ASSERT(norm_after <= max_norm + EPSILON, "Clipping should reduce norm");
    }
    
    /* Test 3: Clipping avec zéros */
    {
        float grad[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float max_norm = 1.0f;
        
        gradient_clip_inplace(grad, 4, max_norm);
        
        float norm_after = gradient_norm(grad, 4);
        TEST_ASSERT_FLOAT_EQ(norm_after, 0.0f, EPSILON, "Zero gradient should stay zero");
    }
    
    printf("PASS: Gradient clipping\n");
    return 1;
}

/* ============================================================
 * Tests Vector Operations (disponibles dans optimatrix)
 * ============================================================ */

static int test_vector_add() {
    printf("Testing vector addition...\n");
    
    const size_t n = 256;
    float *A = (float*)malloc(n * sizeof(float));
    float *B = (float*)malloc(n * sizeof(float));
    float *C = (float*)malloc(n * sizeof(float));
    float *C_ref = (float*)malloc(n * sizeof(float));
    
    fill_random(A, n, -3.0f, 3.0f);
    fill_random(B, n, -3.0f, 3.0f);
    
    /* Test avec la fonction optimatrix */
    vector_add(A, B, C, n);
    
    /* Référence manuelle */
    for (size_t i = 0; i < n; i++) {
        C_ref[i] = A[i] + B[i];
    }
    
    int result = compare_arrays(C, C_ref, n, EPSILON);
    TEST_ASSERT(result, "Vector addition failed");
    
    free(A); free(B); free(C); free(C_ref);
    printf("PASS: Vector addition\n");
    return 1;
}

static int test_vector_scale() {
    printf("Testing vector scaling...\n");
    
    const size_t n = 512;
    float alpha = 2.5f;
    float *A = (float*)malloc(n * sizeof(float));
    float *C = (float*)malloc(n * sizeof(float));
    float *C_ref = (float*)malloc(n * sizeof(float));
    
    fill_random(A, n, -2.0f, 2.0f);
    
    /* Test avec la fonction optimatrix */
    vector_scale(A, alpha, C, n);
    
    /* Référence manuelle */
    for (size_t i = 0; i < n; i++) {
        C_ref[i] = A[i] * alpha;
    }
    
    int result = compare_arrays(C, C_ref, n, EPSILON);
    TEST_ASSERT(result, "Vector scaling failed");
    
    free(A); free(C); free(C_ref);
    printf("PASS: Vector scaling\n");
    return 1;
}

/* ============================================================
 * Tests Hadamard (disponible dans optimatrix)
 * ============================================================ */

static int test_hadamard_basic() {
    printf("Testing Hadamard product basic...\n");
    
    const size_t n = 4;
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float C[4];
    float C_ref[4];
    
    /* Test avec la fonction optimatrix */
    hadamard(A, B, C, n);
    
    /* Référence manuelle */
    for (size_t i = 0; i < n; i++) {
        C_ref[i] = A[i] * B[i];
    }
    
    int result = compare_arrays(C, C_ref, n, EPSILON);
    TEST_ASSERT(result, "Hadamard product basic failed");
    
    printf("PASS: Hadamard product basic\n");
    return 1;
}

/* ============================================================
 * Benchmarks simples
 * ============================================================ */

static void benchmark_vector_operations() {
    printf("Benchmarking vector operations...\n");
    
    const size_t n = 1024 * 1024;  // 1M elements
    const int iterations = 1000;
    
    float *A = (float*)malloc(n * sizeof(float));
    float *B = (float*)malloc(n * sizeof(float));
    float *C = (float*)malloc(n * sizeof(float));
    
    fill_random(A, n, -1.0f, 1.0f);
    fill_random(B, n, -1.0f, 1.0f);
    
    struct timespec start, end;
    
    /* Benchmark vector_add */
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        vector_add(A, B, C, n);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double throughput = (n * iterations) / (elapsed * 1e9);  // G elems/sec
    
    printf("Vector Add: %.3f sec (%.2f G elems/sec)\n", elapsed, throughput);
    
    free(A); free(B); free(C);
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== optimatrix Utilitaires Test Suite ===\n");
    printf("Testing basic utilitaires available in optimatrix\n\n");
    
    srand(42); /* Pour reproductibilité */
    
    int passed = 0, total = 0;
    
    /* Tests Gradient Clipping */
    total++; passed += test_gradient_norm();
    total++; passed += test_gradient_clipping();
    
    /* Tests Vector Operations */
    total++; passed += test_vector_add();
    total++; passed += test_vector_scale();
    
    /* Tests Hadamard */
    total++; passed += test_hadamard_basic();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        
        /* Benchmark */
        printf("\n=== Performance Benchmarks ===\n");
        benchmark_vector_operations();
        
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
