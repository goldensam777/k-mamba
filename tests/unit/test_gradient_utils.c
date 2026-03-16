/*
 * test_gradient_utils.c — Tests des utilitaires de gradient optimatrix
 *
 * Test des fonctions disponibles dans optimatrix/src/optimizer_utils.c
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

/* ============================================================
 * Tests Gradient Norm
 * ============================================================ */

static int test_gradient_norm_basic() {
    printf("Testing gradient norm basic cases...\n");
    
    /* Test 1: Vecteur simple (3,4) -> norm = 5 */
    {
        float grad[2] = {3.0f, 4.0f};
        float norm = gradient_norm(grad, 2);
        float expected = 5.0f;
        
        TEST_ASSERT_FLOAT_EQ(norm, expected, EPSILON, "Basic gradient norm failed");
    }
    
    /* Test 2: Vecteur nul */
    {
        float grad[3] = {0.0f, 0.0f, 0.0f};
        float norm = gradient_norm(grad, 3);
        float expected = 0.0f;
        
        TEST_ASSERT_FLOAT_EQ(norm, expected, EPSILON, "Zero gradient norm failed");
    }
    
    /* Test 3: Vecteur unitaire */
    {
        float grad[1] = {1.0f};
        float norm = gradient_norm(grad, 1);
        float expected = 1.0f;
        
        TEST_ASSERT_FLOAT_EQ(norm, expected, EPSILON, "Unit gradient norm failed");
    }
    
    printf("PASS: Gradient norm basic cases\n");
    return 1;
}

static int test_gradient_norm_random() {
    printf("Testing gradient norm random vectors...\n");
    
    const size_t n = 100;
    float *grad = (float*)malloc(n * sizeof(float));
    
    fill_random(grad, n, -5.0f, 5.0f);
    
    float norm = gradient_norm(grad, n);
    
    /* Vérifications basiques */
    TEST_ASSERT(isfinite(norm), "Gradient norm should be finite");
    TEST_ASSERT(norm >= 0.0f, "Gradient norm should be non-negative");
    
    /* Calcul manuel pour vérifier */
    double manual_sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        manual_sum += (double)grad[i] * (double)grad[i];
    }
    double manual_norm = sqrt(manual_sum);
    
    TEST_ASSERT_FLOAT_EQ(norm, (float)manual_norm, EPSILON, "Gradient norm computation mismatch");
    
    free(grad);
    printf("PASS: Gradient norm random vectors\n");
    return 1;
}

static int test_gradient_norm_edge_cases() {
    printf("Testing gradient norm edge cases...\n");
    
    /* Test vecteur vide */
    {
        float norm = gradient_norm(NULL, 0);
        TEST_ASSERT_FLOAT_EQ(norm, 0.0f, EPSILON, "Empty gradient norm should be 0");
    }
    
    /* Test grand vecteur */
    {
        const size_t n = 10000;
        float *grad = (float*)malloc(n * sizeof(float));
        
        /* Remplir avec 1.0f -> norm = sqrt(10000) = 100 */
        for (size_t i = 0; i < n; i++) {
            grad[i] = 1.0f;
        }
        
        float norm = gradient_norm(grad, n);
        float expected = 100.0f;
        
        TEST_ASSERT_FLOAT_EQ(norm, expected, 1e-4f, "Large gradient norm failed");
        
        free(grad);
    }
    
    printf("PASS: Gradient norm edge cases\n");
    return 1;
}

/* ============================================================
 * Tests Gradient Clipping
 * ============================================================ */

static int test_gradient_clipping_no_clip() {
    printf("Testing gradient clipping (no clipping needed)...\n");
    
    float grad[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float max_norm = 2.0f;
    
    /* Sauvegarder l'original */
    float original[4];
    memcpy(original, grad, 4 * sizeof(float));
    
    float norm_before = gradient_norm(grad, 4);
    gradient_clip_inplace(grad, 4, max_norm);
    float norm_after = gradient_norm(grad, 4);
    
    /* La norme ne devrait pas changer */
    TEST_ASSERT_FLOAT_EQ(norm_before, norm_after, EPSILON, "Norm should not change when no clipping needed");
    
    /* Les valeurs ne devraient pas changer */
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_EQ(grad[i], original[i], EPSILON, "Values should not change when no clipping needed");
    }
    
    printf("PASS: Gradient clipping (no clipping)\n");
    return 1;
}

static int test_gradient_clipping_with_clip() {
    printf("Testing gradient clipping (clipping needed)...\n");
    
    float grad[2] = {3.0f, 4.0f};  // norm = 5
    float max_norm = 2.0f;
    
    float norm_before = gradient_norm(grad, 2);
    gradient_clip_inplace(grad, 2, max_norm);
    float norm_after = gradient_norm(grad, 2);
    
    /* La norme devrait être réduite à max_norm */
    TEST_ASSERT(norm_after <= max_norm + EPSILON, "Norm should be clipped to max_norm");
    
    /* Vérifier que le scaling est correct */
    float expected_norm_before = 5.0f;
    float expected_scale = max_norm / expected_norm_before;
    TEST_ASSERT_FLOAT_EQ(norm_before, expected_norm_before, EPSILON, "Original norm should be 5.0");
    
    /* Les valeurs devraient être scalées */
    float expected_grad[2] = {3.0f * expected_scale, 4.0f * expected_scale};
    TEST_ASSERT_FLOAT_EQ(grad[0], expected_grad[0], EPSILON, "First element should be scaled correctly");
    TEST_ASSERT_FLOAT_EQ(grad[1], expected_grad[1], EPSILON, "Second element should be scaled correctly");
    
    printf("PASS: Gradient clipping (with clipping)\n");
    return 1;
}

static int test_gradient_clipping_edge_cases() {
    printf("Testing gradient clipping edge cases...\n");
    
    /* Test vecteur nul */
    {
        float grad[3] = {0.0f, 0.0f, 0.0f};
        float max_norm = 1.0f;
        
        gradient_clip_inplace(grad, 3, max_norm);
        float norm_after = gradient_norm(grad, 3);
        
        TEST_ASSERT_FLOAT_EQ(norm_after, 0.0f, EPSILON, "Zero gradient should stay zero");
    }
    
    /* Test max_norm = 0 (ne devrait rien faire) */
    {
        float grad[2] = {1.0f, 2.0f};
        float max_norm = 0.0f;
        
        float norm_before = gradient_norm(grad, 2);
        gradient_clip_inplace(grad, 2, max_norm);
        float norm_after = gradient_norm(grad, 2);
        
        TEST_ASSERT_FLOAT_EQ(norm_before, norm_after, EPSILON, "Zero max_norm should not change gradient");
    }
    
    /* Test vecteur vide */
    {
        gradient_clip_inplace(NULL, 0, 1.0f);
        printf("Empty vector clipping handled gracefully\n");
    }
    
    printf("PASS: Gradient clipping edge cases\n");
    return 1;
}

/* ============================================================
 * Test Gradient Clip (version non-inplace)
 * ============================================================ */

static int test_gradient_clip_copy() {
    printf("Testing gradient clipping (copy version)...\n");
    
    float grad[2] = {3.0f, 4.0f};  // norm = 5
    float grad_clipped[2];
    float max_norm = 2.0f;
    
    gradient_clip(grad, grad_clipped, 2, max_norm);
    
    /* Original ne devrait pas changer */
    float norm_original = gradient_norm(grad, 2);
    TEST_ASSERT_FLOAT_EQ(norm_original, 5.0f, EPSILON, "Original gradient should not change");
    
    /* Clipped devrait avoir norm = max_norm */
    float norm_clipped = gradient_norm(grad_clipped, 2);
    TEST_ASSERT(norm_clipped <= max_norm + EPSILON, "Clipped gradient should have norm <= max_norm");
    
    printf("PASS: Gradient clipping (copy version)\n");
    return 1;
}

/* ============================================================
 * Benchmarks
 * ============================================================ */

static void benchmark_gradient_operations() {
    printf("Benchmarking gradient operations...\n");
    
    const size_t n = 1000000;  // 1M elements
    const int iterations = 100;
    
    float *grad = (float*)malloc(n * sizeof(float));
    fill_random(grad, n, -1.0f, 1.0f);
    
    struct timespec start, end;
    
    /* Benchmark gradient_norm */
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        volatile float norm = gradient_norm(grad, n);
        (void)norm;  // Éviter l'optimisation
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double throughput = (n * iterations) / (elapsed * 1e9);  // G elems/sec
    
    printf("Gradient Norm: %.3f sec (%.2f G elems/sec)\n", elapsed, throughput);
    
    /* Benchmark gradient_clip_inplace */
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        gradient_clip_inplace(grad, n, 1.0f);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    throughput = (n * iterations) / (elapsed * 1e9);
    
    printf("Gradient Clip: %.3f sec (%.2f G elems/sec)\n", elapsed, throughput);
    
    free(grad);
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== Gradient Utils Test Suite ===\n");
    printf("Testing gradient utilities from optimatrix\n\n");
    
    srand(42); /* Pour reproductibilité */
    
    int passed = 0, total = 0;
    
    /* Tests Gradient Norm */
    total++; passed += test_gradient_norm_basic();
    total++; passed += test_gradient_norm_random();
    total++; passed += test_gradient_norm_edge_cases();
    
    /* Tests Gradient Clipping */
    total++; passed += test_gradient_clipping_no_clip();
    total++; passed += test_gradient_clipping_with_clip();
    total++; passed += test_gradient_clipping_edge_cases();
    
    /* Test Gradient Clip Copy */
    total++; passed += test_gradient_clip_copy();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        
        /* Benchmark */
        printf("\n=== Performance Benchmarks ===\n");
        benchmark_gradient_operations();
        
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
