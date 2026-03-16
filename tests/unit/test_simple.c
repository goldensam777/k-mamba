/*
 * test_simple.c — Tests simples pour valider le build
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    printf("=== Simple Test Suite ===\n");
    printf("Testing basic functionality\n\n");
    
    // Test 1: Math functions
    printf("Testing math functions...\n");
    float a = 3.0f, b = 4.0f;
    float c = sqrtf(a*a + b*b);
    printf("sqrt(3^2 + 4^2) = %.2f (expected: 5.00)\n", c);
    
    if (fabsf(c - 5.0f) < 1e-6f) {
        printf("PASS: Math functions\n");
    } else {
        printf("FAIL: Math functions\n");
        return 1;
    }
    
    // Test 2: Memory allocation
    printf("\nTesting memory allocation...\n");
    float *array = (float*)malloc(1000 * sizeof(float));
    if (array) {
        for (int i = 0; i < 1000; i++) {
            array[i] = (float)i * 0.001f;
        }
        float sum = 0.0f;
        for (int i = 0; i < 1000; i++) {
            sum += array[i];
        }
        printf("Sum of 0..999: %.2f\n", sum);
        free(array);
        printf("PASS: Memory allocation\n");
    } else {
        printf("FAIL: Memory allocation\n");
        return 1;
    }
    
    printf("\n=== Test Results ===\n");
    printf("All basic tests PASSED!\n");
    printf("Build system working correctly!\n");
    
    return 0;
}
