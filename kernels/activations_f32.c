/*
 * activations_f32.c — Activations in pure C (zero dependency)
 */

#include "kmamba_kernels.h"

void silu_f32(const float *x, float *y, long n) {
    for (long i = 0; i < n; i++) {
        y[i] = silu_scalar_f32(x[i]);
    }
}

void relu_f32(const float *x, float *y, long n) {
    for (long i = 0; i < n; i++) {
        y[i] = relu_scalar_f32(x[i]);
    }
}

void sigmoid_f32(const float *x, float *y, long n) {
    for (long i = 0; i < n; i++) {
        y[i] = sigmoid_scalar_f32(x[i]);
    }
}

void softplus_f32(const float *x, float *y, long n) {
    for (long i = 0; i < n; i++) {
        y[i] = softplus_scalar_f32(x[i]);
    }
}
