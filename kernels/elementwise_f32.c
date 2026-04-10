/*
 * elementwise_f32.c — Elementwise operations in pure C with OpenMP
 */

#include "kmamba_kernels.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void hadamard_f32(const float *x, const float *y, float *z, long n) {
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
}

void vec_add_f32(const float *a, const float *b, float *out, long n) {
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void vec_scale_f32(const float *a, float scale, float *out, long n) {
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        out[i] = a[i] * scale;
    }
}

void vec_copy_f32(const float *src, float *dst, long n) {
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

void vec_set_f32(float *a, float val, long n) {
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        a[i] = val;
    }
}
