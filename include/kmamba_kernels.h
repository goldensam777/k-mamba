/*
 * kmamba_kernels.h — Zero-dependency compute kernels for k-mamba
 *
 * Remplacement complet de OpenBLAS + optimatrix
 * GEMM/GEMV en C pur (fallback) + AVX2 inline (optionnel)
 * Activations, Hadamard, Optimiseurs inline
 */

#ifndef KMAMBA_KERNELS_H
#define KMAMBA_KERNELS_H

#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * BLAS-like Operations (C reference implementation)
 * ============================================================ */

/* GEMM: C[M,N] = A[M,K] @ B[K,N] (row-major) */
void gemm_f32(const float *A, const float *B, float *C,
              int M, int K, int N);

/* GEMV: y[M] = A[M,N] @ x[N] (row-major) */
void gemv_f32(const float *A, const float *x, float *y,
              int M, int N);

/* ============================================================
 * Activations (inline for performance)
 * ============================================================ */

static inline float silu_scalar_f32(float x) {
    return x / (1.0f + expf(-x));
}

static inline float relu_scalar_f32(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float sigmoid_scalar_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float softplus_scalar_f32(float x) {
    return logf(1.0f + expf(x));
}

/* Vector versions */
void silu_f32(const float *x, float *y, long n);
void relu_f32(const float *x, float *y, long n);
void sigmoid_f32(const float *x, float *y, long n);
void softplus_f32(const float *x, float *y, long n);

/* ============================================================
 * Optimizer Utilities
 * ============================================================ */

/* L2 norm of gradient */
float gradient_norm_f32(const float *grad, size_t n);

/* Clip gradient in-place */
void gradient_clip_inplace_f32(float *grad, size_t n, float max_norm);

/* Newton-Schulz orthogonalization for MUON */
void newton_schulz5_inplace_f32(float *G, size_t rows, size_t cols, int steps);

/* AdamW step */
void adamw_step_f32(float *param, const float *grad,
                    float *m, float *v,
                    float lr, float beta1, float beta2,
                    float eps, float wd,
                    size_t n, int t);

/* MUON optimizer config */
typedef struct {
    float lr;
    float mu;
    float beta2;
    float eps;
    float clip_norm;
    float weight_decay;
} MBOptimConfig;

/* MUON update for matrix (with Newton-Schulz) */
void muon_update_mat_f32(float *param, const float *grad, float *m,
                         size_t rows, size_t cols, const MBOptimConfig *conf, int t);

/* MUON update for vector (no orthogonalization) */
void muon_update_vec_f32(float *param, const float *grad, float *m,
                         size_t n, const MBOptimConfig *conf, int t);

/* ============================================================
 * Initialization
 * ============================================================ */

void init_xavier_uniform_f32(float *W, size_t fan_in, size_t fan_out, unsigned int seed);
void init_kaiming_uniform_f32(float *W, size_t fan_in, unsigned int seed);

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_KERNELS_H */
