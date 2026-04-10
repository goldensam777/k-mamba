/*
 * gemm_f32.c — GEMM/GEMV in pure C with OpenMP
 */

#include "kmamba_kernels.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* GEMM: C[M,N] = A[M,K] @ B[K,N] (row-major) */
void gemm_f32(const float *A, const float *B, float *C,
              int M, int K, int N) {
    /* Zero C first */
    #pragma omp parallel for
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }
    
    /* C[i,j] = sum_k A[i,k] * B[k,j] */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

/* GEMM with accumulation: C += A @ B */
void gemm_accum_f32(const float *A, const float *B, float *C,
                    int M, int K, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

/* GEMV: y[M] = A[M,N] @ x[N] (row-major) */
void gemv_f32(const float *A, const float *x, float *y,
              int M, int N) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}
