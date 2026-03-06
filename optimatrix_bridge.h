/*
 * optimatrix_bridge.h — Interface C vers les kernels ASM x86-64 d'optimatrix
 *
 * Toutes les fonctions opèrent sur float32 (real_t = float).
 * Conventions : tableaux row-major, indices long.
 */

#ifndef OPTIMATRIX_BRIDGE_H
#define OPTIMATRIX_BRIDGE_H

/* ---- BLAS Niveau 2 : GEMV ----------------------------------------- */

/* y = A · x   (scalaire)
 * A : m×n float32 row-major, x : n, y : m
 */
void gemv(float *A, float *x, float *y, long m, long n);

/* y = A · x   (AVX2 — 8 floats/cycle)
 * Même interface que gemv, 8–12× plus rapide sur grandes matrices.
 */
void gemv_avx2(float *A, float *x, float *y, long m, long n);

/* ---- BLAS Niveau 3 : GEMM ----------------------------------------- */

/* C = A · B   (scalaire)
 * A : m×k, B : k×n, C : m×n  — tous float32 row-major
 */
void gemm(float *A, float *B, float *C, long m, long k, long n);

/* C = A · B   (AVX2)
 * Même interface, vectorisé sur n (8 floats/iter).
 */
void gemm_avx2(float *A, float *B, float *C, long m, long k, long n);

/* ---- Produit de Hadamard ------------------------------------------ */

/* z[i] = x[i] * y[i]   (scalaire) */
void hadamard(float *x, float *y, float *z, long n);

/* z[i] = x[i] * y[i]   (AVX2) */
void hadamard_avx2(float *x, float *y, float *z, long n);

/* ---- Fonctions d'activation --------------------------------------- */

/* y[i] = max(0, x[i]) */
void relu_f32(float *x, float *y, long n);

/* y[i] = 1 / (1 + expf(-x[i])) */
void sigmoid_f32(float *x, float *y, long n);

/* y[i] = x[i] * sigmoid(x[i])  (SiLU / Swish) */
void silu_f32(float *x, float *y, long n);

/* y[i] = logf(1 + expf(x[i])) */
void softplus_f32(float *x, float *y, long n);

#endif /* OPTIMATRIX_BRIDGE_H */
