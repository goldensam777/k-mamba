#ifndef KMAMBA_SCAN_H
#define KMAMBA_SCAN_H

#include <stddef.h>

/* ============================================================
 * scan.h — Types et fonctions pour les scans sélectifs Mamba
 *
 * Scan 1D (séquences) et Scan 2D (grilles wavefront).
 * Ces opérations sont spécifiques à la logique Mamba —
 * elles ne font pas partie d'optimatrix.
 * ============================================================ */

/* ── Macro CUDA pour vérification d'erreurs ─────────────────── */
#ifdef __CUDACC__
#include <stdio.h>
#include <stdlib.h>
#define OM_CHECK(call)                                                  \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_e));        \
            exit(1);                                                    \
        }                                                               \
    } while (0)
#endif

/* ── Selective Scan 1D — forward ────────────────────────────── */
/*
 * Layout mémoire :
 *   x     : [L, D]
 *   A     : [D, M]       (partagé sur L)
 *   B     : [L, D, M]    (sélectif)
 *   C     : [L, D, M]    (sélectif)
 *   delta : [L, D]
 *   h     : [L, D, M]    (états cachés, stockés pour backward)
 *   y     : [L, D]
 *
 * Récurrence :
 *   h_t[d,m] = exp(dt_t[d] * A[d,m]) * h_{t-1}[d,m]
 *            + dt_t[d] * B_t[d,m] * x_t[d]
 *   y_t[d]   = sum_m C_t[d,m] * h_t[d,m]
 */
typedef struct {
    float *x;
    float *A;
    float *B;
    float *C;
    float *delta;
    float *h;
    float *y;
    long   L;
    long   D;
    long   M;
} ScanParams;

void scan1d(ScanParams *p);

/* ── Selective Scan 1D — backward générique [L, D, M] ───────── */
typedef struct {
    float *x;
    float *A;
    float *B;
    float *C;
    float *delta;
    float *h0;
    float *h;
    float *dy;
    float *dx;
    float *dA;
    float *dB;
    float *dC;
    float *ddelta;
    long   L;
    long   D;
    long   M;
} ScanBackwardParams;

void scan1d_backward(ScanBackwardParams *p);

/* ── Selective Scan 1D — backward M=1 (B/C partagés) ────────── */
typedef struct {
    float *x;
    float *A;
    float *A_diag;   /* exp(dt*A) précompté, ou NULL */
    float *B;
    float *C;
    float *delta;
    float *h0;
    float *h;
    float *dy;
    float *dx;
    float *dA;
    float *dB;
    float *dC;
    float *ddelta;
    long   L;
    long   D;
} ScanBackwardSharedParams;

void scan1d_backward_m1_shared_bc(ScanBackwardSharedParams *p);
void scan1d_backward_m1_shared_bc_asm(ScanBackwardSharedParams *p);
void scan1d_backward_m1_shared_bc_simple_asm(ScanBackwardSharedParams *p);

/* ── Selective Scan 1D — backward générique sur M ───────────── */
typedef struct {
    float *x;
    float *A;
    float *B;
    float *C;
    float *delta;
    float *h0;
    float *h;
    float *dy;
    float *dx;
    float *dA;
    float *dB;
    float *dC;
    float *ddelta;
    long   L;
    long   D;
    long   M;
} ScanBackwardMParams;

void scan1d_backward_m_generic(ScanBackwardMParams *p);

/* ── Selective Scan 2D — wavefront ──────────────────────────── */
/*
 * Récurrence 2D sur grille (d1 x d2) :
 *   h(i,j,d,m) = dA1 * h(i-1,j,d,m)
 *              + dA2 * h(i,j-1,d,m)
 *              + dB  * x(i,j,d)
 *   y(i,j,d)   = sum_m C(i,j,d,m) * h(i,j,d,m)
 *
 * Ordonnancement wavefront : diagonale k = i+j.
 * Les positions sur la même diagonale sont indépendantes.
 */
typedef struct {
    float *x;
    float *A1;
    float *A2;
    float *B;
    float *C;
    float *delta1;
    float *delta2;
    float *h;
    float *y;
    long   d1;
    long   d2;
    long   D;
    long   M;
} Scan2DParams;

void scan2d(Scan2DParams *p);

/* ── CUDA API (scan 1D GPU) ─────────────────────────────────── */
#ifdef __CUDACC__
void om_scan1d_forward(
    const float *d_x,  const float *d_A,
    const float *d_B,  const float *d_C,
    const float *d_dt,
    float *d_y, float *d_h,
    int L, int D, int M
);

void om_scan1d_backward(
    const float *d_dy,
    const float *d_x,  const float *d_A,
    const float *d_B,  const float *d_C,
    const float *d_dt, const float *d_h,
    float *d_dx, float *d_dA,
    float *d_dB, float *d_dC, float *d_ddt,
    int L, int D, int M
);
#endif

#endif /* KMAMBA_SCAN_H */
