/*
 * scan1d_backward.cu — Scan adjoint (backward du selective scan 1D)
 *
 * Forward :
 *   h_t[d,m] = a_t[d,m] * h_{t-1}[d,m] + b_t[d,m]
 *   a_t      = exp(dt_t[d] * A[d,m])
 *   b_t      = dt_t[d] * B_t[d,m] * x_t[d]
 *   y_t[d]   = sum_m  C_t[d,m] * h_t[d,m]
 *
 * ═══════════════════════════════════════════════════════════════
 * Scan adjoint (derivation complete) :
 *
 * Soit g_t[d,m] = dL/dh_t[d,m].
 *
 * Pour t = L-1 .. 0 (reverse) :
 *
 *   g_t = C_t * dL/dy_t  +  a_{t+1} * g_{t+1}
 *         ^--- contribution directe   ^--- propagation du futur
 *         (g_L = 0 par convention)
 *
 * De g_t on tire les gradients des parametres :
 *
 *   dL/dC_t[d,m]  = dL/dy_t[d] * h_t[d,m]
 *
 *   dL/dB_t[d,m]  = g_t[d,m] * dt_t[d] * x_t[d]
 *
 *   dL/dx_t[d]   += sum_m g_t[d,m] * dt_t[d] * B_t[d,m]
 *
 *   dL/dA[d,m]   += g_t[d,m] * h_{t-1}[d,m] * dt_t[d] * a_t[d,m]
 *                   (derivee de exp(dt*A) par rapport a A)
 *
 *   dL/ddt_t[d]  += sum_m g_t[d,m] * (h_{t-1}[d,m] * A[d,m] * a_t[d,m]
 *                                     + B_t[d,m] * x_t[d])
 *
 * ═══════════════════════════════════════════════════════════════
 * Implementation : 1 thread par (d, m), boucle t = L-1 .. 0.
 * Symetrique au kernel scan1d_seq_kernel du forward.
 *
 * Accumulation :
 *   dx, ddt : sum sur m -> atomicAdd
 *   dA      : sum sur t -> atomicAdd
 *   dB, dC  : 1:1 par (t,d,m) -> ecriture directe
 */

#include <stdio.h>
#include "scan.h"
#include <math.h>

/* ── Kernel adjoint ──────────────────────────────────────────────── */

__global__ void scan1d_backward_kernel(
    const float *dy,    /* [L, D]       dL/dy                         */
    const float *x,     /* [L, D]       entree forward                */
    const float *A,     /* [D, M]       transition SSM                */
    const float *B,     /* [L, D, M]    projections selectifs         */
    const float *C,     /* [L, D, M]                                  */
    const float *dt,    /* [L, D]       delta apres softplus          */
    const float *h,     /* [L, D, M]    etats caches (sauvegardes)    */
    float *dx,          /* [L, D]       gradient / x  (zeroise avant) */
    float *dA,          /* [D, M]       gradient / A  (zeroise avant) */
    float *dB,          /* [L, D, M]    gradient / B                  */
    float *dC,          /* [L, D, M]    gradient / C                  */
    float *ddt,         /* [L, D]       gradient / dt (zeroise avant) */
    int L, int D, int M)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= D * M) return;

    int m  = tid % M;
    int d  = tid / M;
    int dm = d * M + m;

    float g_h = 0.0f;   /* g_t = dL/dh_t, initialise a 0 (g_L = 0) */

    for (int t = L - 1; t >= 0; t--) {
        int t_d  = t * D + d;
        int t_dm = t * D * M + dm;

        float dt_val = dt[t_d];
        float a_t    = expf(dt_val * A[dm]);

        /* h_{t-1} : h[t-1, d, m] si t > 0, sinon 0 */
        float h_prev = (t > 0) ? h[(t - 1) * D * M + dm] : 0.0f;

        /* g_t = C_t * dy_t  +  a_{t+1} * g_{t+1}
         * g_h contient deja a_{t+1} * g_{t+1} (iteration precedente).
         * On ajoute maintenant la contribution directe. */
        g_h += C[t_dm] * dy[t_d];

        /* dC[t, d, m] = dy[t, d] * h[t, d, m] */
        dC[t_dm] = dy[t_d] * h[t_dm];

        /* dB[t, d, m] = g_h * dt * x[t, d] */
        dB[t_dm] = g_h * dt_val * x[t_d];

        /* dx[t, d] += g_h * dt * B[t, d, m]   (sum sur m -> atomicAdd) */
        atomicAdd(&dx[t_d], g_h * dt_val * B[t_dm]);

        /* dA[d, m] += g_h * h_{t-1} * dt * a_t  (sum sur t -> atomicAdd) */
        atomicAdd(&dA[dm], g_h * h_prev * dt_val * a_t);

        /* ddt[t, d] += g_h * (h_{t-1} * A * a_t + B[t] * x[t]) */
        atomicAdd(&ddt[t_d],
            g_h * (h_prev * A[dm] * a_t + B[t_dm] * x[t_d]));

        /* Propagation : g_{t-1} recevra a_t * g_t lors du prochain tour */
        g_h = a_t * g_h;
    }
}

/* ── API ─────────────────────────────────────────────────────────── */

void om_scan1d_backward(
    const float *d_dy,
    const float *d_x,  const float *d_A,
    const float *d_B,  const float *d_C,
    const float *d_dt, const float *d_h,
    float *d_dx, float *d_dA,
    float *d_dB, float *d_dC, float *d_ddt,
    int L, int D, int M)
{
    /* Zeroise les buffers qui accumulent (atomicAdd) */
    cudaMemset(d_dx,  0, L * D     * sizeof(float));
    cudaMemset(d_dA,  0, D * M     * sizeof(float));
    cudaMemset(d_ddt, 0, L * D     * sizeof(float));
    /* dB et dC sont ecrits directement (pas besoin de zeroise) */

    int num_dm = D * M;
    int blocks  = (num_dm + 255) / 256;

    scan1d_backward_kernel<<<blocks, 256>>>(
        d_dy, d_x, d_A, d_B, d_C, d_dt, d_h,
        d_dx, d_dA, d_dB, d_dC, d_ddt,
        L, D, M);

    cudaDeviceSynchronize();
}
