/*
 * scan1d.cu — Selective Scan 1D sur GPU
 *
 * Recurrence SSM :
 *   h_t[d,m] = exp(dt_t[d] * A[d,m]) * h_{t-1}[d,m]
 *            + dt_t[d] * B_t[d,m] * x_t[d]
 *   y_t[d]   = sum_m C_t[d,m] * h_t[d,m]
 *
 * Layout memoire :
 *   x  : [L * D]
 *   A  : [D * M]           (partage entre positions)
 *   B  : [L * D * M]       (selectif)
 *   C  : [L * D * M]       (selectif)
 *   dt : [L * D]           (selectif)
 *   h  : [L * D * M]       (etats caches, stockes pour backward)
 *   y  : [L * D]
 *
 * ═══════════════════════════════════════════════════════════════
 * Deux implementations :
 *
 * A) Sequential : 1 thread = (d, m), boucle sur L
 *    Meme logique que le CPU. Parallelisme sur D*M.
 *    Simple, correct, mais sequentiel en L.
 *
 * B) Parallel prefix scan (Blelloch, 1990)
 *    Exploite l'associativite de la recurrence SSM :
 *
 *    (a1, b1) ⊗ (a2, b2) = (a1*a2, a2*b1 + b2)
 *
 *    Ou (at, bt) = (exp(dt*A), dt*B*x).
 *    En appliquant le prefix scan de gauche, on obtient h_t
 *    pour tout t en O(log L) etapes au lieu de O(L).
 *
 *    Complexite :
 *      CPU sequentiel : O(L)  depth, O(L) work
 *      Blelloch GPU   : O(log L) depth, O(L) work
 *      Gain : jusqu'a log2(L) = 10 pour L=1024
 *
 *    Ceci est IMPOSSIBLE sur CPU avec AVX2 :
 *      - La reduction d'arbres necessitant des echanges
 *        de donnees entre "lanes" SIMD sans cout est non-triviale
 *      - Sur GPU : warp shuffle + shared memory
 * ═══════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include "scan.h"
#include <math.h>

/* ── IMPLEMENTATION A : Sequential ──────────────────────────────── */
/*
 * Chaque thread prend en charge un (d, m) et boucle sur L.
 * GPU advantage : D*M threads en parallele vs 1 thread CPU.
 * Pour D=128, M=16 : 2048 threads en parallele.
 */

__global__ void scan1d_seq_kernel(
    const float *x,   const float *A,
    const float *B,   const float *C,
    const float *dt,
    float *y,         float *h,
    int L, int D, int M)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= D * M) return;

    int m = tid % M;
    int d = tid / M;
    int dm = d * M + m;

    float h_prev = 0.0f;

    for (int t = 0; t < L; t++) {
        int t_d  = t * D + d;
        int t_dm = t * D * M + dm;

        float dt_val = dt[t_d];
        float a = expf(dt_val * A[dm]);
        float b = dt_val * B[t_dm] * x[t_d];

        float h_cur = a * h_prev + b;
        h[t_dm]  = h_cur;
        h_prev   = h_cur;

        /* Contribution a y : atomicAdd car plusieurs m ecrivent y[t,d] */
        atomicAdd(&y[t_d], C[t_dm] * h_cur);
    }
}

/* ── IMPLEMENTATION B : Parallel prefix scan (Blelloch) ──────────── */
/*
 * Etape 1 — Precomputation : calculer (a_t, b_t) pour tout t
 *   a_t = exp(dt_t * A[d,m])
 *   b_t = dt_t * B_t[d,m] * x_t[d]
 *   -> kernel parallelise sur (t, d, m)
 *
 * Etape 2 — Prefix scan en shared memory
 *   Operator : (a1, b1) ⊗ (a2, b2) = (a1*a2, a2*b1 + b2)
 *   Associatif -> prefix scan de Blelloch applicable
 *
 *   Pour L <= blockDim.x (L = 1024, 1 bloc par (d,m)) :
 *     Tout tient en shared memory.
 *
 * Etape 3 — Calcul de y
 *   y[t, d] = sum_m C[t,d,m] * h[t,d,m]
 *   -> atomicAdd ou kernel de reduction separe
 */

#define MAX_L 1024  /* L max pour le prefix scan en shared mem */

__global__ void scan1d_precompute_kernel(
    const float *x,  const float *A,
    const float *B,  const float *dt,
    float *d_a,      float *d_b,    /* buffers intermediaires [L*D*M] */
    int L, int D, int M)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L * D * M) return;

    int m = tid % M;
    int d = (tid / M) % D;
    int t = tid / (D * M);

    int dm   = d * M + m;
    int t_d  = t * D + d;
    int t_dm = t * D * M + dm;

    float dt_val = dt[t_d];
    d_a[t_dm] = expf(dt_val * A[dm]);
    d_b[t_dm] = dt_val * B[t_dm] * x[t_d];
}

/*
 * Kernel de prefix scan par bloc.
 * 1 bloc = 1 paire (d, m).
 * blockDim.x = L (L <= MAX_L = 1024).
 *
 * Algorithme de Blelloch (inclusive prefix scan) :
 *
 *   Phase up-sweep (reduction) :
 *     for stride = 1, 2, 4, ..., L/2 :
 *       thread i (ou i % (2*stride) == 2*stride-1) :
 *         s[i] = s[i] ⊗ s[i - stride]
 *
 *   Phase down-sweep (distribution) :
 *     s[L-1] = identite (1, 0)
 *     for stride = L/2, L/4, ..., 1 :
 *       thread i (ou i % (2*stride) == 2*stride-1) :
 *         tmp = s[i - stride]
 *         s[i - stride] = s[i]
 *         s[i] = s[i] ⊗ tmp   (application de l'operateur)
 *
 * h[t] est le resultat du scan inclusif a la position t.
 */

__global__ void scan1d_blelloch_kernel(
    const float *d_a, const float *d_b,   /* [L * D * M] */
    float *h,                              /* [L * D * M] */
    int L, int D, int M)
{
    /* 1 bloc = 1 paire (d, m) */
    int dm_idx = blockIdx.x;              /* d * M + m */
    int d = dm_idx / M;
    int m = dm_idx % M;
    int t = threadIdx.x;                  /* position dans la sequence */

    if (t >= L || d >= D || m >= M) return;

    /* Shared memory : paires (a, b) de longueur L */
    extern __shared__ float smem[];       /* 2 * L floats */
    float *sa = smem;                     /* composante a */
    float *sb = smem + L;                 /* composante b */

    /* Charger depuis global memory */
    int idx = t * D * M + d * M + m;
    sa[t] = d_a[idx];
    sb[t] = d_b[idx];
    __syncthreads();

    /* ── Up-sweep (reduction) ──────────────────────────────────────── */
    for (int stride = 1; stride < L; stride <<= 1) {
        int i = (t + 1) * (stride << 1) - 1;
        if (i < L) {
            int j = i - stride;
            /* (sa[i], sb[i]) = (sa[j], sb[j]) ⊗ (sa[i], sb[i])      */
            /* operateur : (a1,b1) ⊗ (a2,b2) = (a1*a2, a2*b1 + b2)   */
            float new_a = sa[j] * sa[i];
            float new_b = sa[i] * sb[j] + sb[i];
            sa[i] = new_a;
            sb[i] = new_b;
        }
        __syncthreads();
    }

    /* ── Down-sweep ─────────────────────────────────────────────────── */
    if (t == 0) {
        sa[L - 1] = 1.0f;   /* identite pour a : 1 */
        sb[L - 1] = 0.0f;   /* identite pour b : 0 */
    }
    __syncthreads();

    for (int stride = L >> 1; stride > 0; stride >>= 1) {
        int i = (t + 1) * (stride << 1) - 1;
        if (i < L) {
            int j = i - stride;

            float tmp_a  = sa[j];   /* sauvegarde enfant gauche */
            float tmp_b  = sb[j];

            float old_ai = sa[i];   /* sauvegarde enfant droit AVANT ecrasement */
            float old_bi = sb[i];

            /* Echange : l'enfant gauche recoit la valeur de l'enfant droit */
            sa[j] = old_ai;
            sb[j] = old_bi;

            /* L'enfant droit = (old_ai, old_bi) ⊗ (tmp_a, tmp_b)          */
            /* operateur : (a1,b1) ⊗ (a2,b2) = (a1*a2, a2*b1 + b2)        */
            sa[i] = old_ai * tmp_a;
            sb[i] = tmp_a * old_bi + tmp_b;
        }
        __syncthreads();
    }

    /* Stocker h : scan exclusif -> convertir en inclusif */
    /* inclusif[t] = (a_t, b_t) ⊗ exclusif[t] */
    float a_t = d_a[idx];
    float b_t = d_b[idx];
    float h_val = a_t * sb[t] + b_t;   /* = a_t * h_prefix + b_t */
    /* Note : sb[t] apres down-sweep = h_{t-1} (h precedent) */
    h[idx] = h_val;
    __syncthreads();
}

/* Calcul de y depuis h et C */
__global__ void scan1d_output_kernel(
    const float *h, const float *C,
    float *y,
    int L, int D, int M)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L * D * M) return;

    int m  = tid % M;
    int d  = (tid / M) % D;
    int t  = tid / (D * M);

    int t_d  = t * D + d;
    int t_dm = t * D * M + d * M + m;

    atomicAdd(&y[t_d], C[t_dm] * h[t_dm]);
}

/* ── API ─────────────────────────────────────────────────────────── */

void om_scan1d_forward(
    const float *d_x,  const float *d_A,
    const float *d_B,  const float *d_C,
    const float *d_dt,
    float *d_y, float *d_h,
    int L, int D, int M)
{
    /* Zeroiser y (accumulation par atomicAdd) */
    cudaMemset(d_y, 0, L * D * sizeof(float));

    if (L <= MAX_L) {
        /*
         * Implementation B — Parallel prefix scan (Blelloch)
         * 1 bloc par (d, m), L threads par bloc
         * Shared memory : 2 * L * sizeof(float)
         */

        /* Etape 1 : precomputer (a_t, b_t) */
        float *d_a, *d_b;
        OM_CHECK(cudaMalloc(&d_a, L * D * M * sizeof(float)));
        OM_CHECK(cudaMalloc(&d_b, L * D * M * sizeof(float)));

        int total = L * D * M;
        int blocks_pre = (total + 255) / 256;
        scan1d_precompute_kernel<<<blocks_pre, 256>>>(
            d_x, d_A, d_B, d_dt, d_a, d_b, L, D, M);

        /* Etape 2 : prefix scan de Blelloch par (d, m) */
        int num_dm   = D * M;
        size_t smem  = 2 * L * sizeof(float);
        scan1d_blelloch_kernel<<<num_dm, L, smem>>>(
            d_a, d_b, d_h, L, D, M);

        /* Etape 3 : y = sum_m C * h */
        scan1d_output_kernel<<<blocks_pre, 256>>>(
            d_h, d_C, d_y, L, D, M);

        cudaFree(d_a);
        cudaFree(d_b);
    } else {
        /*
         * Implementation A — Sequential pour les grandes sequences
         * Fallback si L > 1024 (depasse la shared memory d'un bloc)
         * TODO : prefix scan par chunks pour les grandes L
         */
        int num_dm = D * M;
        int blocks = (num_dm + 255) / 256;
        scan1d_seq_kernel<<<blocks, 256>>>(
            d_x, d_A, d_B, d_C, d_dt, d_y, d_h, L, D, M);
    }

    cudaDeviceSynchronize();
}
