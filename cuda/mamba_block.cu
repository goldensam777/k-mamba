/*
 * mamba_block.cu — Forward + backward GPU pour un bloc Mamba
 *
 * Pipeline forward :
 *   x [L, dim]
 *     -> W_in^T GEMM  -> u_raw [L, state]  -> SiLU      -> u [L, state]
 *     -> delta_proj   -> dt_raw [L]         -> softplus  -> dt [L]
 *     -> broadcast    -> B_exp, C_exp, dt_exp [L, state]
 *     -> scan1d (Blelloch GPU)              -> h_store [L, state], y_scan [L, state]
 *     -> W_out^T GEMM -> y_proj [L, dim]
 *     -> residual     -> y = y_proj + x
 *
 * Toutes les opérations sont sur GPU.
 * cuBLAS pour les GEMM, kernels custom pour elementwise + reductions.
 *
 * Convention GEMM row-major via cuBLAS (col-major interne) :
 *   C[M,N] = A[M,K] @ B[K,N]  :  cublasSgemm(h,N,N, N,M,K, a, B,N, A,K, b, C,N)
 *   C[M,N] = A[M,K] @ B^T     :  cublasSgemm(h,T,N, N,M,K, a, B,K, A,K, b, C,N)  (B=[N,K])
 *   C[M,N] = A^T   @ B[K,N]   :  cublasSgemm(h,N,T, N,M,K, a, B,N, A,M, b, C,N)  (A=[K,M])
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "scan.h"
#include "mamba_scan_cuda.h"

/* ── Macro de vérification ────────────────────────────────────── */
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d — %d\n", __FILE__, __LINE__, _s); \
        exit(1); \
    } \
} while(0)

/* ── Helpers GEMM row-major ───────────────────────────────────── */

/* C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N] */
static void gemm(cublasHandle_t h, int M, int N, int K,
                 float alpha, const float *A, const float *B,
                 float beta,  float *C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

/* C[M,N] = alpha * A[M,K] @ B^T + beta * C  (B est [N,K]) */
static void gemm_bt(cublasHandle_t h, int M, int N, int K,
                    float alpha, const float *A, const float *B,
                    float beta,  float *C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K, &alpha, B, K, A, K, &beta, C, N));
}

/* C[M,N] = alpha * A^T @ B + beta * C  (A est [K,M], B est [K,N]) */
static void gemm_at(cublasHandle_t h, int M, int N, int K,
                    float alpha, const float *A, const float *B,
                    float beta,  float *C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                             N, M, K, &alpha, B, N, A, M, &beta, C, N));
}

/* ── Kernels elementwise ──────────────────────────────────────── */

__global__ void silu_fwd_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = v / (1.0f + expf(-v));
}

/* dy_dx = silu'(x_raw) = sigmoid(x) * (1 + x*(1-sigmoid(x)))
 * dx = du * dy_dx  */
__global__ void silu_bwd_kernel(const float *du, const float *x_raw,
                                float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v   = x_raw[i];
    float sig = 1.0f / (1.0f + expf(-v));
    dx[i] = du[i] * sig * (1.0f + v * (1.0f - sig));
}

/* softplus avec clamp : dt = clamp(log(1+exp(x)), dt_min, dt_max) */
#define DT_MIN 1e-3f
#define DT_MAX 0.1f

__global__ void softplus_clamp_fwd_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float sp = (v > 20.0f) ? v : log1pf(expf(v));
    y[i] = fmaxf(DT_MIN, fminf(DT_MAX, sp));
}

/* Backward du softplus clampé : ddt_raw = ddt * sigmoid(x) si dans [min,max] */
__global__ void softplus_clamp_bwd_kernel(const float *ddt, const float *x_raw,
                                          const float *dt, float *ddt_raw, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float clamped = dt[i];
    if (clamped <= DT_MIN || clamped >= DT_MAX) {
        ddt_raw[i] = 0.0f;
    } else {
        float sig = 1.0f / (1.0f + expf(-x_raw[i]));
        ddt_raw[i] = ddt[i] * sig;
    }
}

/* Broadcast vec [D] -> out [L, D] : out[t, d] = vec[d] */
__global__ void broadcast_d_to_ld(const float *vec, float *out, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    out[idx] = vec[idx % D];
}

/* Broadcast scalar_per_pos [L] -> out [L, D] : out[t, d] = scalar[t] */
__global__ void broadcast_l_to_ld(const float *scalar, float *out, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    out[idx] = scalar[idx / D];
}

/* Réduction [L, D] -> [D] : out[d] = sum_t in[t, d] (accumule avec +=) */
__global__ void reduce_sum_L(const float *in, float *out, int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.0f;
    for (int t = 0; t < L; t++) acc += in[t * D + d];
    out[d] += acc;  /* += pour accumuler sur le batch */
}

/* Réduction [L, D] -> [L] : out[t] = sum_d in[t, d] */
__global__ void reduce_sum_D(const float *in, float *out, int L, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    float acc = 0.0f;
    for (int d = 0; d < D; d++) acc += in[t * D + d];
    out[t] = acc;   /* écrit (utilisé comme temporaire) */
}

/* y += x (accumulation résiduelle) */
__global__ void add_inplace_kernel(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += x[i];
}

/* dx[L, dim] += ddt[L] outer delta_proj[dim] : dx[t,d] += ddt[t]*dproj[d] */
__global__ void outer_add_kernel(float *dx, const float *ddt,
                                 const float *dproj, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    int t = idx / D;
    int d = idx % D;
    dx[idx] += ddt[t] * dproj[d];
}

/* ── Forward d'un bloc ────────────────────────────────────────── */
/*
 * Tous les pointeurs sont des device pointers (VRAM).
 * Les buffers workspace (u_raw, u, dt_raw, dt, B_exp, C_exp, dt_exp,
 * h_store, y_scan, y_proj) sont pré-alloués par l'appelant.
 */
/* ── Complex SSM sequential kernel (forward) ─────────────────── */
/*
 * Single-threaded sequential scan with R(θ) rotation.
 * Replaces the Blelloch parallel scan for correctness with rotation.
 * h_store[t*D + d] = h_t[d] (state at each timestep)
 * y_scan [t*D + d] = C_t[d] * h_t[d]
 */
__global__ void complex_ssm_fwd_kernel(
    const float *u,       /* [L, D] input (post-SiLU) */
    const float *A_log,   /* [D] */
    const float *B_exp,   /* [L, D] data-dep B */
    const float *C_exp,   /* [L, D] data-dep C */
    const float *dt,      /* [L] per-timestep delta */
    const float *theta,   /* [D/2] rotation angles */
    float *h_store,       /* [L, D] state at each step */
    float *y_scan,        /* [L, D] output */
    int L, int D)
{
    /* Single-threaded execution */
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    /* h_cur is h_store at t=0 (initialized to zero by caller) */
    float *h_cur = h_store;  /* points to h_store[0] */

    /* Temporary buffer for rotated h — we use h_store[t+1] as scratch during step t
     * Actually we compute in-place: rotate into a local array, then update */
    /* Allocate scratch on stack: D ≤ 512 in typical configs, max 2048 OK for stack */
    /* For safety, use a fixed-size stack buffer (state_size <= 1024 assumed) */
    float h_rot_local[1024];

    /* Initialize h_cur to zero */
    for (int d = 0; d < D; d++) h_cur[d] = 0.0f;

    for (int t = 0; t < L; t++) {
        float dt_t = dt[t];

        /* Apply R(θ) to h_cur */
        for (int i = 0; i + 1 < D; i += 2) {
            float th = theta ? theta[i >> 1] : 0.0f;
            float c = cosf(th), s = sinf(th);
            float h0 = h_cur[i], h1 = h_cur[i+1];
            h_rot_local[i]   = c*h0 - s*h1;
            h_rot_local[i+1] = s*h0 + c*h1;
        }
        if (D & 1) h_rot_local[D-1] = h_cur[D-1];

        /* h_t = exp(dt*A)*h_rot + dt*B_t*u_t */
        float *h_next = (t + 1 < L) ? &h_store[(t+1)*D] : h_cur;
        /* write h_t into h_store[t] */
        float *h_out = &h_store[t * D];
        for (int d = 0; d < D; d++) {
            float a = A_log[d];
            if (a > -1e-5f) a = -1e-5f;
            h_out[d] = expf(dt_t * a) * h_rot_local[d]
                       + dt_t * B_exp[t*D+d] * u[t*D+d];
        }
        /* y_scan[t] = C_t * h_t */
        for (int d = 0; d < D; d++)
            y_scan[t*D+d] = C_exp[t*D+d] * h_out[d];

        /* Advance h_cur pointer */
        h_cur = h_out;
        (void)h_next;
    }
}

extern "C" void gpu_block_forward(
    cublasHandle_t cublas,
    /* Paramètres du bloc [VRAM] */
    const float *d_W_in,        /* [state, dim]  */
    const float *d_W_out,       /* [dim,  state] */
    const float *d_A_log,       /* [state]       */
    const float *d_W_B,         /* [state, dim]  data-dependent B projection */
    const float *d_W_C,         /* [state, dim]  data-dependent C projection */
    const float *d_delta_proj,  /* [dim]         */
    const float *d_theta,       /* [state/2]     rotation angles (may be NULL) */
    /* Entrée / sortie */
    const float *d_x,           /* [L, dim]  input  */
    float       *d_y,           /* [L, dim]  output */
    /* Workspace (pré-alloué [L, state] sauf dt_raw/dt [L]) */
    float *d_u_raw,   float *d_u,
    float *d_dt_raw,  float *d_dt,
    float *d_B_exp,   float *d_C_exp, float *d_dt_exp,
    float *d_h_store, float *d_y_scan, float *d_y_proj,
    int L, int state, int dim)
{
    const float a1 = 1.0f, b0 = 0.0f;
    int blk;

    /* 1. in_proj : u_raw [L, state] = x [L, dim] @ W_in^T  (W_in=[state,dim]) */
    gemm_bt(cublas, L, state, dim, a1, d_x, d_W_in, b0, d_u_raw);

    /* 2. SiLU */
    blk = (L * state + 255) / 256;
    silu_fwd_kernel<<<blk, 256>>>(d_u_raw, d_u, L * state);

    /* 3. delta : dt_raw [L] = x [L, dim] @ delta_proj [dim]
     *    Vu comme GEMM : dt_raw [L,1] = x [L,dim] @ delta_proj [dim,1] */
    gemm(cublas, L, 1, dim, a1, d_x, d_delta_proj, b0, d_dt_raw);
    blk = (L + 255) / 256;
    softplus_clamp_fwd_kernel<<<blk, 256>>>(d_dt_raw, d_dt, L);

    /* 4. Data-dependent B, C: B_exp [L,state] = x [L,dim] @ W_B^T [state,dim]
     *                          C_exp [L,state] = x [L,dim] @ W_C^T [state,dim] */
    gemm_bt(cublas, L, state, dim, a1, d_x, d_W_B, b0, d_B_exp);
    gemm_bt(cublas, L, state, dim, a1, d_x, d_W_C, b0, d_C_exp);

    /* 5. Complex SSM sequential scan with R(θ) rotation */
    CUDA_CHECK(cudaMemset(d_h_store, 0, L * state * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_y_scan,  0, L * state * sizeof(float)));
    complex_ssm_fwd_kernel<<<1, 1>>>(
        d_u, d_A_log, d_B_exp, d_C_exp, d_dt,
        d_theta, d_h_store, d_y_scan, L, state);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 6. out_proj : y_proj [L, dim] = y_scan [L, state] @ W_out^T  (W_out=[dim,state]) */
    gemm_bt(cublas, L, dim, state, a1, d_y_scan, d_W_out, b0, d_y_proj);

    /* 7. Résiduel : y = y_proj + x */
    CUDA_CHECK(cudaMemcpy(d_y, d_y_proj, L * dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    blk = (L * dim + 255) / 256;
    add_inplace_kernel<<<blk, 256>>>(d_y, d_x, L * dim);
    /* Note: d_dt_exp is no longer used (kept for API compatibility) */
    (void)d_dt_exp;
}

/* ── Backward d'un bloc ───────────────────────────────────────── */
/*
 * Accumule les gradients des paramètres (+=).
 * dx est écrit (pas accumulé — l'appelant doit additionner si besoin).
 *
 * Buffers temporaires (pré-alloués par l'appelant) :
 *   d_dy_scan [L, state], d_du [L, state], d_du_raw [L, state]
 *   d_ddt [L], d_ddt_raw [L]
 *   d_dB_scan [L, state], d_dC_scan [L, state], d_ddt_scan [L, state]
 *   d_dA_tmp [state]
 */
/* ── Complex SSM sequential backward kernel ─────────────────── */
/*
 * Sequential backward through the complex SSM scan.
 * Computes:
 *   d_du     [L, D]: gradient w.r.t. u (input to SSM)
 *   d_dA_acc [D]:    gradient w.r.t. A_log (accumulated)
 *   d_dB_out [L, D]: gradient w.r.t. B_exp
 *   d_dC_out [L, D]: gradient w.r.t. C_exp
 *   d_ddt_out[L, D]: gradient w.r.t. dt (per-dim, reduce later)
 *   d_g_theta[D/2]:  gradient w.r.t. theta (accumulated)
 *   d_ddy_adj[L, D]: adjoint of y_scan passed down (= d_du_scan)
 */
__global__ void complex_ssm_bwd_kernel(
    const float *d_dy_scan,  /* [L, D] upstream gradient of y_scan */
    const float *u,          /* [L, D] */
    const float *A_log,      /* [D] */
    const float *B_exp,      /* [L, D] */
    const float *C_exp,      /* [L, D] */
    const float *dt,         /* [L] */
    const float *theta,      /* [D/2] */
    const float *h_store,    /* [L, D] state at each step */
    float *d_du,             /* [L, D] out */
    float *d_dA_acc,         /* [D]    out (accumulated) */
    float *d_dB_out,         /* [L, D] out */
    float *d_dC_out,         /* [L, D] out */
    float *d_ddt_out,        /* [L, D] out */
    float *d_g_theta,        /* [D/2]  out (accumulated) */
    int L, int D)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float adj_h[1024];
    for (int d = 0; d < D; d++) adj_h[d] = 0.0f;

    for (int t = L - 1; t >= 0; t--) {
        float dt_t = dt[t];
        const float *h_t    = &h_store[t * D];
        const float *h_prev = (t > 0) ? &h_store[(t-1)*D] : NULL;

        for (int d = 0; d < D; d++) {
            float ct_d  = C_exp[t*D+d];
            float bt_d  = B_exp[t*D+d];
            float ut_d  = u[t*D+d];
            float a_val = A_log[d]; if (a_val > -1e-5f) a_val = -1e-5f;
            float a_diag = expf(dt_t * a_val);
            float dy_s  = d_dy_scan[t*D+d];

            /* adj of h_t: d_dy_scan * C + adj_h */
            float ah = adj_h[d] + dy_s * ct_d;

            /* dC */
            d_dC_out[t*D+d] = dy_s * h_t[d];

            /* dB */
            d_dB_out[t*D+d] = ah * dt_t * ut_d;

            /* du */
            d_du[t*D+d] = ah * dt_t * bt_d;

            /* dA_log: use h_rot = h_store[t] - bt*ut contribution */
            /* h_rot[d] = (h_t[d] - dt*B*u) / a_diag, but simpler: */
            /* h_t[d] = a_diag * h_rot[d] + dt*B*u, so h_rot[d] = (h_t[d] - dt*B*u) / a_diag */
            float h_rot_d = (a_diag > 1e-30f)
                            ? (h_t[d] - dt_t * bt_d * ut_d) / a_diag
                            : 0.0f;
            d_dA_acc[d] += ah * dt_t * a_diag * h_rot_d;

            /* ddt (stored per-dim, reduce to scalar later) */
            d_ddt_out[t*D+d] = ah * (a_val * a_diag * h_rot_d + bt_d * ut_d);

            /* Store d_h_rot[d] = ah * a_diag  (for theta grad below) */
            adj_h[d] = ah * a_diag;  /* temporarily hold d_h_rot */
        }

        /* Gradient for theta and propagate adj_h = R^T * d_h_rot */
        for (int i = 0; i + 1 < D; i += 2) {
            float hp0 = h_prev ? h_prev[i]   : 0.0f;
            float hp1 = h_prev ? h_prev[i+1] : 0.0f;
            float th  = theta ? theta[i >> 1] : 0.0f;
            float c   = cosf(th), sv = sinf(th);
            float dr0 = adj_h[i], dr1 = adj_h[i+1];

            /* dtheta */
            if (theta && d_g_theta)
                d_g_theta[i >> 1] += dr0 * (-sv * hp0 - c * hp1)
                                   + dr1 * (c * hp0 - sv * hp1);

            /* adj_h = R^T * d_h_rot */
            adj_h[i]   = c * dr0 + sv * dr1;
            adj_h[i+1] = -sv * dr0 + c * dr1;
        }
        if (D & 1) { /* adj_h[D-1] stays as is */ }
    }
}

extern "C" void gpu_block_backward(
    cublasHandle_t cublas,
    /* Paramètres (lecture seule) */
    const float *d_W_in, const float *d_W_out,
    const float *d_A_log,
    const float *d_W_B, const float *d_W_C,
    const float *d_delta_proj,
    const float *d_theta,       /* [state/2] rotation angles */
    /* Activations sauvées au forward */
    const float *d_x,
    const float *d_u_raw, const float *d_u,
    const float *d_dt_raw, const float *d_dt,
    const float *d_B_exp, const float *d_C_exp, const float *d_dt_exp,
    const float *d_h_store, const float *d_y_scan,
    /* Gradient entrant */
    const float *d_dy,          /* [L, dim] upstream gradient */
    /* Gradients des paramètres (accumulés, +=) */
    float *d_dW_in, float *d_dW_out,
    float *d_dA_log,
    float *d_dW_B, float *d_dW_C,
    float *d_ddelta_proj,
    float *d_g_theta,           /* [state/2] grad for theta */
    /* Gradient de sortie */
    float *d_dx,                /* [L, dim] downstream gradient */
    /* Workspace temporaire */
    float *d_dy_scan,           /* [L, state] */
    float *d_du,                /* [L, state] */
    float *d_du_raw,            /* [L, state] */
    float *d_ddt,               /* [L] */
    float *d_ddt_raw,           /* [L] */
    float *d_dB_scan,           /* [L, state] scan gradient de B */
    float *d_dC_scan,           /* [L, state] scan gradient de C */
    float *d_ddt_scan,          /* [L, state] scan gradient de dt */
    float *d_dA_tmp,            /* [state]   scan gradient de A (tmp) */
    int L, int state, int dim)
{
    const float a1 = 1.0f, b0 = 0.0f;
    int blk;

    /* ── Résiduel : dy passe aussi vers dx (on initialise dx = dy) ── */
    CUDA_CHECK(cudaMemcpy(d_dx, d_dy, L * dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    /* ── Backward out_proj ──────────────────────────────────────── */
    /* dW_out [dim, state] += dy^T @ y_scan  (A=[L,dim], B=[L,state]) */
    gemm_at(cublas, dim, state, L, a1, d_dy, d_y_scan, a1, d_dW_out);

    /* dy_scan [L, state] = dy [L, dim] @ W_out [dim, state] */
    gemm(cublas, L, state, dim, a1, d_dy, d_W_out, b0, d_dy_scan);

    /* ── Backward complex SSM (sequential) ──────────────────────── */
    /* Zero accumulators */
    CUDA_CHECK(cudaMemset(d_dA_tmp, 0, state * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ddt_scan, 0, L * state * sizeof(float)));

    complex_ssm_bwd_kernel<<<1, 1>>>(
        d_dy_scan,
        d_u, d_A_log, d_B_exp, d_C_exp, d_dt, d_theta, d_h_store,
        d_du, d_dA_tmp, d_dB_scan, d_dC_scan, d_ddt_scan, d_g_theta,
        L, state);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Accumule dA_log */
    blk = (state + 255) / 256;
    add_inplace_kernel<<<blk, 256>>>(d_dA_log, d_dA_tmp, state);

    /* g_W_B [state,dim] += dB_scan^T [state,L] @ x [L,dim] */
    gemm_at(cublas, state, dim, L, a1, d_dB_scan, d_x, a1, d_dW_B);

    /* g_W_C [state,dim] += dC_scan^T [state,L] @ x [L,dim] */
    gemm_at(cublas, state, dim, L, a1, d_dC_scan, d_x, a1, d_dW_C);

    /* ddt [L] = sum_d ddt_scan [t, d] */
    blk = (L + 255) / 256;
    reduce_sum_D<<<blk, 256>>>(d_ddt_scan, d_ddt, L, state);

    /* ── Backward softplus ──────────────────────────────────────── */
    softplus_clamp_bwd_kernel<<<blk, 256>>>(d_ddt, d_dt_raw, d_dt, d_ddt_raw, L);

    /* ── Backward delta_proj ────────────────────────────────────── */
    /* ddelta_proj [dim] += ddt_raw^T @ x */
    gemm_at(cublas, 1, dim, L, a1, d_ddt_raw, d_x, a1, d_ddelta_proj);

    /* dx [L, dim] += ddt_raw [L] outer delta_proj [dim] */
    blk = (L * dim + 255) / 256;
    outer_add_kernel<<<blk, 256>>>(d_dx, d_ddt_raw, d_delta_proj, L, dim);

    /* ── Backward SiLU ──────────────────────────────────────────── */
    blk = (L * state + 255) / 256;
    silu_bwd_kernel<<<blk, 256>>>(d_du, d_u_raw, d_du_raw, L * state);

    /* ── Backward in_proj ───────────────────────────────────────── */
    gemm_at(cublas, state, dim, L, a1, d_du_raw, d_x, a1, d_dW_in);
    gemm(cublas, L, dim, state, a1, d_du_raw, d_W_in, a1, d_dx);

    /* dt_exp no longer used (kept for API compat) */
    (void)d_dt_exp; (void)d_ddt_raw;
}
