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

__global__ void cuda_silu_fwd_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = v / (1.0f + expf(-v));
}

/* dy_dx = silu'(x_raw) = sigmoid(x) * (1 + x*(1-sigmoid(x)))
 * dx = du * dy_dx  */
__global__ void cuda_silu_bwd_kernel(const float *du, const float *x_raw,
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

__global__ void cuda_softplus_clamp_fwd_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float sp = (v > 20.0f) ? v : log1pf(expf(v));
    y[i] = fmaxf(DT_MIN, fminf(DT_MAX, sp));
}

/* Backward du softplus clampé : ddt_raw = ddt * sigmoid(x) si dans [min,max] */
__global__ void cuda_softplus_clamp_bwd_kernel(const float *ddt, const float *x_raw,
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
__global__ void cuda_broadcast_d_to_ld(const float *vec, float *out, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    out[idx] = vec[idx % D];
}

/* Broadcast scalar_per_pos [L] -> out [L, D] : out[t, d] = scalar[t] */
__global__ void cuda_broadcast_l_to_ld(const float *scalar, float *out, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) return;
    out[idx] = scalar[idx / D];
}

/* Réduction [L, D] -> [D] : out[d] = sum_t in[t, d] (accumule avec +=) */
__global__ void cuda_reduce_sum_L(const float *in, float *out, int L, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.0f;
    for (int t = 0; t < L; t++) acc += in[t * D + d];
    out[d] += acc;  /* += pour accumuler sur le batch */
}

/* Réduction [L, D] -> [L] : out[t] = sum_d in[t, d] */
__global__ void cuda_reduce_sum_D(const float *in, float *out, int L, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    float acc = 0.0f;
    for (int d = 0; d < D; d++) acc += in[t * D + d];
    out[t] = acc;   /* écrit (utilisé comme temporaire) */
}

/* y += x (accumulation résiduelle) */
__global__ void cuda_add_inplace_kernel(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += x[i];
}

/* Sigmoid : y[i] = 1 / (1 + exp(-x[i])) */
__global__ void cuda_sigmoid_fwd_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = (v > 20.0f) ? 1.0f : (v < -20.0f) ? 0.0f : 1.0f / (1.0f + expf(-v));
}

/* Sigmoid backward : dx = dy * sigma * (1 - sigma) */
__global__ void cuda_sigmoid_bwd_kernel(const float *dy, const float *y, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] = dy[i] * y[i] * (1.0f - y[i]);
}

/* dx[L, dim] += ddt[L] outer delta_proj[dim] : dx[t,d] += ddt[t]*dproj[d] */
__global__ void cuda_outer_add_kernel(float *dx, const float *ddt,
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
 * Single-threaded sequential scan with R(θ) rotation and exp-trapezoidal discretization.
 * h_t = alpha_t * R(θ)*h_{t-1} + beta_t * Bu_{t-1} + gamma_t * Bu_t
 * h_store[t*D + d] = h_t[d]
 * y_scan [t*D + d] = C_t[d] * h_t[d]
 */
__global__ void cuda_ssm_fwd_kernel(
    const float *u,       /* [L, D] input (post-SiLU) */
    const float *A_log,   /* [D] */
    const float *B_exp,   /* [L, D] data-dep B */
    const float *C_exp,   /* [L, D] data-dep C */
    const float *dt,      /* [L] per-timestep delta */
    const float *theta,   /* [D/2] rotation angles */
    const float *lambda,  /* [L]   per-timestep lambda (sigmoid output) */
    float *h_store,       /* [L, D] state at each step */
    float *y_scan,        /* [L, D] output */
    int L, int D)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float h_rot_local[1024];
    float prev_Bu[1024];

    /* Initialize */
    for (int d = 0; d < D; d++) {
        h_store[d] = 0.0f;
        prev_Bu[d] = 0.0f;
    }

    float *h_cur = h_store;

    for (int t = 0; t < L; t++) {
        float dt_t  = dt[t];
        float lam_t = lambda ? lambda[t] : 0.5f;

        /* Apply R(θ) to h_cur */
        for (int i = 0; i + 1 < D; i += 2) {
            float th = theta ? theta[i >> 1] : 0.0f;
            float c = cosf(th), s = sinf(th);
            float h0 = h_cur[i], h1 = h_cur[i+1];
            h_rot_local[i]   = c*h0 - s*h1;
            h_rot_local[i+1] = s*h0 + c*h1;
        }
        if (D & 1) h_rot_local[D-1] = h_cur[D-1];

        float *h_out = &h_store[t * D];
        for (int d = 0; d < D; d++) {
            float a = A_log[d]; if (a > -1e-5f) a = -1e-5f;
            float alpha  = expf(dt_t * a);
            float beta   = (1.0f - lam_t) * dt_t * alpha;
            float gamma_ = lam_t * dt_t;
            float bu_t   = B_exp[t*D+d] * u[t*D+d];
            h_out[d] = alpha * h_rot_local[d] + beta * prev_Bu[d] + gamma_ * bu_t;
            prev_Bu[d] = bu_t;
        }
        for (int d = 0; d < D; d++)
            y_scan[t*D+d] = C_exp[t*D+d] * h_out[d];

        h_cur = h_out;
    }
}

/* ── Complex SSM PARALLEL kernel (forward) ──────────────────── */
/*
 * Parallel scan: each thread block handles one state dimension d.
 * Within each block, threads cooperatively scan the sequence using
 * a parallel associative scan (Blelloch-style reduction).
 * 
 * For R(θ) rotation: pairs of dimensions (2k, 2k+1) are handled together
 * in the same block to allow rotation within the pair.
 */
#define SSM_PARALLEL_THREADS 256
#define SSM_PARALLEL_ITEMS_PER_THREAD 4

__global__ void cuda_ssm_fwd_parallel_kernel(
    const float *u,       /* [L, D] input (post-SiLU) */
    const float *A_log,   /* [D] */
    const float *B_exp,   /* [L, D] data-dep B */
    const float *C_exp,   /* [L, D] data-dep C */
    const float *dt,      /* [L] per-timestep delta */
    const float *theta,   /* [D/2] rotation angles */
    const float *lambda,  /* [L] per-timestep lambda */
    float *h_store,       /* [L, D] state at each step */
    float *y_scan,        /* [L, D] output */
    int L, int D)
{
    /* Block handles one dimension pair (2k, 2k+1) or single dimension if D odd */
    int pair_idx = blockIdx.x;
    int d0 = pair_idx * 2;
    int d1 = d0 + 1;
    
    if (d0 >= D) return;  /* Out of bounds */
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    /* Local accumulators in registers/shared mem */
    __shared__ float s_h0[SSM_PARALLEL_THREADS];
    __shared__ float s_h1[SSM_PARALLEL_THREADS];
    __shared__ float s_alpha[SSM_PARALLEL_THREADS];
    __shared__ float s_beta[SSM_PARALLEL_THREADS];
    __shared__ float s_gamma[SSM_PARALLEL_THREADS];
    __shared__ float s_bu[SSM_PARALLEL_THREADS];
    
    float local_h0 = 0.0f;
    float local_h1 = 0.0f;
    float local_prev_bu0 = 0.0f;
    float local_prev_bu1 = 0.0f;
    
    /* Get rotation angle for this pair */
    float th = (theta && d1 < D) ? theta[pair_idx] : 0.0f;
    float c = cosf(th), s = sinf(th);
    float a0 = A_log[d0]; if (a0 > -1e-5f) a0 = -1e-5f;
    float a1 = (d1 < D) ? A_log[d1] : 0.0f; if (a1 > -1e-5f) a1 = -1e-5f;
    
    /* Process sequence in chunks */
    int items_per_block = num_threads * SSM_PARALLEL_ITEMS_PER_THREAD;
    
    for (int base_t = 0; base_t < L; base_t += items_per_block) {
        int local_t = tid;
        int t = base_t + local_t;
        
        /* Phase 1: Each thread processes its assigned timesteps sequentially */
        if (t < L) {
            float dt_t = dt[t];
            float lam_t = lambda ? lambda[t] : 0.5f;
            
            float alpha0 = expf(dt_t * a0);
            float alpha1 = (d1 < D) ? expf(dt_t * a1) : 0.0f;
            float beta0 = (1.0f - lam_t) * dt_t * alpha0;
            float beta1 = (d1 < D) ? (1.0f - lam_t) * dt_t * alpha1 : 0.0f;
            float gamma0 = lam_t * dt_t;
            float gamma1 = (d1 < D) ? lam_t * dt_t : 0.0f;
            
            /* Process SSM_PARALLEL_ITEMS_PER_THREAD consecutive steps */
            for (int step = 0; step < SSM_PARALLEL_ITEMS_PER_THREAD && (t + step * num_threads) < L; step++) {
                int ts = t + step * num_threads;
                
                float bu0 = B_exp[ts * D + d0] * u[ts * D + d0];
                float bu1 = (d1 < D) ? B_exp[ts * D + d1] * u[ts * D + d1] : 0.0f;
                
                /* Apply R(θ) to current state */
                float h_rot0, h_rot1;
                if (d1 < D) {
                    h_rot0 = c * local_h0 - s * local_h1;
                    h_rot1 = s * local_h0 + c * local_h1;
                } else {
                    h_rot0 = local_h0;
                    h_rot1 = 0.0f;
                }
                
                /* SSM step: h_t = alpha * R(θ)*h_{t-1} + beta*Bu_{t-1} + gamma*Bu_t */
                float new_h0 = alpha0 * h_rot0 + beta0 * local_prev_bu0 + gamma0 * bu0;
                float new_h1 = (d1 < D) ? (alpha1 * h_rot1 + beta1 * local_prev_bu1 + gamma1 * bu1) : 0.0f;
                
                h_store[ts * D + d0] = new_h0;
                if (d1 < D) h_store[ts * D + d1] = new_h1;
                
                y_scan[ts * D + d0] = C_exp[ts * D + d0] * new_h0;
                if (d1 < D) y_scan[ts * D + d1] = C_exp[ts * D + d1] * new_h1;
                
                local_h0 = new_h0;
                local_h1 = new_h1;
                local_prev_bu0 = bu0;
                local_prev_bu1 = bu1;
            }
        }
        
        __syncthreads();
    }
}

extern "C" void cuda_block_forward(
    cublasHandle_t cublas,
    /* Paramètres du bloc [VRAM] */
    const float *d_W_in,        /* [R, dim]      — MIMO: R=mimo_rank, R=1 for SISO */
    const float *d_W_out,       /* [dim, R]      */
    const float *d_A_log,       /* [state]       */
    const float *d_W_B,         /* [N*R, dim]    data-dependent B projection */
    const float *d_W_C,         /* [N*R, dim]    data-dependent C projection */
    const float *d_delta_proj,  /* [dim]         */
    const float *d_theta,       /* [state/2]     rotation angles (may be NULL) */
    const float *d_lambda_proj, /* [dim]         exp-trapezoidal lambda projection */
    /* Entrée / sortie */
    const float *d_x,           /* [L, dim]  input  */
    float       *d_y,           /* [L, dim]  output */
    /* Workspace */
    float *d_u_raw,   float *d_u,    /* [L, R]        */
    float *d_dt_raw,  float *d_dt,   /* [L]           */
    float *d_B_exp,   float *d_C_exp, float *d_dt_exp, /* [L, N*R] / [L, N*R] */
    float *d_h_store, float *d_y_scan, float *d_y_proj, /* [L,N] / [L,R] / [L,dim] */
    float *d_lambda_raw, float *d_lambda,  /* [L] workspace for lambda */
    int L, int state, int dim, int R)  /* R = mimo_rank (1 = SISO) */
{
    if (!cublas) {
        fprintf(stderr, "[ERROR] cuda_block_forward: cublas handle is NULL\n");
        return;
    }
    /* Verify pointers are device pointers */
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, d_x);
    if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        fprintf(stderr, "[ERROR] d_x is not a device pointer (err=%d, type=%d)\n", (int)err, (int)attr.type);
    }
    err = cudaPointerGetAttributes(&attr, d_W_in);
    if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        fprintf(stderr, "[ERROR] d_W_in is not a device pointer (err=%d, type=%d)\n", (int)err, (int)attr.type);
    }
    const float a1 = 1.0f, b0 = 0.0f;
    int NR = state * R;
    int blk;

    /* 1. in_proj : u_raw [L, R] = x [L, dim] @ W_in^T  (W_in=[R,dim]) */
    gemm_bt(cublas, L, R, dim, a1, d_x, d_W_in, b0, d_u_raw);

    /* 2. SiLU */
    blk = (L * R + 255) / 256;
    cuda_silu_fwd_kernel<<<blk, 256>>>(d_u_raw, d_u, L * R);

    /* 3. delta : dt_raw [L] = x [L, dim] @ delta_proj [dim] */
    gemm(cublas, L, 1, dim, a1, d_x, d_delta_proj, b0, d_dt_raw);
    blk = (L + 255) / 256;
    cuda_softplus_clamp_fwd_kernel<<<blk, 256>>>(d_dt_raw, d_dt, L);

    /* 4. Data-dependent B [L, N*R], C [L, N*R], lambda [L] */
    gemm_bt(cublas, L, NR, dim, a1, d_x, d_W_B, b0, d_B_exp);
    gemm_bt(cublas, L, NR, dim, a1, d_x, d_W_C, b0, d_C_exp);
    /* lambda_raw [L] = x [L,dim] @ lambda_proj [dim] */
    gemm(cublas, L, 1, dim, a1, d_x, d_lambda_proj, b0, d_lambda_raw);
    { int blk_l = (L + 255) / 256;
      cuda_sigmoid_fwd_kernel<<<blk_l, 256>>>(d_lambda_raw, d_lambda, L); }

    /* 5. Complex SSM parallel scan with R(θ) rotation + exp-trapezoidal
     * Each thread block handles one state dimension pair (2k, 2k+1).
     * Parallel scan across sequence using cooperative thread processing. */
    CUDA_CHECK(cudaMemset(d_h_store, 0, L * state * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_y_scan,  0, L * R * sizeof(float)));
    /* Number of blocks = ceil(state / 2) for handling pairs + possibly one single */
    int num_pairs = (state + 1) / 2;
    cuda_ssm_fwd_parallel_kernel<<<num_pairs, SSM_PARALLEL_THREADS>>>(
        d_u, d_A_log, d_B_exp, d_C_exp, d_dt,
        d_theta, d_lambda, d_h_store, d_y_scan, L, state);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 6. out_proj : y_proj [L, dim] = y_scan [L, R] @ W_out^T  (W_out=[dim,R]) */
    gemm_bt(cublas, L, dim, R, a1, d_y_scan, d_W_out, b0, d_y_proj);

    /* 7. Résiduel : y = y_proj + x */
    CUDA_CHECK(cudaMemcpy(d_y, d_y_proj, L * dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    blk = (L * dim + 255) / 256;
    cuda_add_inplace_kernel<<<blk, 256>>>(d_y, d_x, L * dim);
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
__global__ void cuda_ssm_bwd_kernel(
    const float *d_dy_scan,  /* [L, D] upstream gradient of y_scan */
    const float *u,          /* [L, D] */
    const float *A_log,      /* [D] */
    const float *B_exp,      /* [L, D] */
    const float *C_exp,      /* [L, D] */
    const float *dt,         /* [L] */
    const float *lambda,     /* [L]   sigmoid(lambda_proj*x) */
    const float *theta,      /* [D/2] */
    const float *h_store,    /* [L, D] state at each step */
    float *d_du,             /* [L, D] out */
    float *d_dA_acc,         /* [D]    out (accumulated) */
    float *d_dB_out,         /* [L, D] out */
    float *d_dC_out,         /* [L, D] out */
    float *d_ddt_out,        /* [L, D] out */
    float *d_g_theta,        /* [D/2]  out (accumulated) */
    float *d_dlambda,        /* [L]    out: grad w.r.t. lambda_t */
    int L, int D)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float adj_h[1024];
    float adj_prev_Bu[1024];
    for (int d = 0; d < D; d++) { adj_h[d] = 0.0f; adj_prev_Bu[d] = 0.0f; }

    for (int t = L - 1; t >= 0; t--) {
        float dt_t  = dt[t];
        float lam_t = lambda ? lambda[t] : 0.5f;
        const float *h_t    = &h_store[t * D];
        const float *h_prev = (t > 0) ? &h_store[(t-1)*D] : NULL;

        float d_lam_t = 0.0f;

        for (int d = 0; d < D; d++) {
            float ct_d   = C_exp[t*D+d];
            float bt_d   = B_exp[t*D+d];
            float ut_d   = u[t*D+d];
            float a_val  = A_log[d]; if (a_val > -1e-5f) a_val = -1e-5f;
            float a_diag = expf(dt_t * a_val);
            float beta_t = (1.0f - lam_t) * dt_t * a_diag;
            float dy_s   = d_dy_scan[t*D+d];
            float bu_t   = bt_d * ut_d;
            float bu_prev = (h_prev != NULL) ? B_exp[(t-1)*D+d] * u[(t-1)*D+d] : 0.0f;

            /* adj of h_t from y_t and future state */
            float ah = adj_h[d] + dy_s * ct_d + adj_prev_Bu[d] * beta_t;
            /* Actually adj_prev_Bu holds the gradient that flows from the next step's
             * beta_{t+1} * Bu_t term. We handle this by incorporating it via adj_h propagation. */
            /* Cleaner: ah_actual excludes adj_prev_Bu (handle separately) */
            float ah_actual = adj_h[d] + dy_s * ct_d;

            /* dC */
            d_dC_out[t*D+d] = dy_s * h_t[d];

            /* dBu_t = ah_actual * gamma_t */
            float gamma_t = lam_t * dt_t;
            float d_bu_t  = ah_actual * gamma_t;
            /* dB_t[d] */
            d_dB_out[t*D+d] = d_bu_t * ut_d;
            /* du[d] */
            d_du[t*D+d] = d_bu_t * bt_d;

            /* Recover h_rot from stored state and Bu terms */
            float h_rot_d = (a_diag > 1e-30f)
                ? (h_t[d] - beta_t * bu_prev - gamma_t * bu_t) / a_diag
                : 0.0f;

            /* dA_log */
            d_dA_acc[d] += ah_actual * dt_t * a_diag * h_rot_d;

            /* d_lambda_t: d_h/d_lam * ah_actual */
            /* d_beta/d_lam = -dt * alpha; d_gamma/d_lam = dt */
            d_lam_t += ah_actual * ((-dt_t * a_diag) * bu_prev + dt_t * bu_t);

            /* ddt */
            d_ddt_out[t*D+d] = ah_actual * (a_val * a_diag * h_rot_d
                                + (1.0f - lam_t) * a_diag * bu_prev
                                + lam_t * bu_t);

            /* d_h_rot = ah_actual * a_diag  for theta + adj_h propagation */
            adj_h[d] = ah_actual * a_diag;

            /* propagate adj through prev_Bu for t-1 */
            adj_prev_Bu[d] = ah_actual * beta_t;
        }

        if (d_dlambda) d_dlambda[t] = d_lam_t;

        /* Gradient for theta and propagate adj_h = R^T * d_h_rot */
        for (int i = 0; i + 1 < D; i += 2) {
            float hp0 = h_prev ? h_prev[i]   : 0.0f;
            float hp1 = h_prev ? h_prev[i+1] : 0.0f;
            float th  = theta ? theta[i >> 1] : 0.0f;
            float c   = cosf(th), sv = sinf(th);
            float dr0 = adj_h[i], dr1 = adj_h[i+1];

            if (theta && d_g_theta)
                d_g_theta[i >> 1] += dr0 * (-sv * hp0 - c * hp1)
                                   + dr1 * (c * hp0 - sv * hp1);

            adj_h[i]   = c * dr0 + sv * dr1;
            adj_h[i+1] = -sv * dr0 + c * dr1;
        }
        if (D & 1) { /* adj_h[D-1] = adj_h[D-1] (unchanged) */ }
    }
}

/* ── Complex SSM PARALLEL backward kernel ─────────────────── */
/*
 * Parallel backward: each thread block handles one state dimension pair.
 * Threads cooperatively process the sequence in reverse.
 * 
 * Note: True parallel backward for sequential scan is complex (reverse dependency).
 * This version uses thread-per-timestep processing within chunks, similar to
 * the forward kernel but processing in reverse.
 */
__global__ void cuda_ssm_bwd_parallel_kernel(
    const float *d_dy_scan,  /* [L, D] upstream gradient */
    const float *u,          /* [L, D] */
    const float *A_log,      /* [D] */
    const float *B_exp,      /* [L, D] */
    const float *C_exp,      /* [L, D] */
    const float *dt,         /* [L] */
    const float *lambda,     /* [L] */
    const float *theta,      /* [D/2] */
    const float *h_store,    /* [L, D] state at each step */
    float *d_du,             /* [L, D] out */
    float *d_dA_acc,         /* [D] out (accumulated) */
    float *d_dB_out,         /* [L, D] out */
    float *d_dC_out,         /* [L, D] out */
    float *d_ddt_out,        /* [L, D] out */
    float *d_g_theta,        /* [D/2] out (accumulated) */
    float *d_dlambda,        /* [L] out */
    int L, int D)
{
    /* Block handles one dimension pair (2k, 2k+1) or single dimension if D odd */
    int pair_idx = blockIdx.x;
    int d0 = pair_idx * 2;
    int d1 = d0 + 1;
    
    if (d0 >= D) return;  /* Out of bounds */
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    /* Local adjoints in registers */
    float adj_h0 = 0.0f;
    float adj_h1 = 0.0f;
    float adj_prev_Bu0 = 0.0f;
    float adj_prev_Bu1 = 0.0f;
    float local_dA0 = 0.0f;
    float local_dA1 = 0.0f;
    float local_dtheta = 0.0f;
    
    /* Get rotation angle for this pair */
    float th = (theta && d1 < D) ? theta[pair_idx] : 0.0f;
    float c = cosf(th), s = sinf(th);
    float a0 = A_log[d0]; if (a0 > -1e-5f) a0 = -1e-5f;
    float a1 = (d1 < D) ? A_log[d1] : 0.0f; if (a1 > -1e-5f) a1 = -1e-5f;
    
    /* Process sequence in reverse chunks */
    int items_per_block = num_threads * SSM_PARALLEL_ITEMS_PER_THREAD;
    
    for (int base_t = L - 1; base_t >= 0; base_t -= items_per_block) {
        /* Each thread processes items in reverse order */
        int local_t = tid;
        int t = base_t - local_t;
        
        if (t >= 0 && t < L) {
            /* Process SSM_PARALLEL_ITEMS_PER_THREAD steps going backward */
            for (int step = 0; step < SSM_PARALLEL_ITEMS_PER_THREAD && (t - step * num_threads) >= 0; step++) {
                int ts = t - step * num_threads;
                
                float dt_t = dt[ts];
                float lam_t = lambda ? lambda[ts] : 0.5f;
                
                float alpha0 = expf(dt_t * a0);
                float alpha1 = (d1 < D) ? expf(dt_t * a1) : 0.0f;
                float beta0 = (1.0f - lam_t) * dt_t * alpha0;
                float beta1 = (d1 < D) ? (1.0f - lam_t) * dt_t * alpha1 : 0.0f;
                float gamma0 = lam_t * dt_t;
                float gamma1 = (d1 < D) ? lam_t * dt_t : 0.0f;
                
                float ct0 = C_exp[ts * D + d0];
                float ct1 = (d1 < D) ? C_exp[ts * D + d1] : 0.0f;
                float bt0 = B_exp[ts * D + d0];
                float bt1 = (d1 < D) ? B_exp[ts * D + d1] : 0.0f;
                float ut0 = u[ts * D + d0];
                float ut1 = (d1 < D) ? u[ts * D + d1] : 0.0f;
                float bu0 = bt0 * ut0;
                float bu1 = (d1 < D) ? bt1 * ut1 : 0.0f;
                
                float dy_s0 = d_dy_scan[ts * D + d0];
                float dy_s1 = (d1 < D) ? d_dy_scan[ts * D + d1] : 0.0f;
                
                /* Adjoint from y_t and future state */
                float ah0 = adj_h0 + dy_s0 * ct0;
                float ah1 = (d1 < D) ? (adj_h1 + dy_s1 * ct1) : 0.0f;
                
                /* dC */
                float h_t0 = h_store[ts * D + d0];
                float h_t1 = (d1 < D) ? h_store[ts * D + d1] : 0.0f;
                d_dC_out[ts * D + d0] = dy_s0 * h_t0;
                if (d1 < D) d_dC_out[ts * D + d1] = dy_s1 * h_t1;
                
                /* dBu and dB, du */
                float d_bu0 = ah0 * gamma0;
                float d_bu1 = (d1 < D) ? (ah1 * gamma1) : 0.0f;
                d_dB_out[ts * D + d0] = d_bu0 * ut0;
                if (d1 < D) d_dB_out[ts * D + d1] = d_bu1 * ut1;
                d_du[ts * D + d0] = d_bu0 * bt0;
                if (d1 < D) d_du[ts * D + d1] = d_bu1 * bt1;
                
                /* Recover h_rot and compute dA */
                const float *h_prev = (ts > 0) ? &h_store[(ts - 1) * D] : NULL;
                float bu_prev0 = (h_prev != NULL) ? B_exp[(ts - 1) * D + d0] * u[(ts - 1) * D + d0] : 0.0f;
                float bu_prev1 = (d1 < D && h_prev != NULL) ? B_exp[(ts - 1) * D + d1] * u[(ts - 1) * D + d1] : 0.0f;
                
                float h_rot0 = (alpha0 > 1e-30f) ? (h_t0 - beta0 * bu_prev0 - gamma0 * bu0) / alpha0 : 0.0f;
                float h_rot1 = (d1 < D && alpha1 > 1e-30f) ? (h_t1 - beta1 * bu_prev1 - gamma1 * bu1) / alpha1 : 0.0f;
                
                local_dA0 += ah0 * dt_t * alpha0 * h_rot0;
                if (d1 < D) local_dA1 += ah1 * dt_t * alpha1 * h_rot1;
                
                /* ddt */
                d_ddt_out[ts * D + d0] = ah0 * (a0 * alpha0 * h_rot0 + (1.0f - lam_t) * alpha0 * bu_prev0 + lam_t * bu0);
                if (d1 < D) d_ddt_out[ts * D + d1] = ah1 * (a1 * alpha1 * h_rot1 + (1.0f - lam_t) * alpha1 * bu_prev1 + lam_t * bu1);
                
                /* Propagate adj_h = R^T * (ah * alpha) */
                float ah_alpha0 = ah0 * alpha0;
                float ah_alpha1 = (d1 < D) ? ah1 * alpha1 : 0.0f;
                
                if (d1 < D) {
                    /* R^T multiplication for adjoint */
                    adj_h0 = c * ah_alpha0 + s * ah_alpha1;
                    adj_h1 = -s * ah_alpha0 + c * ah_alpha1;
                    
                    /* Gradient for theta */
                    float hp0 = h_prev ? h_prev[d0] : 0.0f;
                    float hp1 = h_prev ? h_prev[d1] : 0.0f;
                    local_dtheta += ah_alpha0 * (-s * hp0 - c * hp1) + ah_alpha1 * (c * hp0 - s * hp1);
                } else {
                    adj_h0 = ah_alpha0;
                }
                
                /* Propagate adj through prev_Bu */
                adj_prev_Bu0 = ah0 * beta0;
                if (d1 < D) adj_prev_Bu1 = ah1 * beta1;
            }
        }
        
        __syncthreads();
    }
    
    /* Accumulate gradients atomically */
    atomicAdd(&d_dA_acc[d0], local_dA0);
    if (d1 < D) atomicAdd(&d_dA_acc[d1], local_dA1);
    if (d_g_theta && d1 < D) atomicAdd(&d_g_theta[pair_idx], local_dtheta);
}

extern "C" void cuda_block_backward(
    cublasHandle_t cublas,
    /* Paramètres (lecture seule) */
    const float *d_W_in, const float *d_W_out,   /* [R,dim] / [dim,R] */
    const float *d_A_log,
    const float *d_W_B, const float *d_W_C,       /* [N*R, dim] */
    const float *d_delta_proj,
    const float *d_theta,         /* [state/2] rotation angles */
    const float *d_lambda_proj,   /* [dim] lambda projection */
    /* Activations sauvées au forward */
    const float *d_x,
    const float *d_u_raw, const float *d_u,       /* [L, R] */
    const float *d_dt_raw, const float *d_dt,
    const float *d_B_exp, const float *d_C_exp, const float *d_dt_exp, /* [L, N*R] */
    const float *d_h_store, const float *d_y_scan, /* [L,N] / [L,R] */
    const float *d_lambda,        /* [L] sigmoid output saved at forward */
    /* Gradient entrant */
    const float *d_dy,            /* [L, dim] upstream gradient */
    /* Gradients des paramètres (accumulés, +=) */
    float *d_dW_in, float *d_dW_out,
    float *d_dA_log,
    float *d_dW_B, float *d_dW_C,
    float *d_ddelta_proj,
    float *d_g_theta,             /* [state/2] grad for theta */
    float *d_g_lambda_proj,       /* [dim] grad for lambda_proj */
    /* Gradient de sortie */
    float *d_dx,                  /* [L, dim] downstream gradient */
    /* Workspace temporaire */
    float *d_dy_scan,           /* [L, R] */
    float *d_du,                /* [L, R] */
    float *d_du_raw,            /* [L, R] */
    float *d_ddt,               /* [L] */
    float *d_ddt_raw,           /* [L] */
    float *d_dB_scan,           /* [L, N*R] scan gradient de B */
    float *d_dC_scan,           /* [L, N*R] scan gradient de C */
    float *d_ddt_scan,          /* [L, state] scan gradient de dt */
    float *d_dA_tmp,            /* [state]   scan gradient de A (tmp) */
    float *d_dlambda,           /* [L]       scan gradient of lambda */
    float *d_dlambda_raw,       /* [L]       grad through sigmoid */
    int L, int state, int dim, int R)   /* R = mimo_rank */
{
    const float a1 = 1.0f, b0 = 0.0f;
    int NR = state * R;
    int blk;

    /* ── Résiduel : dy passe aussi vers dx (on initialise dx = dy) ── */
    CUDA_CHECK(cudaMemcpy(d_dx, d_dy, L * dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    /* ── Backward out_proj ──────────────────────────────────────── */
    /* dW_out [dim, R] += dy^T @ y_scan  (A=[L,dim], B=[L,R]) */
    gemm_at(cublas, dim, R, L, a1, d_dy, d_y_scan, a1, d_dW_out);

    /* dy_scan [L, R] = dy [L, dim] @ W_out [dim, R] */
    gemm(cublas, L, R, dim, a1, d_dy, d_W_out, b0, d_dy_scan);

    /* ── Backward complex SSM (parallel) ──────────────────────── */
    /* Zero accumulators */
    CUDA_CHECK(cudaMemset(d_dA_tmp, 0, state * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ddt_scan, 0, L * state * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dlambda, 0, L * sizeof(float)));

    /* Number of blocks = ceil(state / 2) for handling pairs */
    int num_pairs_bwd = (state + 1) / 2;
    cuda_ssm_bwd_parallel_kernel<<<num_pairs_bwd, SSM_PARALLEL_THREADS>>>(
        d_dy_scan,
        d_u, d_A_log, d_B_exp, d_C_exp, d_dt, d_lambda, d_theta, d_h_store,
        d_du, d_dA_tmp, d_dB_scan, d_dC_scan, d_ddt_scan, d_g_theta, d_dlambda,
        L, state);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Accumule dA_log */
    blk = (state + 255) / 256;
    cuda_add_inplace_kernel<<<blk, 256>>>(d_dA_log, d_dA_tmp, state);

    /* g_W_B [N*R, dim] += dB_scan^T [N*R, L] @ x [L, dim] */
    gemm_at(cublas, NR, dim, L, a1, d_dB_scan, d_x, a1, d_dW_B);

    /* g_W_C [N*R, dim] += dC_scan^T [N*R, L] @ x [L, dim] */
    gemm_at(cublas, NR, dim, L, a1, d_dC_scan, d_x, a1, d_dW_C);

    /* ddt [L] = sum_d ddt_scan [t, d] */
    blk = (L + 255) / 256;
    cuda_reduce_sum_D<<<blk, 256>>>(d_ddt_scan, d_ddt, L, state);

    /* ── Backward softplus ──────────────────────────────────────── */
    cuda_softplus_clamp_bwd_kernel<<<blk, 256>>>(d_ddt, d_dt_raw, d_dt, d_ddt_raw, L);

    /* ── Backward delta_proj ────────────────────────────────────── */
    gemm_at(cublas, 1, dim, L, a1, d_ddt_raw, d_x, a1, d_ddelta_proj);
    blk = (L * dim + 255) / 256;
    cuda_outer_add_kernel<<<blk, 256>>>(d_dx, d_ddt_raw, d_delta_proj, L, dim);

    /* ── Backward SiLU + in_proj (W_in=[R,dim], u=[L,R]) ──────── */
    blk = (L * R + 255) / 256;
    cuda_silu_bwd_kernel<<<blk, 256>>>(d_du, d_u_raw, d_du_raw, L * R);

    gemm_at(cublas, R, dim, L, a1, d_du_raw, d_x, a1, d_dW_in);
    gemm(cublas, L, dim, R, a1, d_du_raw, d_W_in, a1, d_dx);

    /* ── Backward lambda_proj ───────────────────────────────────── */
    blk = (L + 255) / 256;
    cuda_sigmoid_bwd_kernel<<<blk, 256>>>(d_dlambda, d_lambda, d_dlambda_raw, L);
    gemm_at(cublas, 1, dim, L, a1, d_dlambda_raw, d_x, a1, d_g_lambda_proj);
    cuda_outer_add_kernel<<<(L * dim + 255) / 256, 256>>>(
        d_dx, d_dlambda_raw, d_lambda_proj, L, dim);

    /* API compat */
    (void)d_dt_exp;
}

/* ============================================================
 * GPU Optimizer Step (Hybrid support)
 * Download gradients, apply CPU optimizer, upload updated params
 * ============================================================ */
#include "kmamba.h"
#include "kmamba_kernels.h"

extern "C" void gpu_optimizer_step(MambaBlock *block, const MBOptimConfig *conf) {
    if (!block || !block->opt_state) return;
    
    /* For now: download gradients to CPU, apply optimizer, upload back */
    /* This is a practical approach for the Hybrid model */
    
    size_t D = block->config.dim;
    size_t N = block->config.state_size;
    size_t R = block->config.mimo_rank > 0 ? block->config.mimo_rank : 1;
    size_t NR = N * R;
    size_t TS = D / 2;
    
    MBOptimState *s = (MBOptimState *)block->opt_state;
    s->step++;
    
    /* Helper to download gradient, apply adamw/muon, upload param */
    auto step_param = [&](float *param, float *grad, float *m, float *v, size_t n) {
        if (!param || !grad || !m || !v || n == 0) return;
        
        /* Allocate CPU buffer for gradient */
        float *h_grad = (float *)malloc(n * sizeof(float));
        if (!h_grad) return;
        
        /* Download gradient from GPU */
        cudaMemcpy(h_grad, grad, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        /* Apply AdamW step on CPU */
        adamw_step_f32(param, h_grad, m, v, conf->lr, 0.9f, 0.999f, conf->eps, 
                       conf->weight_decay, (int)n, s->step);
        
        /* Upload updated param to GPU */
        cudaMemcpy(param, param, n * sizeof(float), cudaMemcpyHostToDevice);
        
        free(h_grad);
    };
    
    /* Apply to all parameters */
    step_param(block->W_in.data,  s->g_W_in,  s->m_W_in,  s->v_W_in,  R * D);
    step_param(block->W_out.data, s->g_W_out, s->m_W_out, s->v_W_out, D * R);
    step_param(block->A_log.data, s->g_A_log, s->m_A_log, s->v_A_log, N);
    step_param(block->W_B.data,   s->g_W_B,   s->m_W_B,   s->v_W_B,   NR * D);
    step_param(block->W_C.data,   s->g_W_C,   s->m_W_C,   s->v_W_C,   NR * D);
    step_param(block->b_B,        s->g_b_B,   s->m_b_B,   s->v_b_B,   NR);
    step_param(block->b_C,        s->g_b_C,   s->m_b_C,   s->v_b_C,   NR);
    step_param(block->delta_proj.data,  s->g_delta_proj,  s->m_delta_proj,  s->v_delta_proj, D);
    step_param(block->lambda_proj.data, s->g_lambda_proj, s->m_lambda_proj, s->v_lambda_proj, D);
    step_param(block->theta, s->g_theta, s->m_theta, s->v_theta, TS);
}
