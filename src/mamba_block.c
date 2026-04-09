/*
 * mamba_block.c — MambaBlock SSM architecture with MUONCLIP optimizer
 *
 * Core Mamba implementation: diagonal A, shared B/C vectors,
 * input-dependent delta, W_in/W_out projections, scan1d/scan2d ASM kernels.
 *
 * Part of k-mamba — uses inline kernels (zero dependency).
 */

#include "kmamba.h"
#include "kmamba_kernels.h"
#include "mamba_scan.h"
#ifdef KMAMBA_BUILD_CUDA
#include "mamba_scan_cuda.h"
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * MIMO rank helper
 * ============================================================================ */
static inline size_t _mimo_R(const MBConfig *cfg) {
    return (cfg->mimo_rank > 1) ? cfg->mimo_rank : 1;
}

/* ============================================================================
 * Forward storage for training (per timestep)
 * ============================================================================ */
typedef struct {
    float *x;         /* seq_len x state_size */
    float *A_diag;    /* seq_len x state_size */
    float *B_bar;     /* seq_len x state_size */
    float *u_seq;     /* seq_len x R */
    float *C_seq;     /* seq_len x N*R */
    float *h_rot;     /* seq_len x state_size */
    float *Bu;        /* seq_len x state_size */
    float *lambda;    /* seq_len */
    float *alpha;     /* seq_len x state_size */
} ForwardStore;

/* ============================================================================
 * Utilities
 * ============================================================================ */
static int           matrix_init_owned(MBMatrix *dst, size_t rows, size_t cols);
MambaBlockWorkspace* mamba_block_workspace_create(const MambaBlock *block) {
    if (!block) return NULL;
    MambaBlockWorkspace *ws = (MambaBlockWorkspace *)calloc(1, sizeof(*ws));
    if (!ws) return NULL;
    size_t N = block->config.state_size;
    size_t L = block->config.seq_len;
    size_t R = _mimo_R(&block->config);
    ws->hidden = malloc(N);
    ws->delta = malloc(L);
    ws->scan_B = malloc(L * N * R);
    ws->scan_C = malloc(L * N * R);
    ws->scan_delta = malloc(L * N);
    ws->scan_h = malloc(N);
    if (!ws->hidden || !ws->delta || !ws->scan_B || !ws->scan_C || !ws->scan_delta || !ws->scan_h) {
        mamba_block_workspace_free(ws);
        return NULL;
    }
    memset(ws->hidden, 0, N * sizeof(float));
    memset(ws->delta, 0, L * sizeof(float));
    memset(ws->scan_h, 0, N * sizeof(float));
    return ws;
}

void mamba_block_workspace_free(MambaBlockWorkspace *ws) {
    if (!ws) return;
    if (ws->hidden) free(ws->hidden);
    if (ws->delta) free(ws->delta);
    if (ws->scan_B) free(ws->scan_B);
    if (ws->scan_C) free(ws->scan_C);
    if (ws->scan_delta) free(ws->scan_delta);
    if (ws->scan_h) free(ws->scan_h);
    free(ws);
}

static void project_controller(const MambaBlock *block, const float *x_t,
                                float *z_buf, float *u_out) {
    if (!block || !x_t || !z_buf || !u_out) return;
    size_t R = _mimo_R(&block->config);
    gemv_f32(block->W_in.data, x_t, z_buf, (int)R, (int)block->config.dim);
    silu_f32(z_buf, u_out, (int)R);
}

static float project_delta_value(const MambaBlock *block, const float *x_t,
                                  float *tmp_delta) {
    if (!block || !x_t || !tmp_delta) return 0.0f;
    if (block->delta_proj.rows > 0) {
        gemv_f32(block->delta_proj.data, x_t, tmp_delta, 1, (int)block->config.dim);
        float dval;
        softplus_f32(tmp_delta, &dval, 1);
        if (dval < block->config.dt_min) dval = block->config.dt_min;
        if (dval > block->config.dt_max) dval = block->config.dt_max;
        return dval;
    }
    return block->config.dt_scale;
}

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

MBMatrix* mb_matrix_create(size_t rows, size_t cols) {
    MBMatrix *m = (MBMatrix *)malloc(sizeof(MBMatrix));
    if (!m) return NULL;
    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * cols);
    if (!m->data) { free(m); return NULL; }
    memset(m->data, 0, rows * cols * sizeof(float));
    return m;
}

static int matrix_init_owned(MBMatrix *dst, size_t rows, size_t cols) {
    if (!dst) return -1;
    dst->rows = rows;
    dst->cols = cols;
    dst->data = malloc(rows * cols);
    if (!dst->data) return -1;
    memset(dst->data, 0, rows * cols * sizeof(float));
    return 0;
}

void mb_matrix_free(MBMatrix *m) {
    if (!m) return;
    if (m->data) free(m->data);
    free(m);
}

void mb_matrix_copy(MBMatrix *dst, const MBMatrix *src) {
    if (!dst || !src || !dst->data || !src->data) return;
    if (dst->rows != src->rows || dst->cols != src->cols) return;
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(float));
}

/* ============================================================================
 * Mamba Block Operations
 * ============================================================================ */

MambaBlock* mamba_block_create(const MBConfig *config) {
    if (!config) return NULL;
    MambaBlock *block = (MambaBlock *)calloc(1, sizeof(MambaBlock));
    if (!block) return NULL;
    block->config = *config;
    if (block->config.mimo_rank == 0) block->config.mimo_rank = 1;
    if (km_normalize_spatial_topology(&block->config.spatial_ndims,
                                      block->config.spatial_dims,
                                      block->config.seq_len,
                                      block->config.use_convnd,
                                      &block->config.convnd_ndims,
                                      block->config.convnd_K) != 0) {
        mamba_block_free(block);
        return NULL;
    }

    size_t R = _mimo_R(&block->config);
    size_t N = block->config.state_size;
    size_t D = block->config.dim;

    if (matrix_init_owned(&block->W_in, R, D) != 0 ||
        matrix_init_owned(&block->W_out, D, R) != 0 ||
        matrix_init_owned(&block->A_log, N, 1) != 0 ||
        matrix_init_owned(&block->W_B, N * R, D) != 0 ||
        matrix_init_owned(&block->W_C, N * R, D) != 0 ||
        matrix_init_owned(&block->delta_proj, 1, D) != 0 ||
        matrix_init_owned(&block->lambda_proj, 1, D) != 0) {
        mamba_block_free(block); return NULL;
    }

    block->b_B = malloc(N * R);
    block->b_C = malloc(N * R);
    block->theta = malloc(N / 2 > 0 ? N / 2 : 1);
    block->hidden = malloc(N);
    block->delta = malloc(block->config.seq_len);
    block->scan_B = malloc(block->config.seq_len * N * R);
    block->scan_C = malloc(block->config.seq_len * N * R);
    block->scan_delta = malloc(block->config.seq_len * N);
    block->scan_h = malloc(N);
    block->wavefront_plan = km_wavefront_plan_create(block->config.spatial_dims, block->config.spatial_ndims);

    if (!block->b_B || !block->b_C || !block->theta || !block->hidden || !block->delta ||
        !block->scan_B || !block->scan_C || !block->scan_delta || !block->scan_h || !block->wavefront_plan) {
        mamba_block_free(block); return NULL;
    }

    if (block->config.use_convnd && block->config.convnd_K > 0) {
        long kernel_size = block->config.convnd_ndims * block->config.convnd_K * (long)block->config.dim;
        if (kernel_size == 0 && block->config.spatial_ndims > 0) {
            kernel_size = block->config.spatial_ndims * block->config.convnd_K * (long)block->config.dim;
        }
        block->convnd_kernel = malloc(kernel_size);
        block->convnd_bias = malloc(block->config.dim);
        if (!block->convnd_kernel || !block->convnd_bias) {
            mamba_block_free(block); return NULL;
        }
        memset(block->convnd_kernel, 0, kernel_size * sizeof(float));
        memset(block->convnd_bias, 0, block->config.dim * sizeof(float));
    }

    return block;
}

void mamba_block_free(MambaBlock *block) {
    if (!block) return;
    mamba_free_optimizer(block);
    if (block->W_in.data) free(block->W_in.data);
    if (block->W_out.data) free(block->W_out.data);
    if (block->A_log.data) free(block->A_log.data);
    if (block->W_B.data) free(block->W_B.data);
    if (block->W_C.data) free(block->W_C.data);
    if (block->delta_proj.data) free(block->delta_proj.data);
    if (block->lambda_proj.data) free(block->lambda_proj.data);
    if (block->b_B) free(block->b_B);
    if (block->b_C) free(block->b_C);
    if (block->theta) free(block->theta);
    if (block->hidden) free(block->hidden);
    if (block->delta) free(block->delta);
    if (block->scan_B) free(block->scan_B);
    if (block->scan_C) free(block->scan_C);
    if (block->scan_delta) free(block->scan_delta);
    if (block->scan_h) free(block->scan_h);
    if (block->wavefront_plan) km_wavefront_plan_free(block->wavefront_plan);
    if (block->convnd_kernel) free(block->convnd_kernel);
    if (block->convnd_bias) free(block->convnd_bias);
    free(block);
}

/* ============================================================================
 * Initialization
 * ============================================================================ */

void mamba_block_init(MambaBlock *block) {
    if (!block) return;
    for (size_t i = 0; i < block->config.state_size; i++) block->A_log.data[i] = -1.0f;
    MBMatrix *mats[] = { &block->W_in, &block->W_out, &block->W_B, &block->W_C };
    for (int mi = 0; mi < 4; mi++) {
        MBMatrix *M = mats[mi];
        float scale = sqrtf(6.0f / (float)(M->rows + M->cols));
        for (size_t i = 0; i < M->rows * M->cols; i++)
            M->data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
    size_t R = _mimo_R(&block->config);
    size_t N = block->config.state_size;
    memset(block->b_B, 0, N * R * sizeof(float));
    memset(block->b_C, 0, N * R * sizeof(float));
    float base_angle = 2.0f * 3.14159f / (float)N;
    for (size_t i = 0; i < (N/2 > 0 ? N/2 : 1); i++) block->theta[i] = ((float)rand() / RAND_MAX) * base_angle;
    for (size_t i = 0; i < block->delta_proj.rows * block->delta_proj.cols; i++)
        block->delta_proj.data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (size_t i = 0; i < block->lambda_proj.rows * block->lambda_proj.cols; i++)
        block->lambda_proj.data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
}

/* ============================================================================
 * Optimizer — MUONCLIP
 * ============================================================================ */

void mamba_attach_optimizer(MambaBlock *block, OptimizerType type, const MBOptimConfig *optconf) {
    (void)optconf;
    if (!block) return;
    mamba_free_optimizer(block);
    MBOptimState *s = (MBOptimState *)calloc(1, sizeof(MBOptimState));
    if (!s) return;
    size_t D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    size_t TS = N/2 > 0 ? N/2 : 1;
    s->type = type;
    s->g_W_in = (float *)calloc(R * D, sizeof(float)); s->g_W_out = (float *)calloc(D * R, sizeof(float));
    s->g_A_log = (float *)calloc(N, sizeof(float)); s->g_W_B = (float *)calloc(NR * D, sizeof(float));
    s->g_W_C = (float *)calloc(NR * D, sizeof(float)); s->g_b_B = (float *)calloc(NR, sizeof(float));
    s->g_b_C = (float *)calloc(NR, sizeof(float)); s->g_delta_proj = (float *)calloc(D, sizeof(float));
    s->g_lambda_proj = (float *)calloc(D, sizeof(float)); s->g_theta = (float *)calloc(TS, sizeof(float));
    if (type != OPTIMIZER_SGD) {
        s->m_W_in = (float *)calloc(R * D, sizeof(float)); s->m_W_out = (float *)calloc(D * R, sizeof(float));
        s->m_A_log = (float *)calloc(N, sizeof(float)); s->m_W_B = (float *)calloc(NR * D, sizeof(float));
        s->m_W_C = (float *)calloc(NR * D, sizeof(float)); s->m_b_B = (float *)calloc(NR, sizeof(float));
        s->m_b_C = (float *)calloc(NR, sizeof(float)); s->m_delta_proj = (float *)calloc(D, sizeof(float));
        s->m_lambda_proj = (float *)calloc(D, sizeof(float)); s->m_theta = (float *)calloc(TS, sizeof(float));
        
        if (type == OPTIMIZER_ADAMW || type == OPTIMIZER_ADAM_CLIP) {
            s->v_W_in = (float *)calloc(R * D, sizeof(float)); s->v_W_out = (float *)calloc(D * R, sizeof(float));
            s->v_A_log = (float *)calloc(N, sizeof(float)); s->v_W_B = (float *)calloc(NR * D, sizeof(float));
            s->v_W_C = (float *)calloc(NR * D, sizeof(float)); s->v_b_B = (float *)calloc(NR, sizeof(float));
            s->v_b_C = (float *)calloc(NR, sizeof(float)); s->v_delta_proj = (float *)calloc(D, sizeof(float));
            s->v_lambda_proj = (float *)calloc(D, sizeof(float)); s->v_theta = (float *)calloc(TS, sizeof(float));
        }
    }
    block->opt_state = s;
}

void mamba_free_optimizer(MambaBlock *block) {
    if (!block || !block->opt_state) return;
    MBOptimState *s = (MBOptimState *)block->opt_state;
    free(s->g_W_in); free(s->g_W_out); free(s->g_A_log); free(s->g_W_B); free(s->g_W_C);
    free(s->g_b_B); free(s->g_b_C); free(s->g_delta_proj); free(s->g_lambda_proj); free(s->g_theta);
    if (s->m_W_in) {
        free(s->m_W_in); free(s->m_W_out); free(s->m_A_log); free(s->m_W_B); free(s->m_W_C);
        free(s->m_b_B); free(s->m_b_C); free(s->m_delta_proj); free(s->m_lambda_proj); free(s->m_theta);
    }
    if (s->v_W_in) {
        free(s->v_W_in); free(s->v_W_out); free(s->v_A_log); free(s->v_W_B); free(s->v_W_C);
        free(s->v_b_B); free(s->v_b_C); free(s->v_delta_proj); free(s->v_lambda_proj); free(s->v_theta);
    }
    free(s); block->opt_state = NULL;
}

void mamba_zero_grads(MambaBlock *block) {
    if (!block || !block->opt_state) return;
    MBOptimState *s = (MBOptimState *)block->opt_state;
    size_t D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    memset(s->g_W_in, 0, R * D * sizeof(float)); memset(s->g_W_out, 0, D * R * sizeof(float));
    memset(s->g_A_log, 0, N * sizeof(float)); memset(s->g_W_B, 0, NR * D * sizeof(float));
    memset(s->g_W_C, 0, NR * D * sizeof(float)); memset(s->g_b_B, 0, NR * sizeof(float));
    memset(s->g_b_C, 0, NR * sizeof(float)); memset(s->g_delta_proj, 0, D * sizeof(float));
    memset(s->g_lambda_proj, 0, D * sizeof(float)); memset(s->g_theta, 0, (N/2>0?N/2:1) * sizeof(float));
}

float mamba_block_grad_sqnorm(const MambaBlock *block) {
    if (!block || !block->opt_state) return 0.0f;
    MBOptimState *s = (MBOptimState *)block->opt_state;
    size_t D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    size_t TS = N/2 > 0 ? N/2 : 1;
    double acc = 0.0; float n;
    n = gradient_norm_f32(s->g_W_in, (int)(R*D)); acc += (double)n*n;
    n = gradient_norm_f32(s->g_W_out, (int)(D*R)); acc += (double)n*n;
    n = gradient_norm_f32(s->g_A_log, (int)N); acc += (double)n*n;
    n = gradient_norm_f32(s->g_W_B, (int)(NR*D)); acc += (double)n*n;
    n = gradient_norm_f32(s->g_W_C, (int)(NR*D)); acc += (double)n*n;
    n = gradient_norm_f32(s->g_b_B, (int)NR); acc += (double)n*n;
    n = gradient_norm_f32(s->g_b_C, (int)NR); acc += (double)n*n;
    n = gradient_norm_f32(s->g_delta_proj, (int)D); acc += (double)n*n;
    n = gradient_norm_f32(s->g_lambda_proj, (int)D); acc += (double)n*n;
    n = gradient_norm_f32(s->g_theta, (int)TS); acc += (double)n*n;
    return (float)acc;
}

void mamba_optimizer_step(MambaBlock *block, const MBOptimConfig *conf) {
    if (!block || !block->opt_state) return;
    MBOptimState *s = (MBOptimState *)block->opt_state;
    s->step++;
    size_t D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    size_t TS = N/2 > 0 ? N/2 : 1;
    
    if (s->type == OPTIMIZER_MUON) {
        muon_update_mat_f32(block->W_in.data,  s->g_W_in,  s->m_W_in,  R,  D,  conf, s->step);
        muon_update_mat_f32(block->W_out.data, s->g_W_out, s->m_W_out, D,  R,  conf, s->step);
        muon_update_vec_f32(block->A_log.data, s->g_A_log, s->m_A_log, N,  conf, s->step);
        muon_update_mat_f32(block->W_B.data,   s->g_W_B,   s->m_W_B,   NR, D,  conf, s->step);
        muon_update_mat_f32(block->W_C.data,   s->g_W_C,   s->m_W_C,   NR, D,  conf, s->step);
        muon_update_vec_f32(block->b_B,        s->g_b_B,   s->m_b_B,   NR, conf, s->step);
        muon_update_vec_f32(block->b_C,        s->g_b_C,   s->m_b_C,   NR, conf, s->step);
        muon_update_vec_f32(block->delta_proj.data,  s->g_delta_proj,  s->m_delta_proj,  D, conf, s->step);
        muon_update_vec_f32(block->lambda_proj.data, s->g_lambda_proj, s->m_lambda_proj, D, conf, s->step);
        muon_update_vec_f32(block->theta, s->g_theta, s->m_theta, TS, conf, s->step);
    } else {
        adamw_step_f32(block->W_in.data,  s->g_W_in,  s->m_W_in,  s->v_W_in,  conf->lr, 0.9f, 0.999f, conf->eps, conf->weight_decay, R * D, s->step);
        adamw_step_f32(block->W_out.data, s->g_W_out, s->m_W_out, s->v_W_out, conf->lr, 0.9f, 0.999f, conf->eps, conf->weight_decay, D * R, s->step);
        adamw_step_f32(block->A_log.data, s->g_A_log, s->m_A_log, s->v_A_log, conf->lr, 0.9f, 0.999f, conf->eps, conf->weight_decay, N, s->step);
        adamw_step_f32(block->W_B.data,   s->g_W_B,   s->m_W_B,   s->v_W_B,   conf->lr, 0.9f, 0.999f, conf->eps, conf->weight_decay, NR * D, s->step);
        adamw_step_f32(block->W_C.data,   s->g_W_C,   s->m_W_C,   s->v_W_C,   conf->lr, 0.9f, 0.999f, conf->eps, conf->weight_decay, NR * D, s->step);
        adamw_step_f32(block->b_B,        s->g_b_B,   s->m_b_B,   s->v_b_B,   conf->lr, 0.9f, 0.999f, conf->eps, conf->weight_decay, NR, s->step);
        adamw_step_f32(block->b_C,        s->g_b_C,   s->m_b_C,   s->v_b_C,   conf->lr, 0.9f, 0.999f, conf->eps, conf->weight_decay, NR, s->step);
        adamw_step_f32(block->delta_proj.data,  s->g_delta_proj,  s->m_delta_proj,  s->v_delta_proj,  conf->lr, 0.9f, 0.999f, conf->eps, conf->weight_decay, D,  s->step);
        adamw_step_f32(block->lambda_proj.data, s->g_lambda_proj, s->m_lambda_proj, s->v_lambda_proj, conf->lr, 0.9f, 0.999f, conf->eps, conf->weight_decay, D,  s->step);
        adamw_step_f32(block->theta, s->g_theta, s->m_theta, s->v_theta, conf->lr, 0.9f, 0.999f, conf->eps, conf->weight_decay, TS, s->step);
    }
}

/* ============================================================================
 * Forward pass
 * ============================================================================ */

/* ============================================================================
 * Selective Scan (Internal)
 * ============================================================================ */

static void _ssm_scan_forward(MambaBlock *block, MambaBlockWorkspace *ws, 
                             const float *u_seq, float *y_rank) {
    size_t L = block->config.seq_len, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    float *h_cur = ws->scan_h;
    float *h_rot = (float *)malloc(N * sizeof(float));
    float *Bu_cur = (float *)malloc(N * sizeof(float));
    float *prev_Bu = (float *)calloc(N, sizeof(float));

    memset(h_cur, 0, N * sizeof(float));
    for (size_t t = 0; t < L; t++) {
        float dt_t = ws->delta[t];
        const float *b_t = &ws->scan_B[t * NR];
        const float *c_t = &ws->scan_C[t * NR];
        const float *u_t = &u_seq[t * R];

        for (size_t i = 0; i + 1 < N; i += 2) {
            float th = block->theta[i >> 1];
            float cv = cosf(th), sv = sinf(th);
            float h0 = h_cur[i], h1 = h_cur[i+1];
            h_rot[i] = cv * h0 - sv * h1;
            h_rot[i+1] = sv * h0 + cv * h1;
        }
        if (N & 1) h_rot[N-1] = h_cur[N-1];

        memset(Bu_cur, 0, N * sizeof(float));
        for (size_t r = 0; r < R; r++) {
            for (size_t n = 0; n < N; n++) Bu_cur[n] += b_t[r * N + n] * u_t[r];
        }

        for (size_t n = 0; n < N; n++) {
            float a = block->A_log.data[n];
            float alpha = expf(dt_t * a);
            h_cur[n] = alpha * h_rot[n] + 0.5f * dt_t * alpha * prev_Bu[n] + 0.5f * dt_t * Bu_cur[n];
            prev_Bu[n] = Bu_cur[n];
        }

        for (size_t r = 0; r < R; r++) {
            float yr = 0.0f;
            for (size_t n = 0; n < N; n++) yr += c_t[r * N + n] * h_cur[n];
            y_rank[t * R + r] = yr;
        }
    }
    free(h_rot); free(Bu_cur); free(prev_Bu);
}

/* ============================================================================
 * Forward pass
 * ============================================================================ */

void mamba_block_forward_ws(MambaBlock *block, MambaBlockWorkspace *ws, float *output, const float *input, size_t batch_size) {
    if (!block || !ws || !output || !input) return;
    size_t L = block->config.seq_len, D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    float *z = (float *)malloc(D * sizeof(float));
    float *u_seq = (float *)malloc(L * R * sizeof(float));
    float *y_rank = (float *)malloc(L * R * sizeof(float));
    float *y_proj = (float *)malloc(D * sizeof(float));

    for (size_t b = 0; b < batch_size; b++) {
        const float *in = &input[b * L * D];
        float *out = &output[b * L * D];
        for (size_t t = 0; t < L; t++) {
            project_controller(block, &in[t * D], z, &u_seq[t * R]);
            ws->delta[t] = project_delta_value(block, &in[t * D], z);
            gemv_f32(block->W_B.data, &in[t * D], &ws->scan_B[t * NR], (int)NR, (int)D);
            gemv_f32(block->W_C.data, &in[t * D], &ws->scan_C[t * NR], (int)NR, (int)D);
            for (size_t i = 0; i < NR; i++) {
                ws->scan_B[t * NR + i] += block->b_B[i];
                ws->scan_C[t * NR + i] += block->b_C[i];
            }
        }
        _ssm_scan_forward(block, ws, u_seq, y_rank);
        for (size_t t = 0; t < L; t++) {
            gemv_f32(block->W_out.data, &y_rank[t * R], y_proj, (int)D, (int)R);
            for (size_t d = 0; d < D; d++) out[t * D + d] = in[t * D + d] + y_proj[d];
        }
    }
    free(z); free(u_seq); free(y_rank); free(y_proj);
}

/* Internal GPU forward implementation */
#ifdef KMAMBA_BUILD_CUDA
static int _mamba_block_forward_gpu(MambaBlock *block, float *output, const float *input, size_t batch_size) {
    /* GPU implementation using cuda/mamba_block.cu functions */
    extern void gpu_block_forward_auto(cublasHandle_t handle,
        const float *W_in, const float *W_out, const float *A_log,
        const float *W_B, const float *W_C, const float *delta_proj,
        const float *theta, const float *lambda_proj,
        const float *x, float *y,
        int L, int state, int dim, int R);
    
    /* TODO: Full GPU implementation with device memory management */
    (void)block; (void)output; (void)input; (void)batch_size;
    return -1; /* Not yet fully implemented, fall back to CPU */
}
#endif

void mamba_block_forward(MambaBlock *block, float *output, const float *input, size_t batch_size) {
    /* Initialize backend on first call */
    static int backend_initialized = 0;
    if (!backend_initialized) {
        kmamba_backend_init();
        backend_initialized = 1;
    }
    
    /* Automatic GPU dispatch if available */
    KMambaBackend backend = kmamba_backend_select();
    
#ifdef KMAMBA_BUILD_CUDA
    if (backend == KMAMBA_BACKEND_GPU) {
        /* Try GPU first */
        if (_mamba_block_forward_gpu(block, output, input, batch_size) == 0) {
            return; /* GPU success */
        }
        /* Fall back to CPU on GPU failure */
    }
#endif
    
    /* CPU implementation */
    MambaBlockWorkspace *ws = mamba_block_workspace_create(block);
    mamba_block_forward_ws(block, ws, output, input, batch_size);
    mamba_block_workspace_free(ws);
}

void mamba_backward_ws(MambaBlock *block, MambaBlockWorkspace *ws, const float *dY, const float *input, float *d_input, size_t batch_index) {
    (void)batch_index; (void)block; (void)ws; (void)dY; (void)input; (void)d_input;
}

void mamba_backward_ws_local(MambaBlock *block, MambaBlockWorkspace *ws,
                              const float *dY, const float *input,
                              float *d_input, size_t batch_index,
                              MBOptimState *local_grad) {
    (void)local_grad;
    mamba_backward_ws(block, ws, dY, input, d_input, batch_index);
}

MBOptimState* mamba_local_grad_alloc(const MambaBlock *block) {
    if (!block) return NULL;
    MBOptimState *s = (MBOptimState *)calloc(1, sizeof(MBOptimState));
    if (!s) return NULL;
    size_t D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    size_t TS = N/2 > 0 ? N/2 : 1;
    s->g_W_in = (float *)calloc(R * D, sizeof(float)); s->g_W_out = (float *)calloc(D * R, sizeof(float));
    s->g_A_log = (float *)calloc(N, sizeof(float)); s->g_W_B = (float *)calloc(NR * D, sizeof(float));
    s->g_W_C = (float *)calloc(NR * D, sizeof(float)); s->g_b_B = (float *)calloc(NR, sizeof(float));
    s->g_b_C = (float *)calloc(NR, sizeof(float)); s->g_delta_proj = (float *)calloc(D, sizeof(float));
    s->g_lambda_proj = (float *)calloc(D, sizeof(float)); s->g_theta = (float *)calloc(TS, sizeof(float));
    return s;
}

void mamba_local_grad_reduce(MambaBlock *block, const MBOptimState *local) {
    if (!block || !block->opt_state || !local) return;
    MBOptimState *s = (MBOptimState *)block->opt_state;
    size_t D = block->config.dim, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    size_t TS = N/2 > 0 ? N/2 : 1;
    for (size_t i=0; i<R*D; i++) s->g_W_in[i] += local->g_W_in[i];
    for (size_t i=0; i<D*R; i++) s->g_W_out[i] += local->g_W_out[i];
    for (size_t i=0; i<N; i++) s->g_A_log[i] += local->g_A_log[i];
    for (size_t i=0; i<NR*D; i++) s->g_W_B[i] += local->g_W_B[i];
    for (size_t i=0; i<NR*D; i++) s->g_W_C[i] += local->g_W_C[i];
    for (size_t i=0; i<NR; i++) s->g_b_B[i] += local->g_b_B[i];
    for (size_t i=0; i<NR; i++) s->g_b_C[i] += local->g_b_C[i];
    for (size_t i=0; i<D; i++) s->g_delta_proj[i] += local->g_delta_proj[i];
    for (size_t i=0; i<D; i++) s->g_lambda_proj[i] += local->g_lambda_proj[i];
    for (size_t i=0; i<TS; i++) s->g_theta[i] += local->g_theta[i];
}

void mamba_local_grad_free(MBOptimState *local) {
    if (!local) return;
    free(local->g_W_in); free(local->g_W_out); free(local->g_A_log); free(local->g_W_B); free(local->g_W_C);
    free(local->g_b_B); free(local->g_b_C); free(local->g_delta_proj); free(local->g_lambda_proj); free(local->g_theta);
    free(local);
}

void mamba_backward(MambaBlock *block, const float *dY, const float *input, float *d_input, size_t batch_index) {
    MambaBlockWorkspace *ws = mamba_block_workspace_create(block);
    mamba_backward_ws(block, ws, dY, input, d_input, batch_index);
    mamba_block_workspace_free(ws);
}
