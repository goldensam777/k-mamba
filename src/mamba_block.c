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
    ws->hidden = malloc(N * sizeof(float));
    ws->delta = malloc(L * sizeof(float));
    ws->scan_B = malloc(L * N * R * sizeof(float));
    ws->scan_C = malloc(L * N * R * sizeof(float));
    ws->scan_delta = malloc(L * N * sizeof(float));
    ws->scan_h = malloc(N * sizeof(float));
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

typedef struct {
    float delta;   /* dt_t après softplus + clamp */
    float lambda;  /* lambda_t après sigmoid */
} Mamba3Control;

static void project_controller_full(const MambaBlock *block, const float *x_t,
                                     float *z_buf, float *u_out,
                                     Mamba3Control *ctrl) {
    if (!block || !x_t || !z_buf || !u_out || !ctrl) return;
    size_t R = _mimo_R(&block->config);
    size_t D = block->config.dim;

    /* 1. W_in @ x_t → z_buf, SiLU → u_out (MIMO: u_t ∈ R^R) */
    gemv_f32(block->W_in.data, x_t, z_buf, (int)R, (int)D);
    silu_f32(z_buf, u_out, (int)R);

    /* 2. delta_proj @ x_t → delta, softplus + clamp */
    if (block->delta_proj.rows > 0 && block->delta_proj.data) {
        float dval;
        gemv_f32(block->delta_proj.data, x_t, &dval, 1, (int)D);
        softplus_f32(&dval, &dval, 1);
        if (dval < block->config.dt_min) dval = block->config.dt_min;
        if (dval > block->config.dt_max) dval = block->config.dt_max;
        ctrl->delta = dval;
    } else {
        ctrl->delta = block->config.dt_scale;
    }

    /* 3. lambda_proj @ x_t → lambda_raw, sigmoid → lambda */
    if (block->lambda_proj.rows > 0 && block->lambda_proj.data) {
        float lambda_raw;
        gemv_f32(block->lambda_proj.data, x_t, &lambda_raw, 1, (int)D);
        ctrl->lambda = 1.0f / (1.0f + expf(-lambda_raw));  /* sigmoid */
    } else {
        ctrl->lambda = 0.5f;  /* default: trapezoidal symétrique */
    }
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
    m->data = malloc(rows * cols * sizeof(float));
    if (!m->data) { free(m); return NULL; }
    memset(m->data, 0, rows * cols * sizeof(float));
    return m;
}

static int matrix_init_owned(MBMatrix *dst, size_t rows, size_t cols) {
    if (!dst) return -1;
    dst->rows = rows;
    dst->cols = cols;
    dst->data = malloc(rows * cols * sizeof(float));
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

    block->b_B = malloc(N * R * sizeof(float));
    block->b_C = malloc(N * R * sizeof(float));
    block->theta = malloc((N / 2 > 0 ? N / 2 : 1) * sizeof(float));
    block->hidden = malloc(N * sizeof(float));
    block->delta = malloc(block->config.seq_len * sizeof(float));
    block->scan_B = malloc(block->config.seq_len * N * R * sizeof(float));
    block->scan_C = malloc(block->config.seq_len * N * R * sizeof(float));
    block->scan_delta = malloc(block->config.seq_len * N * sizeof(float));
    block->scan_h = malloc(N * sizeof(float));
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
                             const float *u_seq, const float *lambda_seq, float *y_rank) {
    size_t L = block->config.seq_len, N = block->config.state_size, R = _mimo_R(&block->config), NR = N * R;
    float *h_cur = ws->scan_h;
    float *h_rot = (float *)malloc(N * sizeof(float));
    float *Bu_cur = (float *)malloc(N * sizeof(float));
    float *prev_Bu = (float *)calloc(N, sizeof(float));

    memset(h_cur, 0, N * sizeof(float));
    for (size_t t = 0; t < L; t++) {
        float dt_t = ws->delta[t];
        float lam_t = lambda_seq ? lambda_seq[t] : 0.5f;
        const float *b_t = &ws->scan_B[t * NR];
        const float *c_t = &ws->scan_C[t * NR];
        const float *u_t = &u_seq[t * R];

        /* 1. Appliquer R(θ) à h_cur → h_rot */
        for (size_t i = 0; i + 1 < N; i += 2) {
            float th = block->theta[i >> 1];
            float cv = cosf(th), sv = sinf(th);
            float h0 = h_cur[i], h1 = h_cur[i+1];
            h_rot[i]   = cv * h0 - sv * h1;
            h_rot[i+1] = sv * h0 + cv * h1;
        }
        if (N & 1) h_rot[N-1] = h_cur[N-1];

        /* 2. Calculer Bu_t = sum_r B_t[n,r] * u_t[r] pour chaque n */
        memset(Bu_cur, 0, N * sizeof(float));
        for (size_t r = 0; r < R; r++) {
            for (size_t n = 0; n < N; n++) Bu_cur[n] += b_t[r * N + n] * u_t[r];
        }

        /* 3. Mamba-3 exp-trapezoidal: h_t = alpha*R(θ)*h_{t-1} + beta*Bu_{t-1} + gamma*Bu_t */
        for (size_t n = 0; n < N; n++) {
            float a = block->A_log.data[n];
            float alpha = expf(dt_t * a);
            float beta  = (1.0f - lam_t) * dt_t * alpha;  /* ← terme Bu_{t-1} */
            float gamma = lam_t * dt_t;                    /* ← terme Bu_t */
            h_cur[n] = alpha * h_rot[n] + beta * prev_Bu[n] + gamma * Bu_cur[n];
        }

        /* 4. Output: y_t[r] = sum_n C_t[n,r] * h_t[n] */
        for (size_t r = 0; r < R; r++) {
            float yr = 0.0f;
            for (size_t n = 0; n < N; n++) yr += c_t[r * N + n] * h_cur[n];
            y_rank[t * R + r] = yr;
        }

        /* 5. Stocker Bu_cur pour le prochain timestep */
        memcpy(prev_Bu, Bu_cur, N * sizeof(float));
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
    float *lambda_seq = (float *)malloc(L * sizeof(float));
    float *y_rank = (float *)malloc(L * R * sizeof(float));
    float *y_proj = (float *)malloc(D * sizeof(float));

    for (size_t b = 0; b < batch_size; b++) {
        const float *in = &input[b * L * D];
        float *out = &output[b * L * D];
        for (size_t t = 0; t < L; t++) {
            Mamba3Control ctrl;
            project_controller_full(block, &in[t * D], z, &u_seq[t * R], &ctrl);
            ws->delta[t] = ctrl.delta;
            lambda_seq[t] = ctrl.lambda;
            gemv_f32(block->W_B.data, &in[t * D], &ws->scan_B[t * NR], (int)NR, (int)D);
            gemv_f32(block->W_C.data, &in[t * D], &ws->scan_C[t * NR], (int)NR, (int)D);
            for (size_t i = 0; i < NR; i++) {
                ws->scan_B[t * NR + i] += block->b_B[i];
                ws->scan_C[t * NR + i] += block->b_C[i];
            }
        }
        _ssm_scan_forward(block, ws, u_seq, lambda_seq, y_rank);
        for (size_t t = 0; t < L; t++) {
            gemv_f32(block->W_out.data, &y_rank[t * R], y_proj, (int)D, (int)R);
            for (size_t d = 0; d < D; d++) out[t * D + d] = in[t * D + d] + y_proj[d];
        }
    }
    free(z); free(u_seq); free(lambda_seq); free(y_rank); free(y_proj);
}

/* Internal GPU forward implementation - connects to cuda/mamba_block.cu */
#ifdef KMAMBA_BUILD_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* External declaration of cuda_block_forward from cuda/mamba_block.cu */
extern void cuda_block_forward(
    cublasHandle_t cublas,
    const float *d_W_in, const float *d_W_out, const float *d_A_log,
    const float *d_W_B, const float *d_W_C, const float *d_delta_proj,
    const float *d_theta, const float *d_lambda_proj,
    const float *d_x, float *d_y,
    float *d_u_raw, float *d_u, float *d_dt_raw, float *d_dt,
    float *d_B_exp, float *d_C_exp, float *d_dt_exp,
    float *d_h_store, float *d_y_scan, float *d_y_proj,
    float *d_lambda_raw, float *d_lambda,
    int L, int state, int dim, int R);

static int _mamba_block_forward_gpu(MambaBlock *block, float *output, const float *input, size_t batch_size) {
    if (!block || !output || !input || batch_size == 0) return -1;

    /* Lazy initialization of cuBLAS handle */
    static cublasHandle_t cublas_handle = NULL;
    static int cuda_init_done = 0;
    if (!cuda_init_done) {
        cudaError_t cuda_err = cudaFree(0);  /* Init CUDA runtime */
        if (cuda_err != cudaSuccess) return -1;
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) return -1;
        cuda_init_done = 1;
    }

    size_t L = block->config.seq_len;
    size_t D = block->config.dim;
    size_t N = block->config.state_size;
    size_t R = _mimo_R(&block->config);
    size_t NR = N * R;

    size_t bytes_L_D = L * D * sizeof(float);
    size_t bytes_L_R = L * R * sizeof(float);
    size_t bytes_L_NR = L * NR * sizeof(float);
    size_t bytes_L_N = L * N * sizeof(float);

    /* Device memory allocations */
    float *d_input = NULL, *d_output = NULL;
    float *d_W_in = NULL, *d_W_out = NULL, *d_A_log = NULL;
    float *d_W_B = NULL, *d_W_C = NULL, *d_delta_proj = NULL;
    float *d_theta = NULL, *d_lambda_proj = NULL;
    float *d_u_raw = NULL, *d_u = NULL, *d_dt_raw = NULL, *d_dt = NULL;
    float *d_B_exp = NULL, *d_C_exp = NULL, *d_dt_exp = NULL;
    float *d_h_store = NULL, *d_y_scan = NULL, *d_y_proj = NULL;
    float *d_lambda_raw = NULL, *d_lambda = NULL;

#define CUDA_CHECK_ALLOC(ptr, size) do { \
    cudaError_t err = cudaMalloc((void **)&(ptr), (size)); \
    if (err != cudaSuccess) { goto gpu_cleanup; } \
} while(0)

    CUDA_CHECK_ALLOC(d_input, bytes_L_D);
    CUDA_CHECK_ALLOC(d_output, bytes_L_D);
    CUDA_CHECK_ALLOC(d_W_in, R * D * sizeof(float));
    CUDA_CHECK_ALLOC(d_W_out, D * R * sizeof(float));
    CUDA_CHECK_ALLOC(d_A_log, N * sizeof(float));
    CUDA_CHECK_ALLOC(d_W_B, NR * D * sizeof(float));
    CUDA_CHECK_ALLOC(d_W_C, NR * D * sizeof(float));
    CUDA_CHECK_ALLOC(d_delta_proj, D * sizeof(float));
    CUDA_CHECK_ALLOC(d_theta, (N/2) * sizeof(float));
    CUDA_CHECK_ALLOC(d_lambda_proj, D * sizeof(float));
    CUDA_CHECK_ALLOC(d_u_raw, bytes_L_R);
    CUDA_CHECK_ALLOC(d_u, bytes_L_R);
    CUDA_CHECK_ALLOC(d_dt_raw, L * sizeof(float));
    CUDA_CHECK_ALLOC(d_dt, L * sizeof(float));
    CUDA_CHECK_ALLOC(d_B_exp, bytes_L_NR);
    CUDA_CHECK_ALLOC(d_C_exp, bytes_L_NR);
    CUDA_CHECK_ALLOC(d_dt_exp, bytes_L_N);  /* API compat, unused */
    CUDA_CHECK_ALLOC(d_h_store, bytes_L_N);
    CUDA_CHECK_ALLOC(d_y_scan, bytes_L_R);
    CUDA_CHECK_ALLOC(d_y_proj, bytes_L_D);
    CUDA_CHECK_ALLOC(d_lambda_raw, L * sizeof(float));
    CUDA_CHECK_ALLOC(d_lambda, L * sizeof(float));

#undef CUDA_CHECK_ALLOC

    /* Copy parameters to device (once for all batches) */
    cudaMemcpy(d_W_in, block->W_in.data, R * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_out, block->W_out.data, D * R * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_log, block->A_log.data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_B, block->W_B.data, NR * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_C, block->W_C.data, NR * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_proj, block->delta_proj.data, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, block->theta, (N/2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lambda_proj, block->lambda_proj.data, D * sizeof(float), cudaMemcpyHostToDevice);

    /* Process each batch */
    for (size_t b = 0; b < batch_size; b++) {
        const float *h_in = &input[b * L * D];
        float *h_out = &output[b * L * D];

        /* Copy input */
        cudaMemcpy(d_input, h_in, bytes_L_D, cudaMemcpyHostToDevice);

        /* Call GPU kernel */
        cuda_block_forward(
            cublas_handle,
            d_W_in, d_W_out, d_A_log, d_W_B, d_W_C, d_delta_proj,
            d_theta, d_lambda_proj,
            d_input, d_output,
            d_u_raw, d_u, d_dt_raw, d_dt,
            d_B_exp, d_C_exp, d_dt_exp,
            d_h_store, d_y_scan, d_y_proj,
            d_lambda_raw, d_lambda,
            L, N, D, R);

        /* Copy output */
        cudaMemcpy(h_out, d_output, bytes_L_D, cudaMemcpyDeviceToHost);
    }

gpu_cleanup:
    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_W_in); cudaFree(d_W_out); cudaFree(d_A_log);
    cudaFree(d_W_B); cudaFree(d_W_C);
    cudaFree(d_delta_proj); cudaFree(d_theta); cudaFree(d_lambda_proj);
    cudaFree(d_u_raw); cudaFree(d_u); cudaFree(d_dt_raw); cudaFree(d_dt);
    cudaFree(d_B_exp); cudaFree(d_C_exp); cudaFree(d_dt_exp);
    cudaFree(d_h_store); cudaFree(d_y_scan); cudaFree(d_y_proj);
    cudaFree(d_lambda_raw); cudaFree(d_lambda);

    return 0;  /* Success */
}

/* External declaration of gpu_block_backward from cuda/mamba_block.cu */
extern void gpu_block_backward(
    cublasHandle_t cublas,
    const float *d_W_in, const float *d_W_out, const float *d_A_log,
    const float *d_W_B, const float *d_W_C, const float *d_delta_proj,
    const float *d_theta, const float *d_lambda_proj,
    const float *d_x, const float *d_u_raw, const float *d_u,
    const float *d_dt_raw, const float *d_dt,
    const float *d_B_exp, const float *d_C_exp, const float *d_dt_exp,
    const float *d_h_store, const float *d_y_scan, const float *d_lambda,
    const float *d_dy,
    float *d_dW_in, float *d_dW_out, float *d_dA_log,
    float *d_dW_B, float *d_dW_C, float *d_ddelta_proj,
    float *d_g_theta, float *d_g_lambda_proj,
    float *d_dx,
    float *d_dy_scan, float *d_du, float *d_du_raw,
    float *d_ddt, float *d_ddt_raw,
    float *d_dB_scan, float *d_dC_scan, float *d_ddt_scan, float *d_dA_tmp,
    float *d_dlambda, float *d_dlambda_raw,
    int L, int state, int dim, int R);

static int _mamba_block_backward_gpu(MambaBlock *block, 
                                      const float *dY, const float *input,
                                      float *d_input, size_t batch_index,
                                      /* Forward activations saved on device */
                                      const float *d_u_raw, const float *d_u,
                                      const float *d_dt_raw, const float *d_dt,
                                      const float *d_B_exp, const float *d_C_exp,
                                      const float *d_h_store, const float *d_y_scan,
                                      const float *d_lambda) {
    (void)batch_index;
    if (!block || !dY || !input || !d_input) return -1;

    static cublasHandle_t cublas_handle = NULL;
    static int cuda_init_done = 0;
    if (!cuda_init_done) {
        cudaError_t cuda_err = cudaFree(0);
        if (cuda_err != cudaSuccess) return -1;
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) return -1;
        cuda_init_done = 1;
    }

    size_t L = block->config.seq_len;
    size_t D = block->config.dim;
    size_t N = block->config.state_size;
    size_t R = _mimo_R(&block->config);
    size_t NR = N * R;

    size_t bytes_L_D = L * D * sizeof(float);
    size_t bytes_L_R = L * R * sizeof(float);
    size_t bytes_L_NR = L * NR * sizeof(float);
    size_t bytes_L_N = L * N * sizeof(float);

    /* Device memory for gradients */
    float *d_dW_in = NULL, *d_dW_out = NULL, *d_dA_log = NULL;
    float *d_dW_B = NULL, *d_dW_C = NULL, *d_ddelta_proj = NULL;
    float *d_g_theta = NULL, *d_g_lambda_proj = NULL;
    float *d_dy = NULL, *d_dx = NULL;
    float *d_dy_scan = NULL, *d_du = NULL, *d_du_raw = NULL;
    float *d_ddt = NULL, *d_ddt_raw = NULL;
    float *d_dB_scan = NULL, *d_dC_scan = NULL, *d_ddt_scan = NULL, *d_dA_tmp = NULL;
    float *d_dlambda = NULL, *d_dlambda_raw = NULL;
    float *d_W_params = NULL; /* Temp for copying params */

#define CUDA_CHECK_ALLOC_BWD(ptr, size) do { \
    cudaError_t err = cudaMalloc((void **)&(ptr), (size)); \
    if (err != cudaSuccess) { goto gpu_bwd_cleanup; } \
} while(0)

    /* Allocate gradient buffers */
    CUDA_CHECK_ALLOC_BWD(d_dW_in, R * D * sizeof(float));
    CUDA_CHECK_ALLOC_BWD(d_dW_out, D * R * sizeof(float));
    CUDA_CHECK_ALLOC_BWD(d_dA_log, N * sizeof(float));
    CUDA_CHECK_ALLOC_BWD(d_dW_B, NR * D * sizeof(float));
    CUDA_CHECK_ALLOC_BWD(d_dW_C, NR * D * sizeof(float));
    CUDA_CHECK_ALLOC_BWD(d_ddelta_proj, D * sizeof(float));
    CUDA_CHECK_ALLOC_BWD(d_g_theta, (N/2) * sizeof(float));
    CUDA_CHECK_ALLOC_BWD(d_g_lambda_proj, D * sizeof(float));

    /* Activations/gradients */
    CUDA_CHECK_ALLOC_BWD(d_dy, bytes_L_D);
    CUDA_CHECK_ALLOC_BWD(d_dx, bytes_L_D);
    CUDA_CHECK_ALLOC_BWD(d_dy_scan, bytes_L_R);
    CUDA_CHECK_ALLOC_BWD(d_du, bytes_L_R);
    CUDA_CHECK_ALLOC_BWD(d_du_raw, bytes_L_R);
    CUDA_CHECK_ALLOC_BWD(d_ddt, L * sizeof(float));
    CUDA_CHECK_ALLOC_BWD(d_ddt_raw, L * sizeof(float));
    CUDA_CHECK_ALLOC_BWD(d_dB_scan, bytes_L_NR);
    CUDA_CHECK_ALLOC_BWD(d_dC_scan, bytes_L_NR);
    CUDA_CHECK_ALLOC_BWD(d_ddt_scan, bytes_L_N);
    CUDA_CHECK_ALLOC_BWD(d_dA_tmp, N * sizeof(float));
    CUDA_CHECK_ALLOC_BWD(d_dlambda, L * sizeof(float));
    CUDA_CHECK_ALLOC_BWD(d_dlambda_raw, L * sizeof(float));

    /* Copy W_params to device */
    CUDA_CHECK_ALLOC_BWD(d_W_params, (R*D + D*R + N + NR*D*2 + D + N/2 + D) * sizeof(float));

#undef CUDA_CHECK_ALLOC_BWD

    /* Zero gradients */
    cudaMemset(d_dW_in, 0, R * D * sizeof(float));
    cudaMemset(d_dW_out, 0, D * R * sizeof(float));
    cudaMemset(d_dA_log, 0, N * sizeof(float));
    cudaMemset(d_dW_B, 0, NR * D * sizeof(float));
    cudaMemset(d_dW_C, 0, NR * D * sizeof(float));
    cudaMemset(d_ddelta_proj, 0, D * sizeof(float));
    cudaMemset(d_g_theta, 0, (N/2) * sizeof(float));
    cudaMemset(d_g_lambda_proj, 0, D * sizeof(float));

    /* Copy dY from host */
    cudaMemcpy(d_dy, dY, bytes_L_D, cudaMemcpyHostToDevice);

    /* Copy params to device */
    float *d_W_in = d_W_params;
    float *d_W_out = d_W_in + R * D;
    float *d_A_log = d_W_out + D * R;
    float *d_W_B = d_A_log + N;
    float *d_W_C = d_W_B + NR * D;
    float *d_delta_proj = d_W_C + NR * D;
    float *d_theta = d_delta_proj + D;
    float *d_lambda_proj = d_theta + (N/2);

    cudaMemcpy(d_W_in, block->W_in.data, R * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_out, block->W_out.data, D * R * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_log, block->A_log.data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_B, block->W_B.data, NR * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_C, block->W_C.data, NR * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_proj, block->delta_proj.data, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, block->theta, (N/2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lambda_proj, block->lambda_proj.data, D * sizeof(float), cudaMemcpyHostToDevice);

    /* Call GPU backward - d_dt_exp parameter not used by kernel but kept for API compat */
    gpu_block_backward(
        cublas_handle,
        d_W_in, d_W_out, d_A_log, d_W_B, d_W_C, d_delta_proj,
        d_theta, d_lambda_proj,
        d_input, d_u_raw, d_u, d_dt_raw, d_dt,
        d_B_exp, d_C_exp, NULL,  /* d_dt_exp not used */
        d_h_store, d_y_scan, d_lambda,
        d_dy,
        d_dW_in, d_dW_out, d_dA_log, d_dW_B, d_dW_C, d_ddelta_proj,
        d_g_theta, d_g_lambda_proj,
        d_dx,
        d_dy_scan, d_du, d_du_raw, d_ddt, d_ddt_raw,
        d_dB_scan, d_dC_scan, d_ddt_scan, d_dA_tmp,
        d_dlambda, d_dlambda_raw,
        (int)L, (int)N, (int)D, (int)R);

    /* Copy gradients back to host opt_state */
    if (block->opt_state) {
        MBOptimState *s = block->opt_state;
        cudaMemcpy(s->g_W_in, d_dW_in, R * D * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(s->g_W_out, d_dW_out, D * R * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(s->g_A_log, d_dA_log, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(s->g_W_B, d_dW_B, NR * D * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(s->g_W_C, d_dW_C, NR * D * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(s->g_delta_proj, d_ddelta_proj, D * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(s->g_theta, d_g_theta, (N/2) * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(s->g_lambda_proj, d_g_lambda_proj, D * sizeof(float), cudaMemcpyDeviceToHost);
    }

    /* Copy dX back to host */
    cudaMemcpy(d_input, d_dx, bytes_L_D, cudaMemcpyDeviceToHost);

gpu_bwd_cleanup:
    cudaFree(d_dW_in); cudaFree(d_dW_out); cudaFree(d_dA_log);
    cudaFree(d_dW_B); cudaFree(d_dW_C); cudaFree(d_ddelta_proj);
    cudaFree(d_g_theta); cudaFree(d_g_lambda_proj);
    cudaFree(d_dy); cudaFree(d_dx);
    cudaFree(d_dy_scan); cudaFree(d_du); cudaFree(d_du_raw);
    cudaFree(d_ddt); cudaFree(d_ddt_raw);
    cudaFree(d_dB_scan); cudaFree(d_dC_scan); cudaFree(d_ddt_scan); cudaFree(d_dA_tmp);
    cudaFree(d_dlambda); cudaFree(d_dlambda_raw);
    cudaFree(d_W_params);

    return 0;
}
#endif

void mamba_block_forward(MambaBlock *block, float *output, const float *input, size_t batch_size) {
    /* Automatic GPU dispatch if available */
    KMAMBA_AUTO_BACKEND();
    
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
