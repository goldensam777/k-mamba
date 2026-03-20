/*
 * mamba_block.c — MambaBlock SSM architecture with MUONCLIP optimizer
 *
 * Core Mamba implementation: diagonal A, shared B/C vectors,
 * input-dependent delta, W_in/W_out projections, scan1d/scan2d ASM kernels.
 *
 * Part of k-mamba — uses optimatrix for compute kernels.
 */

#include "kmamba.h"
#include "optimatrix.h"
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
 * Scalar activations (internal — vectorized versions in ASM)
 * ============================================================================ */

static float scalar_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + expf(x));
}

static float scalar_sigmoid(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/* ============================================================================
 * Forward storage for training (per timestep)
 * ============================================================================ */
typedef struct {
    float *x;         /* seq_len x state_size */
    float *A_diag;    /* seq_len x state_size */
    float *B_bar;     /* seq_len x state_size */
    float *u_seq;     /* seq_len x state_size (controller vectors per timestep) */
    float *h_rot;     /* seq_len x state_size  R(θ)·h_{t-1} at each step (for dA grad) */
} ForwardStore;

/* ============================================================================
 * Global registry mapping MambaBlock -> MBOptimState
 * ============================================================================ */
static MBOptimState *g_opt_states[256];
static MambaBlock   *g_opt_blocks[256];
static size_t        g_opt_n = 0;

/* forward declarations */
static void          _mamba_free_opt_for(MambaBlock *block);
static MBOptimState* _find_opt(MambaBlock *block);
static int           matrix_init_owned(MBMatrix *dst, size_t rows, size_t cols);
static void          project_controller(const MambaBlock *block, const float *x_t,
                                        float *z_buf, float *u_out);
static float         project_delta_value(const MambaBlock *block, const float *x_t,
                                         float *tmp_delta, size_t position,
                                         size_t total_positions);
static void          transpose_row_major(const float *src, float *dst,
                                         size_t rows, size_t cols);

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

MBMatrix* mb_matrix_create(size_t rows, size_t cols) {
    MBMatrix *m = (MBMatrix *)malloc(sizeof(MBMatrix));
    if (!m) return NULL;

    m->rows = rows;
    m->cols = cols;
    m->data = (float *)calloc(rows * cols, sizeof(float));

    if (!m->data) {
        free(m);
        return NULL;
    }

    return m;
}

static int matrix_init_owned(MBMatrix *dst, size_t rows, size_t cols) {
    MBMatrix *m;

    if (!dst) return -1;
    m = mb_matrix_create(rows, cols);
    if (!m) {
        memset(dst, 0, sizeof(*dst));
        return -1;
    }
    *dst = *m;
    free(m);
    return 0;
}

void mb_matrix_free(MBMatrix *m) {
    if (!m) return;
    if (m->data) free(m->data);
    free(m);
}

void mb_matrix_zero(MBMatrix *m) {
    if (!m || !m->data) return;
    memset(m->data, 0, m->rows * m->cols * sizeof(float));
}

void mb_matrix_copy(MBMatrix *dst, const MBMatrix *src) {
    if (!dst || !src || !dst->data || !src->data) return;
    if (dst->rows != src->rows || dst->cols != src->cols) return;
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(float));
}

void mb_matrix_print(const MBMatrix *m) {
    if (!m || !m->data) return;

    printf("Matrix (%zu x %zu):\n", m->rows, m->cols);
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            printf("%10.6f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
}

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

void mb_matrix_vec_mult(float *out, const MBMatrix *m, const float *v) {
    if (!out || !m || !v || !m->data) return;
    gemv_avx2(m->data, (float *)v, out, (long)m->rows, (long)m->cols);
}

void mb_vec_add(float *y, const float *x, size_t n) {
    if (!y || !x) return;
    for (size_t i = 0; i < n; i++) {
        y[i] += x[i];
    }
}

void mb_vec_scale(float *v, float alpha, size_t n) {
    if (!v) return;
    for (size_t i = 0; i < n; i++) {
        v[i] *= alpha;
    }
}

static void project_controller(const MambaBlock *block, const float *x_t,
                                float *z_buf, float *u_out) {
    if (!block || !x_t || !z_buf || !u_out) return;
    mb_matrix_vec_mult(z_buf, &block->W_in, x_t);
    silu_f32(z_buf, u_out, (long)block->config.state_size);
}

static float project_delta_value(const MambaBlock *block, const float *x_t,
                                  float *tmp_delta, size_t position,
                                  size_t total_positions) {
    float dval;

    if (!block || !x_t || !tmp_delta || total_positions == 0) return 0.0f;

    if (block->delta_proj.rows > 0) {
        mb_matrix_vec_mult(tmp_delta, &block->delta_proj, x_t);
        dval = scalar_softplus(tmp_delta[0]);
        if (dval < block->config.dt_min) dval = block->config.dt_min;
        if (dval > block->config.dt_max) dval = block->config.dt_max;
        return dval;
    }

    return block->config.dt_scale *
           ((float)position / (float)total_positions + 1.0f);
}

static void transpose_row_major(const float *src, float *dst,
                                 size_t rows, size_t cols) {
    if (!src || !dst) return;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/* ============================================================================
 * Discretization Functions
 * ============================================================================ */

void mb_discretize_A(MBMatrix *A_bar, const MBMatrix *A, float dt) {
    if (!A_bar || !A) return;

    for (size_t i = 0; i < A_bar->rows && i < A->rows; i++) {
        float a_ii = A->data[i * A->cols + i];
        A_bar->data[i * A_bar->cols + i] = expf(dt * a_ii);
    }
}

void mb_discretize_B(float *B_bar, const MBMatrix *A, const float *B,
                     float dt, size_t state_size) {
    if (!B_bar || !A || !B) return;
    (void)A;

    for (size_t i = 0; i < state_size; i++) {
        B_bar[i] = dt * B[i];
    }
}

/* ============================================================================
 * Selective Scan - Core Mamba Operation
 * ============================================================================ */

void mb_selective_scan(float *output, float *state,
                       const float *input, const float *delta,
                       const MBMatrix *A_bar, const float *B_bar,
                       const MBMatrix *C, float D,
                       size_t seq_len, size_t state_size) {

    if (!output || !state || !input || !delta || !A_bar || !B_bar || !C) {
        return;
    }
    (void)C; (void)D;

    memset(state, 0, state_size * sizeof(float));

    float *temp_state = (float *)malloc(state_size * sizeof(float));
    float *A_diag_t = (float *)malloc(state_size * sizeof(float));
    float *B_bar_t = (float *)malloc(state_size * sizeof(float));
    if (!temp_state || !A_diag_t || !B_bar_t) {
        free(temp_state); free(A_diag_t); free(B_bar_t);
        return;
    }

    for (size_t t = 0; t < seq_len; t++) {
        const float *u_t = &input[t * state_size];
        float dt_t = delta[t];

        for (size_t i = 0; i < state_size; i++) {
            float a_val = A_bar->data[i * state_size + i];
            if (a_val > -1e-5f) a_val = -1e-5f;   /* clamp A ≤ -1e-5 */
            float a_diag = expf(dt_t * a_val);
            A_diag_t[i] = a_diag;
            if (fabsf(a_val) < 1e-8f) {
                B_bar_t[i] = dt_t * B_bar[i];
            } else {
                B_bar_t[i] = (a_diag - 1.0f) / a_val * B_bar[i];
            }
        }

        hadamard_avx2(A_diag_t, state, temp_state, (long)state_size);
        hadamard_avx2(B_bar_t, (float *)u_t, state, (long)state_size);
        mb_vec_add(state, temp_state, state_size);

        memcpy(&output[t * state_size], state, state_size * sizeof(float));
    }

    free(A_diag_t); free(B_bar_t); free(temp_state);
}

/* ============================================================================
 * Mamba Block Operations
 * ============================================================================ */

MambaBlock* mamba_block_create(const MBConfig *config) {
    if (!config) return NULL;

    MambaBlock *block = (MambaBlock *)malloc(sizeof(MambaBlock));
    if (!block) return NULL;
    memset(block, 0, sizeof(*block));

    block->config = *config;

    if (matrix_init_owned(&block->W_in, config->state_size, config->dim) != 0 ||
        matrix_init_owned(&block->W_out, config->dim, config->state_size) != 0 ||
        matrix_init_owned(&block->A_log, config->state_size, 1) != 0 ||
        matrix_init_owned(&block->W_B, config->state_size, config->dim) != 0 ||
        matrix_init_owned(&block->W_C, config->state_size, config->dim) != 0 ||
        matrix_init_owned(&block->delta_proj, 1, config->dim) != 0) {
        mamba_block_free(block);
        return NULL;
    }
    /* BCNorm biases — init séparé pour clarté */
    block->b_B = (float *)calloc(config->state_size, sizeof(float));
    block->b_C = (float *)calloc(config->state_size, sizeof(float));
    if (!block->b_B || !block->b_C) {
        mamba_block_free(block);
        return NULL;
    }

    /* Complex SSM rotation angles theta [state_size/2] */
    size_t theta_size = config->state_size / 2;
    if (theta_size == 0) theta_size = 1;  /* guard for tiny state_size */
    block->theta = (float *)calloc(theta_size, sizeof(float));
    if (!block->theta) {
        mamba_block_free(block);
        return NULL;
    }

    block->hidden = (float *)calloc(config->state_size, sizeof(float));
    block->delta  = (float *)calloc(config->seq_len, sizeof(float));

    size_t LD = config->seq_len * config->state_size;
    block->scan_B     = (float *)malloc(LD * sizeof(float));
    block->scan_C     = (float *)malloc(LD * sizeof(float));
    block->scan_delta = (float *)malloc(LD * sizeof(float));
    block->scan_h     = (float *)calloc(config->state_size, sizeof(float));

    if (!block->W_in.data || !block->W_out.data || !block->A_log.data ||
        !block->W_B.data || !block->W_C.data || !block->delta_proj.data ||
        !block->hidden || !block->delta ||
        !block->scan_B || !block->scan_C || !block->scan_delta || !block->scan_h) {
        mamba_block_free(block);
        return NULL;
    }

    /* Allocate ConvND resources if enabled */
    if (config->use_convnd && config->convnd_K > 0 && config->convnd_ndims > 0) {
        long kernel_size = config->convnd_ndims * config->convnd_K * config->dim;
        block->convnd_kernel = (float *)calloc(kernel_size, sizeof(float));
        block->convnd_bias = (float *)calloc(config->dim, sizeof(float));
        
        ConvNDParams p = {
            .dims = NULL,  /* Will be set per forward call */
            .ndims = config->convnd_ndims,
            .D = config->dim,
            .K = config->convnd_K
        };
        block->convnd_ws = convnd_workspace_create(&p);
        
        if (!block->convnd_kernel || !block->convnd_bias || !block->convnd_ws) {
            mamba_block_free(block);
            return NULL;
        }
    }

    return block;
}

void mamba_block_free(MambaBlock *block) {
    if (!block) return;

    if (block->W_in.data) free(block->W_in.data);
    if (block->W_out.data) free(block->W_out.data);
    if (block->A_log.data) free(block->A_log.data);
    if (block->W_B.data) free(block->W_B.data);
    if (block->W_C.data) free(block->W_C.data);
    if (block->b_B) free(block->b_B);
    if (block->b_C) free(block->b_C);
    if (block->theta) free(block->theta);
    if (block->delta_proj.data) free(block->delta_proj.data);
    if (block->hidden) free(block->hidden);
    if (block->delta) free(block->delta);
    if (block->scan_B)     free(block->scan_B);
    if (block->scan_C)     free(block->scan_C);
    if (block->scan_delta) free(block->scan_delta);
    if (block->scan_h)     free(block->scan_h);
    
    /* Free ConvND resources */
    if (block->convnd_kernel) free(block->convnd_kernel);
    if (block->convnd_bias)   free(block->convnd_bias);
    if (block->convnd_ws)     convnd_workspace_free(block->convnd_ws);

    free(block);
}

void mamba_block_init(MambaBlock *block) {
    if (!block) return;

    /* A_log stocke directement A (valeur négative).
     * A = -1 pour toutes les dimensions → decay = exp(-Δ) ∈ (0.37, 0.999).
     * La stabilité est garantie par le clamp A ≤ -1e-5 dans le forward. */
    for (size_t i = 0; i < block->config.state_size; i++) {
        block->A_log.data[i] = -1.0f;
    }

    /* Xavier uniform init for W_in (state_size x dim) */
    {
        float fan_in  = (float)block->W_in.cols;
        float fan_out = (float)block->W_in.rows;
        float scale   = sqrtf(6.0f / (fan_in + fan_out));
        for (size_t i = 0; i < block->W_in.rows * block->W_in.cols; i++) {
            block->W_in.data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
    /* Xavier uniform init for W_out (dim x state_size) */
    {
        float fan_in  = (float)block->W_out.cols;
        float fan_out = (float)block->W_out.rows;
        float scale   = sqrtf(6.0f / (fan_in + fan_out));
        for (size_t i = 0; i < block->W_out.rows * block->W_out.cols; i++) {
            block->W_out.data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
    /* Xavier uniform init for W_B (state_size x dim) */
    {
        float fan_in  = (float)block->W_B.cols;
        float fan_out = (float)block->W_B.rows;
        float scale   = sqrtf(6.0f / (fan_in + fan_out));
        for (size_t i = 0; i < block->W_B.rows * block->W_B.cols; i++) {
            block->W_B.data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
    /* Xavier uniform init for W_C (state_size x dim) */
    {
        float fan_in  = (float)block->W_C.cols;
        float fan_out = (float)block->W_C.rows;
        float scale   = sqrtf(6.0f / (fan_in + fan_out));
        for (size_t i = 0; i < block->W_C.rows * block->W_C.cols; i++) {
            block->W_C.data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
    /* BCNorm biases — initialisés à zéro (comportement neutre au départ) */
    memset(block->b_B, 0, block->config.state_size * sizeof(float));
    memset(block->b_C, 0, block->config.state_size * sizeof(float));

    /* Complex SSM rotation angles — small init ~2π/state_size */
    {
        size_t theta_size = block->config.state_size / 2;
        if (theta_size == 0) theta_size = 1;
        float base_angle = 2.0f * 3.14159265358979323846f / (float)block->config.state_size;
        for (size_t i = 0; i < theta_size; i++) {
            block->theta[i] = ((float)rand() / RAND_MAX) * base_angle;
        }
    }

    /* Small uniform init for delta_proj (1 x dim) */
    for (size_t i = 0; i < block->delta_proj.rows * block->delta_proj.cols; i++) {
        block->delta_proj.data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
}

void mb_compute_delta(float *delta_out, const MambaBlock *block,
                      const float *delta_in, size_t seq_len) {
    if (!delta_out || !block || !delta_in) return;

    for (size_t i = 0; i < seq_len; i++) {
        float delta_val = delta_in[i];
        delta_val = scalar_softplus(delta_val);

        if (delta_val < block->config.dt_min) {
            delta_val = block->config.dt_min;
        }
        if (delta_val > block->config.dt_max) {
            delta_val = block->config.dt_max;
        }

        delta_out[i] = delta_val;
    }
}

/* ============================================================================
 * Optimizer — MUONCLIP
 * ============================================================================ */

void mamba_attach_optimizer(MambaBlock *block, OptimizerType type, const MBOptimConfig *optconf) {
    if (!block) return;
    MBOptimState *s = (MBOptimState *)malloc(sizeof(MBOptimState));
    size_t dim = block->config.dim;
    size_t state = block->config.state_size;
    size_t size_in = state * dim;
    size_t size_out = dim * state;
    memset(s, 0, sizeof(MBOptimState));
    
    s->type = type;

    size_t size_bc = state * dim;
    s->g_W_in = (float *)calloc(size_in, sizeof(float));
    s->g_W_out = (float *)calloc(size_out, sizeof(float));
    s->g_A_log = (float *)calloc(state, sizeof(float));
    s->g_W_B = (float *)calloc(size_bc, sizeof(float));
    s->g_W_C = (float *)calloc(size_bc, sizeof(float));
    s->g_b_B = (float *)calloc(state,   sizeof(float));
    s->g_b_C = (float *)calloc(state,   sizeof(float));
    s->g_delta_proj = (float *)calloc(dim, sizeof(float));
    size_t theta_size = state / 2; if (theta_size == 0) theta_size = 1;
    s->g_theta      = (float *)calloc(theta_size, sizeof(float));

    /* Allocate first moments for all momentum-based optimizers */
    if (type == OPTIMIZER_ADAM_CLIP || type == OPTIMIZER_ADAMW ||
        type == OPTIMIZER_MUON      || type == OPTIMIZER_SGD) {
        s->m_W_in       = (float *)calloc(size_in,  sizeof(float));
        s->m_W_out      = (float *)calloc(size_out, sizeof(float));
        s->m_A_log      = (float *)calloc(state,    sizeof(float));
        s->m_W_B        = (float *)calloc(size_bc,  sizeof(float));
        s->m_W_C        = (float *)calloc(size_bc,  sizeof(float));
        s->m_b_B        = (float *)calloc(state,    sizeof(float));
        s->m_b_C        = (float *)calloc(state,    sizeof(float));
        s->m_delta_proj = (float *)calloc(dim,      sizeof(float));
        s->m_theta      = (float *)calloc(theta_size, sizeof(float));
    }
    /* Allocate second moments only for Adam-based optimizers */
    if (type == OPTIMIZER_ADAM_CLIP || type == OPTIMIZER_ADAMW) {
        s->v_W_in       = (float *)calloc(size_in,  sizeof(float));
        s->v_W_out      = (float *)calloc(size_out, sizeof(float));
        s->v_A_log      = (float *)calloc(state,    sizeof(float));
        s->v_W_B        = (float *)calloc(size_bc,  sizeof(float));
        s->v_W_C        = (float *)calloc(size_bc,  sizeof(float));
        s->v_b_B        = (float *)calloc(state,    sizeof(float));
        s->v_b_C        = (float *)calloc(state,    sizeof(float));
        s->v_delta_proj = (float *)calloc(dim,      sizeof(float));
        s->v_theta      = (float *)calloc(theta_size, sizeof(float));
    }

    s->step = 0;
    if (g_opt_n < 256) { g_opt_blocks[g_opt_n] = block; g_opt_states[g_opt_n] = s; g_opt_n++; }
    else free(s);
    (void)optconf;
}

void mamba_free_optimizer(MambaBlock *block) {
    _mamba_free_opt_for(block);
}

static void _mamba_free_opt_for(MambaBlock *block) {
    for (size_t i = 0; i < g_opt_n; i++) {
        if (g_opt_blocks[i] == block) {
            MBOptimState *s = g_opt_states[i];
            if (!s) return;
            free(s->g_W_in); free(s->g_W_out); free(s->g_A_log);
            free(s->g_W_B); free(s->g_W_C); free(s->g_b_B); free(s->g_b_C);
            free(s->g_delta_proj);
            if (s->g_theta) free(s->g_theta);

            /* Free moments only if allocated */
            if (s->m_W_in) { free(s->m_W_in); free(s->v_W_in); }
            if (s->m_W_out) { free(s->m_W_out); free(s->v_W_out); }
            if (s->m_A_log) { free(s->m_A_log); free(s->v_A_log); }
            if (s->m_W_B) { free(s->m_W_B); if (s->v_W_B) free(s->v_W_B); }
            if (s->m_W_C) { free(s->m_W_C); if (s->v_W_C) free(s->v_W_C); }
            if (s->m_b_B) { free(s->m_b_B); if (s->v_b_B) free(s->v_b_B); }
            if (s->m_b_C) { free(s->m_b_C); if (s->v_b_C) free(s->v_b_C); }
            if (s->m_delta_proj) { free(s->m_delta_proj); free(s->v_delta_proj); }
            if (s->m_theta) { free(s->m_theta); if (s->v_theta) free(s->v_theta); }
            
            free(s);
            for (size_t j = i; j + 1 < g_opt_n; j++) { g_opt_blocks[j] = g_opt_blocks[j+1]; g_opt_states[j] = g_opt_states[j+1]; }
            g_opt_n--;
            return;
        }
    }
}

void mamba_zero_grads(MambaBlock *block) {
    for (size_t i = 0; i < g_opt_n; i++) {
        if (g_opt_blocks[i] == block) {
            MBOptimState *s = g_opt_states[i];
            size_t dim = block->config.dim; size_t state = block->config.state_size;
            size_t size_in = state * dim; size_t size_out = dim * state;
            memset(s->g_W_in, 0, size_in * sizeof(float)); memset(s->g_W_out, 0, size_out * sizeof(float));
            memset(s->g_A_log, 0, state * sizeof(float));
            memset(s->g_W_B, 0, size_in * sizeof(float)); memset(s->g_W_C, 0, size_in * sizeof(float));
            memset(s->g_b_B, 0, state * sizeof(float));   memset(s->g_b_C, 0, state * sizeof(float));
            memset(s->g_delta_proj, 0, dim * sizeof(float));
            { size_t ts = state / 2; if (ts == 0) ts = 1; memset(s->g_theta, 0, ts * sizeof(float)); }
            return;
        }
    }
}

static MBOptimState* _find_opt(MambaBlock *block) {
    for (size_t i = 0; i < g_opt_n; i++) if (g_opt_blocks[i] == block) return g_opt_states[i];
    return NULL;
}

/* ============================================================================
 * Optimizer Implementations
 * ============================================================================ */

static void adam_clip_update(MambaBlock *block, MBOptimState *s, const MBOptimConfig *conf) {
    size_t dim = block->config.dim; size_t state = block->config.state_size;
    size_t size_in = state * dim; size_t size_out = dim * state;

#ifdef KMAMBA_BUILD_CUDA
    adamw_update_cuda(block->W_in.data,       s->g_W_in,       s->m_W_in,       s->v_W_in,       size_in,   conf, s->step);
    adamw_update_cuda(block->W_out.data,      s->g_W_out,      s->m_W_out,      s->v_W_out,      size_out,  conf, s->step);
    adamw_update_cuda(block->A_log.data,      s->g_A_log,      s->m_A_log,      s->v_A_log,      state,     conf, s->step);
    adamw_update_cuda(block->W_B.data,        s->g_W_B,        s->m_W_B,        s->v_W_B,        size_in,   conf, s->step);
    adamw_update_cuda(block->W_C.data,        s->g_W_C,        s->m_W_C,        s->v_W_C,        size_in,   conf, s->step);
    adamw_update_cuda(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, s->v_delta_proj, dim,       conf, s->step);
#else
    float lr = conf->lr; float mu = conf->mu; float beta2 = conf->beta2;
    float eps = conf->eps; float clip = conf->clip_norm; float wd = conf->weight_decay;

#define ADAM_CLIP_UPDATE(param, grad, m, v, N) do { \
    if (clip > 0.0f) gradient_clip_inplace(grad, (N), clip); \
    for (size_t _i=0; _i < (N); _i++) { float g = grad[_i] + wd * param[_i]; \
        m[_i] = mu * m[_i] + (1.0f - mu) * g; \
        v[_i] = beta2 * v[_i] + (1.0f - beta2) * (g * g); \
        float m_hat = m[_i] / (1.0f - powf(mu, (float)s->step)); \
        float v_hat = v[_i] / (1.0f - powf(beta2, (float)s->step)); \
        param[_i] -= lr * m_hat / (sqrtf(v_hat) + eps); } \
    } while (0)

    ADAM_CLIP_UPDATE(block->W_in.data, s->g_W_in, s->m_W_in, s->v_W_in, size_in);
    ADAM_CLIP_UPDATE(block->W_out.data, s->g_W_out, s->m_W_out, s->v_W_out, size_out);
    ADAM_CLIP_UPDATE(block->A_log.data, s->g_A_log, s->m_A_log, s->v_A_log, state);
    ADAM_CLIP_UPDATE(block->W_B.data, s->g_W_B, s->m_W_B, s->v_W_B, size_in);
    ADAM_CLIP_UPDATE(block->W_C.data, s->g_W_C, s->m_W_C, s->v_W_C, size_in);
    ADAM_CLIP_UPDATE(block->b_B, s->g_b_B, s->m_b_B, s->v_b_B, state);
    ADAM_CLIP_UPDATE(block->b_C, s->g_b_C, s->m_b_C, s->v_b_C, state);
    ADAM_CLIP_UPDATE(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, s->v_delta_proj, dim);
    { size_t ts = state/2; if (ts==0) ts=1;
      ADAM_CLIP_UPDATE(block->theta, s->g_theta, s->m_theta, s->v_theta, ts); }

#undef ADAM_CLIP_UPDATE
#endif
}

static void adamw_update(MambaBlock *block, MBOptimState *s, const MBOptimConfig *conf) {
    size_t dim = block->config.dim; size_t state = block->config.state_size;
    size_t size_in = state * dim; size_t size_out = dim * state;
    float lr = conf->lr; float mu = conf->mu; float beta2 = conf->beta2; 
    float eps = conf->eps; float wd = conf->weight_decay;

#define ADAMW_UPDATE(param, grad, m, v, N) do { \
    for (size_t _i=0; _i < (N); _i++) { float g = grad[_i]; \
        m[_i] = mu * m[_i] + (1.0f - mu) * g; \
        v[_i] = beta2 * v[_i] + (1.0f - beta2) * (g * g); \
        float m_hat = m[_i] / (1.0f - powf(mu, (float)s->step)); \
        float v_hat = v[_i] / (1.0f - powf(beta2, (float)s->step)); \
        param[_i] = param[_i] * (1.0f - lr * wd) - lr * m_hat / (sqrtf(v_hat) + eps); } \
    } while (0)

    ADAMW_UPDATE(block->W_in.data, s->g_W_in, s->m_W_in, s->v_W_in, size_in);
    ADAMW_UPDATE(block->W_out.data, s->g_W_out, s->m_W_out, s->v_W_out, size_out);
    ADAMW_UPDATE(block->A_log.data, s->g_A_log, s->m_A_log, s->v_A_log, state);
    ADAMW_UPDATE(block->W_B.data, s->g_W_B, s->m_W_B, s->v_W_B, size_in);
    ADAMW_UPDATE(block->W_C.data, s->g_W_C, s->m_W_C, s->v_W_C, size_in);
    ADAMW_UPDATE(block->b_B, s->g_b_B, s->m_b_B, s->v_b_B, state);
    ADAMW_UPDATE(block->b_C, s->g_b_C, s->m_b_C, s->v_b_C, state);
    ADAMW_UPDATE(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, s->v_delta_proj, dim);
    { size_t ts = state/2; if (ts==0) ts=1;
      ADAMW_UPDATE(block->theta, s->g_theta, s->m_theta, s->v_theta, ts); }

#undef ADAMW_UPDATE
}

static void sgd_update(MambaBlock *block, MBOptimState *s, const MBOptimConfig *conf) {
    size_t dim = block->config.dim; size_t state = block->config.state_size;
    size_t size_in = state * dim; size_t size_out = dim * state;
    float lr = conf->lr; float mu = conf->mu; float wd = conf->weight_decay;

#define SGD_UPDATE(param, grad, m, N) do { \
    for (size_t _i=0; _i < (N); _i++) { float g = grad[_i] + wd * param[_i]; \
        m[_i] = mu * m[_i] + g; \
        param[_i] -= lr * m[_i]; } \
    } while (0)

    SGD_UPDATE(block->W_in.data, s->g_W_in, s->m_W_in, size_in);
    SGD_UPDATE(block->W_out.data, s->g_W_out, s->m_W_out, size_out);
    SGD_UPDATE(block->A_log.data, s->g_A_log, s->m_A_log, state);
    SGD_UPDATE(block->W_B.data, s->g_W_B, s->m_W_B, size_in);
    SGD_UPDATE(block->W_C.data, s->g_W_C, s->m_W_C, size_in);
    SGD_UPDATE(block->b_B, s->g_b_B, s->m_b_B, state);
    SGD_UPDATE(block->b_C, s->g_b_C, s->m_b_C, state);
    SGD_UPDATE(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, dim);
    { size_t ts = state/2; if (ts==0) ts=1;
      SGD_UPDATE(block->theta, s->g_theta, s->m_theta, ts); }

#undef SGD_UPDATE
}

/* MUON : matrices (W_in, W_out) — Newton-Schulz + momentum + clipping */
static void muon_update(MambaBlock *block, MBOptimState *s, const MBOptimConfig *conf) {
    size_t dim = block->config.dim; size_t state = block->config.state_size;

#ifdef KMAMBA_BUILD_CUDA
    muon_update_mat_cuda(block->W_in.data,  s->g_W_in,  s->m_W_in,  state, dim,   conf);
    muon_update_mat_cuda(block->W_out.data, s->g_W_out, s->m_W_out, dim,   state, conf);
    muon_update_vec_cuda(block->A_log.data,      s->g_A_log,      s->m_A_log,      state, conf);
    muon_update_mat_cuda(block->W_B.data,   s->g_W_B,   s->m_W_B,   state, dim,   conf);
    muon_update_mat_cuda(block->W_C.data,   s->g_W_C,   s->m_W_C,   state, dim,   conf);
    muon_update_vec_cuda(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, dim,   conf);
#else
    float lr = conf->lr; float mu = conf->mu; float wd = conf->weight_decay;
    float clip = conf->clip_norm;

#define MUON_UPDATE_MAT(param, grad, m, rows, cols) do { \
    size_t _N = (rows) * (cols); \
    newton_schulz5_inplace(grad, rows, cols, 5); \
    for (size_t _i = 0; _i < _N; _i++) { \
        float _g = grad[_i] + wd * param[_i]; \
        m[_i] = mu * m[_i] + (1.0f - mu) * _g; \
    } \
    if (clip > 0.0f) gradient_clip_inplace(m, _N, clip); \
    for (size_t _i = 0; _i < _N; _i++) param[_i] -= lr * m[_i]; \
    } while (0)

#define MUON_UPDATE_VEC(param, grad, m, N) do { \
    for (size_t _i = 0; _i < (N); _i++) { \
        float _g = grad[_i] + wd * param[_i]; \
        m[_i] = mu * m[_i] + (1.0f - mu) * _g; \
    } \
    if (clip > 0.0f) gradient_clip_inplace(m, (N), clip); \
    for (size_t _i = 0; _i < (N); _i++) param[_i] -= lr * m[_i]; \
    } while (0)

    MUON_UPDATE_MAT(block->W_in.data,  s->g_W_in,  s->m_W_in,  state, dim);
    MUON_UPDATE_MAT(block->W_out.data, s->g_W_out, s->m_W_out, dim,   state);
    MUON_UPDATE_VEC(block->A_log.data,       s->g_A_log,       s->m_A_log,       state);
    MUON_UPDATE_MAT(block->W_B.data,   s->g_W_B,   s->m_W_B,   state, dim);
    MUON_UPDATE_MAT(block->W_C.data,   s->g_W_C,   s->m_W_C,   state, dim);
    MUON_UPDATE_VEC(block->b_B,        s->g_b_B,   s->m_b_B,   state);
    MUON_UPDATE_VEC(block->b_C,        s->g_b_C,   s->m_b_C,   state);
    MUON_UPDATE_VEC(block->delta_proj.data,  s->g_delta_proj,  s->m_delta_proj,  dim);
    { size_t ts = state/2; if (ts==0) ts=1;
      MUON_UPDATE_VEC(block->theta, s->g_theta, s->m_theta, ts); }

#undef MUON_UPDATE_MAT
#undef MUON_UPDATE_VEC
#endif
}

void mamba_optimizer_step(MambaBlock *block, const MBOptimConfig *conf) {
    MBOptimState *s = _find_opt(block);
    if (!s) return;
    s->step += 1;

    switch (s->type) {
        case OPTIMIZER_ADAM_CLIP:
            adam_clip_update(block, s, conf);
            break;
        case OPTIMIZER_ADAMW:
            adamw_update(block, s, conf);
            break;
        case OPTIMIZER_SGD:
            sgd_update(block, s, conf);
            break;
        case OPTIMIZER_MUON:
            muon_update(block, s, conf);
            break;
        default:
            /* Default to ADAM_CLIP for safety */
            adam_clip_update(block, s, conf);
            break;
    }
}

/* ============================================================================
 * Forward scan with storage for backward
 * ============================================================================ */

/* B_seq: per-timestep B vectors [seq_len x state_size] (= W_B * x_t for each t) */
static void selective_scan_forward_store(ForwardStore *store, float *state,
                    const float *input, const float *delta,
                    const MBMatrix *A_bar, const float *B_seq,
                    const MBMatrix *C, float D_unused,
                    const float *theta,
                    size_t seq_len, size_t state_size) {
    if (!store || !state) return;
    (void)C; (void)D_unused;

    store->x = (float *)calloc(seq_len * state_size, sizeof(float));
    store->A_diag = (float *)calloc(seq_len * state_size, sizeof(float));
    store->B_bar = (float *)calloc(seq_len * state_size, sizeof(float));
    store->u_seq = (float *)calloc(seq_len * state_size, sizeof(float));
    store->h_rot = (float *)calloc(seq_len * state_size, sizeof(float));

    memset(state, 0, state_size * sizeof(float));

    float *zero_prev = (float *)calloc(state_size, sizeof(float));
    float *h_rot_tmp = (float *)malloc(state_size * sizeof(float));

    for (size_t t = 0; t < seq_len; t++) {
        const float *u_t = &input[t * state_size];
        float dt_t = delta[t];

        for (size_t i = 0; i < state_size; i++) store->u_seq[t * state_size + i] = u_t[i];

        /* Compute A_diag and B_bar */
        for (size_t i = 0; i < state_size; i++) {
            float a_val = A_bar->data[i * state_size + i];
            if (a_val > -1e-5f) a_val = -1e-5f;
            float a_diag_t = expf(dt_t * a_val);
            store->A_diag[t * state_size + i] = a_diag_t;
            store->B_bar[t * state_size + i] = dt_t * B_seq[t * state_size + i];
        }

        /* Apply R(θ) to h_prev → h_rot_tmp */
        float *h_prev = (t == 0) ? zero_prev : &store->x[(t-1)*state_size];
        if (theta) {
            for (size_t i = 0; i + 1 < state_size; i += 2) {
                float th = theta[i >> 1];
                float c = cosf(th), s = sinf(th);
                float h0 = h_prev[i], h1 = h_prev[i+1];
                h_rot_tmp[i]   = c*h0 - s*h1;
                h_rot_tmp[i+1] = s*h0 + c*h1;
            }
            if (state_size & 1) h_rot_tmp[state_size-1] = h_prev[state_size-1];
        } else {
            memcpy(h_rot_tmp, h_prev, state_size * sizeof(float));
        }
        /* Store h_rot for dA gradient in backward */
        memcpy(&store->h_rot[t * state_size], h_rot_tmp, state_size * sizeof(float));

        float *x_t = &store->x[t * state_size];
        for (size_t i = 0; i < state_size; i++) {
            float a_diag_t = store->A_diag[t * state_size + i];
            float bbar = store->B_bar[t * state_size + i];
            x_t[i] = a_diag_t * h_rot_tmp[i] + bbar * u_t[i];
            state[i] = x_t[i];
        }
    }

    free(zero_prev);
    free(h_rot_tmp);
}

/* ============================================================================
 * Backward through stored forward trace
 * ============================================================================ */

static void selective_scan_backward(ForwardStore *store, MambaBlock *block,
                                    const float *dY, const float *input_flat,
                                    float *d_input_out,
                                    const float *theta,
                                    size_t seq_len, size_t state_size) {
    if (!store || !block) return;
    size_t dim = block->config.dim;

    MBOptimState *s = _find_opt(block);
    if (!s) return;

    /* scan_out[t] = C_t * h_t  (elementwise), where C_t = W_C * x_t */
    float *scan_out  = (float *)calloc(seq_len * state_size, sizeof(float));
    float *dY_T      = (float *)calloc(dim * seq_len, sizeof(float));
    float *adj_y     = (float *)calloc(seq_len * state_size, sizeof(float));
    float *scan_du   = (float *)calloc(seq_len * state_size, sizeof(float));
    float *scan_dA   = (float *)calloc(state_size, sizeof(float));
    float *scan_ddelta = (float *)calloc(seq_len, sizeof(float));
    float *contrib_T = (float *)calloc(state_size * seq_len, sizeof(float));
    float *z         = (float *)malloc(state_size * sizeof(float));
    float *C_t       = (float *)malloc(state_size * sizeof(float));
    float *adj_h     = (float *)calloc(state_size, sizeof(float));
    /* Per-timestep B/C gradients: dB_t [seq_len x state_size], dC_t [seq_len x state_size] */
    float *dB_seq    = (float *)calloc(seq_len * state_size, sizeof(float));
    float *dC_seq    = (float *)calloc(seq_len * state_size, sizeof(float));

    if (!scan_out || !dY_T || !adj_y || !scan_du || !scan_dA ||
        !scan_ddelta || !contrib_T || !z || !C_t || !adj_h || !dB_seq || !dC_seq) {
        free(scan_out); free(dY_T); free(adj_y); free(scan_du); free(scan_dA);
        free(scan_ddelta); free(contrib_T); free(z); free(C_t); free(adj_h);
        free(dB_seq); free(dC_seq);
        return;
    }

    /* Recompute scan_out[t] = C_t * h_t for W_out gradient */
    for (size_t t = 0; t < seq_len; t++) {
        mb_matrix_vec_mult(C_t, &block->W_C, &input_flat[t * dim]);
        hadamard_avx2(store->x + t * state_size, C_t,
                      scan_out + t * state_size, (long)state_size);
    }
    transpose_row_major(dY, dY_T, seq_len, dim);

    /* g_W_out += dY^T @ scan_out */
    gemm_avx2(dY_T, scan_out, s->g_W_out,
              (long)dim, (long)seq_len, (long)state_size);

    /* adj_y = dY @ W_out  [seq_len x state_size] */
    gemm_avx2((float *)dY, block->W_out.data, adj_y,
              (long)seq_len, (long)dim, (long)state_size);

    /* ------------------------------------------------------------------ *
     * Inline backward through the SSM scan with data-dependent B and C.  *
     * Recurrence: h_t = A_diag_t * h_{t-1} + B_bar_t * u_t              *
     *             y_t = C_t * h_t       (adj_y is adjoint of y)          *
     * B_bar_t = store->B_bar[t*D + d]   (= delta_t * B_t)               *
     * C_t     = W_C * x_t              (recomputed on the fly)           *
     * ------------------------------------------------------------------ */
    memset(scan_dA,     0, state_size * sizeof(float));
    memset(scan_ddelta, 0, seq_len    * sizeof(float));
    memset(adj_h,       0, state_size * sizeof(float));

    for (long t = (long)seq_len - 1; t >= 0; t--) {
        float dt_t = block->delta[t];
        mb_matrix_vec_mult(C_t, &block->W_C, &input_flat[t * dim]);

        /* d_h_rot[d] will hold ah * a_diag  (intermediate for theta grad + adj_h) */
        float *d_h_rot = (float *)malloc(state_size * sizeof(float));
        if (!d_h_rot) continue;

        for (size_t d = 0; d < state_size; d++) {
            size_t td = (size_t)t * state_size + d;
            float c_t_d   = C_t[d];
            float h_t_d   = store->x[td];
            float b_bar_t = store->B_bar[td];   /* = delta_t * B_t[d]    */
            float u_t_d   = store->u_seq[td];   /* controller from W_in  */
            float a_diag  = store->A_diag[td];  /* exp(delta_t * A_log)  */
            float h_rot_d = store->h_rot[td];   /* R(θ)·h_{t-1}[d]      */
            float a_log_d = block->A_log.data[d];
            if (a_log_d > -1e-5f) a_log_d = -1e-5f;

            /* Adjoint of h_t */
            float ah = adj_h[d] + adj_y[td] * c_t_d;

            /* Gradient for C_t: dC_t[d] = adj_y[t][d] * h_t[d] */
            dC_seq[td] = adj_y[td] * h_t_d;

            /* Gradient for B_t: dB_t[d] = ah * delta_t * u_t[d] */
            dB_seq[td] = ah * dt_t * u_t_d;

            /* Gradient for A_log[d]: use h_rot (not h_prev directly) */
            scan_dA[d] += ah * dt_t * a_diag * h_rot_d;

            /* Gradient for u_t (through B_bar_t * u_t) */
            scan_du[td] = ah * b_bar_t;

            /* Gradient for delta_t (through both A_bar and B_bar) */
            scan_ddelta[(size_t)t] += ah * (a_log_d * a_diag * h_rot_d +
                                            (b_bar_t / (dt_t > 1e-8f ? dt_t : 1e-8f)) * u_t_d);

            /* d_h_rot[d] = ah * a_diag  (gradient through rotation) */
            d_h_rot[d] = ah * a_diag;
        }

        /* Gradient for theta and adj_h through R(θ) */
        {
            MBOptimState *s_opt = _find_opt(block);
            float *h_prev_vec = (t > 0) ? &store->x[((size_t)t - 1) * state_size] : NULL;

            for (size_t i = 0; i + 1 < state_size; i += 2) {
                float th = theta ? theta[i >> 1] : 0.0f;
                float c = cosf(th), s_val = sinf(th);
                float hp0 = h_prev_vec ? h_prev_vec[i]   : 0.0f;
                float hp1 = h_prev_vec ? h_prev_vec[i+1] : 0.0f;
                float dr0 = d_h_rot[i], dr1 = d_h_rot[i+1];

                /* dtheta_i = d_h_rot[2i] * d(R*h)_{2i}/dtheta_i
                 *           + d_h_rot[2i+1] * d(R*h)_{2i+1}/dtheta_i
                 * d/dtheta [ c*h0 - s*h1 ] = -s*h0 - c*h1
                 * d/dtheta [ s*h0 + c*h1 ] =  c*h0 - s*h1 */
                if (theta && s_opt && s_opt->g_theta)
                    s_opt->g_theta[i >> 1] += dr0 * (-s_val * hp0 - c * hp1)
                                            + dr1 * (c * hp0 - s_val * hp1);

                /* adj_h = R^T * d_h_rot  (R^T = rotation by -theta) */
                adj_h[i]   = c * dr0 + s_val * dr1;
                adj_h[i+1] = -s_val * dr0 + c * dr1;
            }
            if (state_size & 1) adj_h[state_size-1] = d_h_rot[state_size-1];
        }
        free(d_h_rot);
    }

    /* Accumulate A gradient */
    for (size_t i = 0; i < state_size; i++)
        s->g_A_log[i] += scan_dA[i];

    /* ---- Backward BCNorm pour B et C ---------------------------------- *
     * Forward:  z = W·x,  rms = 1/sqrt(mean(z²)+eps),  out = z*rms + bias *
     * Backward: d_bias += d_out                                            *
     *           d_z = d_out * rms - z * rms³ * mean(d_out * z)            *
     * -------------------------------------------------------------------- */
    {
        const float eps = 1e-6f;
        for (size_t t = 0; t < seq_len; t++) {
            /* Recompute z_B = W_B · x_t  and  z_C = W_C · x_t */
            float *z_B = (float *)malloc(state_size * sizeof(float));
            float *z_C = (float *)malloc(state_size * sizeof(float));
            if (!z_B || !z_C) { free(z_B); free(z_C); continue; }
            mb_matrix_vec_mult(z_B, &block->W_B, &input_flat[t * dim]);
            mb_matrix_vec_mult(z_C, &block->W_C, &input_flat[t * dim]);

            /* g_b_B += dB_t,  g_b_C += dC_t  (gradient du biais) */
            for (size_t d = 0; d < state_size; d++) {
                s->g_b_B[d] += dB_seq[t * state_size + d];
                s->g_b_C[d] += dC_seq[t * state_size + d];
            }

            /* Backward RMSNorm pour B */
            float rms_b = 0.0f;
            for (size_t d = 0; d < state_size; d++) rms_b += z_B[d] * z_B[d];
            rms_b = 1.0f / sqrtf(rms_b / (float)state_size + eps);
            float dot_b = 0.0f;
            for (size_t d = 0; d < state_size; d++)
                dot_b += dB_seq[t * state_size + d] * z_B[d];
            dot_b /= (float)state_size;
            for (size_t d = 0; d < state_size; d++)
                dB_seq[t * state_size + d] = dB_seq[t * state_size + d] * rms_b
                                            - z_B[d] * (rms_b * rms_b * rms_b) * dot_b;

            /* Backward RMSNorm pour C */
            float rms_c = 0.0f;
            for (size_t d = 0; d < state_size; d++) rms_c += z_C[d] * z_C[d];
            rms_c = 1.0f / sqrtf(rms_c / (float)state_size + eps);
            float dot_c = 0.0f;
            for (size_t d = 0; d < state_size; d++)
                dot_c += dC_seq[t * state_size + d] * z_C[d];
            dot_c /= (float)state_size;
            for (size_t d = 0; d < state_size; d++)
                dC_seq[t * state_size + d] = dC_seq[t * state_size + d] * rms_c
                                            - z_C[d] * (rms_c * rms_c * rms_c) * dot_c;

            free(z_B); free(z_C);
        }
    }

    /* g_W_C += sum_t dC_t @ x_t^T  (après backward BCNorm) */
    {
        float *dC_seq_T = (float *)malloc(state_size * seq_len * sizeof(float));
        if (dC_seq_T) {
            transpose_row_major(dC_seq, dC_seq_T, seq_len, state_size);
            gemm_avx2(dC_seq_T, (float *)input_flat, s->g_W_C,
                      (long)state_size, (long)seq_len, (long)dim);
            free(dC_seq_T);
        }
    }

    /* g_W_B += sum_t dB_t @ x_t^T */
    {
        float *dB_seq_T = (float *)malloc(state_size * seq_len * sizeof(float));
        if (dB_seq_T) {
            transpose_row_major(dB_seq, dB_seq_T, seq_len, state_size);
            gemm_avx2(dB_seq_T, (float *)input_flat, s->g_W_B,
                      (long)state_size, (long)seq_len, (long)dim);
            free(dB_seq_T);
        }
    }

    /* Delta and W_in gradients (unchanged from static-BC version) */
    for (size_t t = 0; t < seq_len; t++) {
        const float *x_input_t = &input_flat[t * dim];
        float ddt_t = scan_ddelta[t];

        {
            float raw_t = 0.0f;
            for (size_t k = 0; k < dim; k++)
                raw_t += block->delta_proj.data[k] * x_input_t[k];
            float draw = ddt_t * scalar_sigmoid(raw_t);
            for (size_t k = 0; k < dim; k++)
                s->g_delta_proj[k] += draw * x_input_t[k];
        }

        mb_matrix_vec_mult(z, &block->W_in, x_input_t);
        for (size_t j = 0; j < state_size; j++) {
            float sig = scalar_sigmoid(z[j]);
            float dz = sig * (1.0f + z[j] * (1.0f - sig));
            scan_out[t * state_size + j] = scan_du[t * state_size + j] * dz;
        }
    }

    transpose_row_major(scan_out, contrib_T, seq_len, state_size);
    gemm_avx2(contrib_T, (float *)input_flat, s->g_W_in,
              (long)state_size, (long)seq_len, (long)dim);

    /* d_input = d_z @ W_in  (seq_len x state_size) @ (state_size x dim) */
    if (d_input_out) {
        gemm_avx2(scan_out, block->W_in.data, d_input_out,
                  (long)seq_len, (long)state_size, (long)dim);
        /* Residual gradient: d_input += dY (identity path) */
        for (size_t i = 0; i < seq_len * dim; i++)
            d_input_out[i] += dY[i];
    }

    free(scan_out); free(dY_T); free(z); free(C_t); free(adj_y);
    free(scan_du); free(scan_dA); free(scan_ddelta); free(contrib_T);
    free(adj_h); free(dB_seq); free(dC_seq);
}

/* ============================================================================
 * Backward entrypoint
 * ============================================================================ */

void mamba_backward(MambaBlock *block, const float *dY, const float *input,
                    float *d_input, size_t batch_index) {
    (void)batch_index;
    size_t seq_len = block->config.seq_len;
    size_t state_size = block->config.state_size;

    MBMatrix *A_bar = mb_matrix_create(state_size, state_size);
    for (size_t i = 0; i < state_size; i++) A_bar->data[i * state_size + i] = block->A_log.data[i];

    ForwardStore store;
    memset(&store, 0, sizeof(store));

    size_t dim = block->config.dim;
    float *u_seq  = (float *)calloc(seq_len * state_size, sizeof(float));
    /* Per-timestep B: B_seq[t] = W_B * x_t */
    float *B_seq  = (float *)malloc(seq_len * state_size * sizeof(float));
    if (!u_seq || !B_seq) { free(u_seq); free(B_seq); mb_matrix_free(A_bar); return; }

    float *tmp_delta = (float *)calloc(block->delta_proj.rows ? block->delta_proj.rows : 1,
                                        sizeof(float));
    float *z = (float *)malloc(state_size * sizeof(float));
    if (!tmp_delta || !z) {
        free(z); free(tmp_delta); free(u_seq); free(B_seq);
        mb_matrix_free(A_bar);
        return;
    }

    {
        const float eps = 1e-6f;
        for (size_t t = 0; t < seq_len; t++) {
            const float *x_t = &input[t * dim];
            project_controller(block, x_t, z, &u_seq[t * state_size]);
            block->delta[t] = project_delta_value(block, x_t, tmp_delta, t, seq_len);

            /* B_seq[t] = RMSNorm(W_B · x_t)  (biais non inclus : annulé dans le scan backward) */
            float *b_t = &B_seq[t * state_size];
            mb_matrix_vec_mult(b_t, &block->W_B, x_t);
            float rms = 0.0f;
            for (size_t d = 0; d < state_size; d++) rms += b_t[d] * b_t[d];
            rms = 1.0f / sqrtf(rms / (float)state_size + eps);
            for (size_t d = 0; d < state_size; d++) b_t[d] = b_t[d] * rms + block->b_B[d];
        }
    }
    free(z);
    free(tmp_delta);

    selective_scan_forward_store(&store, block->hidden, u_seq, block->delta,
                                A_bar, B_seq, &block->W_C, 0.0f,
                                block->theta,
                                seq_len, state_size);

    selective_scan_backward(&store, block, dY, input, d_input,
                            block->theta, seq_len, state_size);

    free(store.x); free(store.A_diag); free(store.B_bar); free(store.u_seq);
    if (store.h_rot) free(store.h_rot);
    free(u_seq); free(B_seq);
    mb_matrix_free(A_bar);
}

/* ============================================================================
 * Forward pass — 1D
 * ============================================================================ */

void mamba_block_forward(MambaBlock *block, float *output, const float *input,
                         size_t batch_size) {
    if (!block || !output || !input) return;

    size_t seq_len = block->config.seq_len;
    size_t dim = block->config.dim;
    size_t state_size = block->config.state_size;

    for (size_t b = 0; b < batch_size; b++) {
        const float *batch_input = &input[b * seq_len * dim];
        float *batch_output = &output[b * seq_len * dim];

        float *u_seq = (float *)calloc(seq_len * state_size, sizeof(float));
        float *z = (float *)malloc(state_size * sizeof(float));
        if (!u_seq || !z) {
            free(z); free(u_seq);
            continue;
        }

        float *tmp_delta = (float *)calloc(block->delta_proj.rows ? block->delta_proj.rows : 1,
                                            sizeof(float));
        if (!tmp_delta) {
            free(z); free(u_seq);
            continue;
        }

        for (size_t t = 0; t < seq_len; t++) {
            const float *x_t = &batch_input[t * dim];
            project_controller(block, x_t, z, &u_seq[t * state_size]);
            block->delta[t] = project_delta_value(block, x_t, tmp_delta, t, seq_len);
        }
        free(z);
        free(tmp_delta);

        float *scan_out = (float *)malloc(seq_len * state_size * sizeof(float));
        if (!scan_out) { free(u_seq); continue; }

        long L = (long)seq_len, D = (long)state_size;

        /* Data-dependent B/C with BCNorm + biases (Mamba-3) */
        {
            const float eps = 1e-6f;
            for (long t = 0; t < L; t++) {
                const float *x_t = &batch_input[t * dim];
                float *b_out = &block->scan_B[t*D];
                float *c_out = &block->scan_C[t*D];

                mb_matrix_vec_mult(b_out, &block->W_B, x_t);
                mb_matrix_vec_mult(c_out, &block->W_C, x_t);

                /* RMSNorm(B_t) */
                float rms_b = 0.0f;
                for (long d = 0; d < D; d++) rms_b += b_out[d] * b_out[d];
                rms_b = 1.0f / sqrtf(rms_b / (float)D + eps);
                for (long d = 0; d < D; d++) b_out[d] = b_out[d] * rms_b + block->b_B[d];

                /* RMSNorm(C_t) */
                float rms_c = 0.0f;
                for (long d = 0; d < D; d++) rms_c += c_out[d] * c_out[d];
                rms_c = 1.0f / sqrtf(rms_c / (float)D + eps);
                for (long d = 0; d < D; d++) c_out[d] = c_out[d] * rms_c + block->b_C[d];

                for (long d = 0; d < D; d++)
                    block->scan_delta[t*D + d] = block->delta[t];
            }
        }
        memset(block->scan_h, 0, (size_t)D * sizeof(float));

        /* Complex SSM scan with R(θ) rotation (Mamba-3 §3.2) */
        {
            float *h_cur = block->scan_h;
            float *h_rot = (float *)malloc((size_t)D * sizeof(float));
            if (!h_rot) { free(u_seq); free(scan_out); continue; }

            for (long t = 0; t < L; t++) {
                float dt_t = block->delta[t];

                /* Apply R(θ) to h_cur → h_rot */
                for (long i = 0; i + 1 < D; i += 2) {
                    float th = block->theta[i >> 1];
                    float c = cosf(th), s = sinf(th);
                    float h0 = h_cur[i], h1 = h_cur[i+1];
                    h_rot[i]   = c*h0 - s*h1;
                    h_rot[i+1] = s*h0 + c*h1;
                }
                if (D & 1) h_rot[D-1] = h_cur[D-1];

                /* h_t = exp(dt*A)*h_rot + dt*B_t*u_t */
                for (long d = 0; d < D; d++) {
                    float a = block->A_log.data[d];
                    if (a > -1e-5f) a = -1e-5f;
                    h_cur[d] = expf(dt_t * a) * h_rot[d]
                               + dt_t * block->scan_B[t*D+d] * u_seq[t*D+d];
                }

                /* scan_out[t] = C_t * h_t */
                for (long d = 0; d < D; d++)
                    scan_out[t*D+d] = block->scan_C[t*D+d] * h_cur[d];
            }
            free(h_rot);
        }
        memcpy(block->hidden, block->scan_h, (size_t)D * sizeof(float));

        float *ybuf = (float *)malloc(dim * sizeof(float));
        if (ybuf) {
            for (size_t t = 0; t < seq_len; t++) {
                const float *state_t = &scan_out[t * state_size];
                mb_matrix_vec_mult(ybuf, &block->W_out, state_t);
                /* Residual connection: output = input + mamba(input) */
                for (size_t j = 0; j < dim; j++)
                    batch_output[t * dim + j] = batch_input[t * dim + j] + ybuf[j];
            }
            free(ybuf);
        }

        free(u_seq);
        free(scan_out);
    }
}

