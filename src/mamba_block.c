/*
 * mamba_block.c — MambaBlock SSM architecture with MUONCLIP optimizer
 *
 * Core Mamba implementation: diagonal A, shared B/C vectors,
 * input-dependent delta, W_in/W_out projections, scan1d/scan2d ASM kernels.
 *
 * Part of k-mamba — uses OpenBLAS for compute kernels.
 */

#include "kmamba.h"
#include "openblas_utils.h"
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
    float *B_bar;     /* seq_len x state_size  (= dt * B_t  for Euler; reused for storage) */
    float *u_seq;     /* seq_len x R (controller vectors per timestep) */
    float *C_seq;     /* seq_len x N*R (BCNorm'd C vectors per timestep) */
    float *h_rot;     /* seq_len x state_size  R(θ)·h_{t-1} at each step (for dA grad) */
    float *Bu;        /* seq_len x state_size  B_t * u_t at each timestep */
    float *lambda;    /* seq_len  scalar lambda_t at each timestep */
    float *alpha;     /* seq_len x state_size  exp(dt*A) at each step */
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
static double        sum_squares_f32(const float *x, size_t n);
static void          bind_default_workspace(MambaBlockWorkspace *ws, MambaBlock *block);
static int           workspace_matches_block(const MambaBlock *block,
                                             const MambaBlockWorkspace *ws);
static int           spatial_dims_product(const long *dims, long ndims,
                                          size_t *product_out);
static int           normalize_spatial_topology(MBConfig *cfg);

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
    gemv_rowmajor(m->data, v, out, (int)m->rows, (int)m->cols);
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
    /* W_in is [R x dim]; output u_out ∈ R^R (MIMO rank) */
    size_t R = _mimo_R(&block->config);
    mb_matrix_vec_mult(z_buf, &block->W_in, x_t);
    silu_f32(z_buf, u_out, (long)R);
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

static double sum_squares_f32(const float *x, size_t n) {
    double acc = 0.0;
    if (!x) return 0.0;
    for (size_t i = 0; i < n; i++) {
        double v = (double)x[i];
        acc += v * v;
    }
    return acc;
}

static void bind_default_workspace(MambaBlockWorkspace *ws, MambaBlock *block) {
    if (!ws || !block) return;
    ws->hidden     = block->hidden;
    ws->delta      = block->delta;
    ws->scan_B     = block->scan_B;
    ws->scan_C     = block->scan_C;
    ws->scan_delta = block->scan_delta;
    ws->scan_h     = block->scan_h;
}

static int workspace_matches_block(const MambaBlock *block,
                                   const MambaBlockWorkspace *ws) {
    size_t state_size;
    size_t seq_len;
    size_t rank;

    if (!block || !ws) return 0;
    state_size = block->config.state_size;
    seq_len = block->config.seq_len;
    rank = _mimo_R(&block->config);

    return ws->hidden &&
           ws->delta &&
           ws->scan_B &&
           ws->scan_C &&
           ws->scan_delta &&
           ws->scan_h &&
           state_size > 0 &&
           seq_len > 0 &&
           rank > 0;
}

static int spatial_dims_product(const long *dims, long ndims,
                                size_t *product_out) {
    size_t product = 1;

    if (!dims || !product_out || ndims <= 0 || ndims > KMAMBA_MAX_NDIMS) return 0;

    for (long axis = 0; axis < ndims; axis++) {
        size_t axis_extent;

        if (dims[axis] <= 0) return 0;
        axis_extent = (size_t)dims[axis];
        if (product > ((size_t)-1) / axis_extent) return 0;
        product *= axis_extent;
    }

    *product_out = product;
    return 1;
}

static int normalize_spatial_topology(MBConfig *cfg) {
    size_t total_points;

    if (!cfg) return -1;

    if (cfg->spatial_ndims <= 0) {
        memset(cfg->spatial_dims, 0, sizeof(cfg->spatial_dims));
        cfg->spatial_ndims = 1;
        cfg->spatial_dims[0] = (long)cfg->seq_len;
    }

    if (cfg->spatial_ndims > KMAMBA_MAX_NDIMS) return -1;
    if (!wavefront_nd_validate_dims(cfg->spatial_dims, cfg->spatial_ndims)) return -1;
    if (!spatial_dims_product(cfg->spatial_dims, cfg->spatial_ndims, &total_points)) return -1;
    if (total_points != cfg->seq_len) return -1;

    if (cfg->use_convnd) {
        if (cfg->convnd_K <= 0) return -1;
        if (cfg->convnd_ndims <= 0) cfg->convnd_ndims = cfg->spatial_ndims;
        if (cfg->convnd_ndims != cfg->spatial_ndims) return -1;
    }

    return 0;
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

        hadamard(A_diag_t, state, temp_state, (long)state_size);
        hadamard(B_bar_t, (float *)u_t, state, (long)state_size);
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
    /* Ensure mimo_rank is at least 1 */
    if (block->config.mimo_rank == 0) block->config.mimo_rank = 1;
    if (normalize_spatial_topology(&block->config) != 0) {
        mamba_block_free(block);
        return NULL;
    }

    size_t R = _mimo_R(&block->config);
    size_t N = block->config.state_size;

    if (matrix_init_owned(&block->W_in, R, block->config.dim) != 0 ||
        matrix_init_owned(&block->W_out, block->config.dim, R) != 0 ||
        matrix_init_owned(&block->A_log, N, 1) != 0 ||
        matrix_init_owned(&block->W_B, N * R, block->config.dim) != 0 ||
        matrix_init_owned(&block->W_C, N * R, block->config.dim) != 0 ||
        matrix_init_owned(&block->delta_proj, 1, block->config.dim) != 0 ||
        matrix_init_owned(&block->lambda_proj, 1, block->config.dim) != 0) {
        mamba_block_free(block);
        return NULL;
    }
    /* BCNorm biases — init séparé pour clarté */
    block->b_B = (float *)calloc(block->config.state_size * R, sizeof(float));
    block->b_C = (float *)calloc(block->config.state_size * R, sizeof(float));
    if (!block->b_B || !block->b_C) {
        mamba_block_free(block);
        return NULL;
    }

    /* Complex SSM rotation angles theta [state_size/2] */
    size_t theta_size = block->config.state_size / 2;
    if (theta_size == 0) theta_size = 1;  /* guard for tiny state_size */
    block->theta = (float *)calloc(theta_size, sizeof(float));
    if (!block->theta) {
        mamba_block_free(block);
        return NULL;
    }

    block->hidden = (float *)calloc(N, sizeof(float));
    block->delta  = (float *)calloc(block->config.seq_len, sizeof(float));

    /* scan_B/scan_C: [L x N*R] for MIMO (R=1 → identical to previous) */
    size_t LNR = block->config.seq_len * N * R;
    size_t LD  = block->config.seq_len * N;
    block->scan_B     = (float *)malloc(LNR * sizeof(float));
    block->scan_C     = (float *)malloc(LNR * sizeof(float));
    block->scan_delta = (float *)malloc(LD  * sizeof(float));
    block->scan_h     = (float *)calloc(N, sizeof(float));
    block->wavefront_plan = km_wavefront_plan_create(block->config.spatial_dims,
                                                     block->config.spatial_ndims);

    if (!block->W_in.data || !block->W_out.data || !block->A_log.data ||
        !block->W_B.data || !block->W_C.data || !block->delta_proj.data ||
        !block->hidden || !block->delta ||
        !block->scan_B || !block->scan_C || !block->scan_delta || !block->scan_h ||
        !block->wavefront_plan) {
        mamba_block_free(block);
        return NULL;
    }

    /* Allocate ConvND resources if enabled */
    if (block->config.use_convnd &&
        block->config.convnd_K > 0 &&
        block->config.convnd_ndims > 0) {
        long kernel_size = block->config.convnd_ndims *
                           block->config.convnd_K *
                           (long)block->config.dim;
        block->convnd_kernel = (float *)calloc(kernel_size, sizeof(float));
        block->convnd_bias = (float *)calloc(block->config.dim, sizeof(float));

        /* The workspace size depends on the concrete ND shape, which is not
         * known at block creation time. Allocate it lazily at call site. */
        block->convnd_ws = NULL;

        if (!block->convnd_kernel || !block->convnd_bias) {
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
    if (block->lambda_proj.data) free(block->lambda_proj.data);
    if (block->hidden) free(block->hidden);
    if (block->delta) free(block->delta);
    if (block->scan_B)     free(block->scan_B);
    if (block->scan_C)     free(block->scan_C);
    if (block->scan_delta) free(block->scan_delta);
    if (block->scan_h)     free(block->scan_h);
    if (block->wavefront_plan) km_wavefront_plan_free(block->wavefront_plan);
    
    /* Free ConvND resources */
    if (block->convnd_kernel) free(block->convnd_kernel);
    if (block->convnd_bias)   free(block->convnd_bias);
    if (block->convnd_ws)     convnd_workspace_free(block->convnd_ws);

    free(block);
}

MambaBlockWorkspace* mamba_block_workspace_create(const MambaBlock *block) {
    MambaBlockWorkspace *ws;
    size_t state_size;
    size_t seq_len;
    size_t rank;
    size_t scan_nr;
    size_t scan_nd;

    if (!block) return NULL;

    state_size = block->config.state_size;
    seq_len = block->config.seq_len;
    rank = _mimo_R(&block->config);
    scan_nr = seq_len * state_size * rank;
    scan_nd = seq_len * state_size;

    ws = (MambaBlockWorkspace *)calloc(1, sizeof(*ws));
    if (!ws) return NULL;

    ws->hidden     = (float *)calloc(state_size, sizeof(float));
    ws->delta      = (float *)calloc(seq_len, sizeof(float));
    ws->scan_B     = (float *)malloc(scan_nr * sizeof(float));
    ws->scan_C     = (float *)malloc(scan_nr * sizeof(float));
    ws->scan_delta = (float *)malloc(scan_nd * sizeof(float));
    ws->scan_h     = (float *)calloc(state_size, sizeof(float));

    if (!workspace_matches_block(block, ws)) {
        mamba_block_workspace_free(ws);
        return NULL;
    }

    return ws;
}

void mamba_block_workspace_free(MambaBlockWorkspace *ws) {
    if (!ws) return;
    free(ws->hidden);
    free(ws->delta);
    free(ws->scan_B);
    free(ws->scan_C);
    free(ws->scan_delta);
    free(ws->scan_h);
    free(ws);
}

void mamba_block_init(MambaBlock *block) {
    if (!block) return;

    /* A_log stocke directement A (valeur négative).
     * A = -1 pour toutes les dimensions → decay = exp(-Δ) ∈ (0.37, 0.999).
     * La stabilité est garantie par le clamp A ≤ -1e-5 dans le forward. */
    for (size_t i = 0; i < block->config.state_size; i++) {
        block->A_log.data[i] = -1.0f;
    }

    /* Xavier uniform init for all weight matrices (sizes set in create()) */
    {
        MBMatrix *mats[] = { &block->W_in, &block->W_out, &block->W_B, &block->W_C };
        for (int mi = 0; mi < 4; mi++) {
            MBMatrix *M = mats[mi];
            float fan_in  = (float)M->cols;
            float fan_out = (float)M->rows;
            float scale   = sqrtf(6.0f / (fan_in + fan_out));
            size_t total  = M->rows * M->cols;
            for (size_t i = 0; i < total; i++)
                M->data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
    /* BCNorm biases — initialisés à zéro (comportement neutre au départ) */
    {
        size_t R_init = _mimo_R(&block->config);
        memset(block->b_B, 0, block->config.state_size * R_init * sizeof(float));
        memset(block->b_C, 0, block->config.state_size * R_init * sizeof(float));
    }

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

    /* Small uniform init for lambda_proj (1 x dim) — sigmoid(0) = 0.5, neutral start */
    for (size_t i = 0; i < block->lambda_proj.rows * block->lambda_proj.cols; i++) {
        block->lambda_proj.data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
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
    size_t dim   = block->config.dim;
    size_t state = block->config.state_size;
    size_t R     = _mimo_R(&block->config);
    /* MIMO-aware sizes */
    size_t size_in  = R * dim;        /* W_in:  [R x dim] */
    size_t size_out = dim * R;        /* W_out: [dim x R] */
    size_t size_bc  = state * R * dim;/* W_B/W_C: [N*R x dim] */
    memset(s, 0, sizeof(MBOptimState));

    s->type = type;

    s->g_W_in = (float *)calloc(size_in, sizeof(float));
    s->g_W_out = (float *)calloc(size_out, sizeof(float));
    s->g_A_log = (float *)calloc(state, sizeof(float));
    s->g_W_B = (float *)calloc(size_bc, sizeof(float));
    s->g_W_C = (float *)calloc(size_bc, sizeof(float));
    s->g_b_B = (float *)calloc(state * R, sizeof(float));
    s->g_b_C = (float *)calloc(state * R, sizeof(float));
    s->g_delta_proj  = (float *)calloc(dim, sizeof(float));
    s->g_lambda_proj = (float *)calloc(dim, sizeof(float));
    size_t theta_size = state / 2; if (theta_size == 0) theta_size = 1;
    s->g_theta       = (float *)calloc(theta_size, sizeof(float));

    /* Allocate first moments for all momentum-based optimizers */
    if (type == OPTIMIZER_ADAM_CLIP || type == OPTIMIZER_ADAMW ||
        type == OPTIMIZER_MUON      || type == OPTIMIZER_SGD) {
        s->m_W_in       = (float *)calloc(size_in,  sizeof(float));
        s->m_W_out      = (float *)calloc(size_out, sizeof(float));
        s->m_A_log      = (float *)calloc(state,    sizeof(float));
        s->m_W_B        = (float *)calloc(size_bc,  sizeof(float));
        s->m_W_C        = (float *)calloc(size_bc,  sizeof(float));
        s->m_b_B        = (float *)calloc(state * R, sizeof(float));
        s->m_b_C        = (float *)calloc(state * R, sizeof(float));
        s->m_delta_proj  = (float *)calloc(dim,       sizeof(float));
        s->m_lambda_proj = (float *)calloc(dim,       sizeof(float));
        s->m_theta       = (float *)calloc(theta_size, sizeof(float));
    }
    /* Allocate second moments only for Adam-based optimizers */
    if (type == OPTIMIZER_ADAM_CLIP || type == OPTIMIZER_ADAMW) {
        s->v_W_in        = (float *)calloc(size_in,  sizeof(float));
        s->v_W_out       = (float *)calloc(size_out, sizeof(float));
        s->v_A_log       = (float *)calloc(state,    sizeof(float));
        s->v_W_B         = (float *)calloc(size_bc,  sizeof(float));
        s->v_W_C         = (float *)calloc(size_bc,  sizeof(float));
        s->v_b_B         = (float *)calloc(state * R, sizeof(float));
        s->v_b_C         = (float *)calloc(state * R, sizeof(float));
        s->v_delta_proj  = (float *)calloc(dim,      sizeof(float));
        s->v_lambda_proj = (float *)calloc(dim,      sizeof(float));
        s->v_theta       = (float *)calloc(theta_size, sizeof(float));
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
            if (s->g_lambda_proj) free(s->g_lambda_proj);

            /* Free moments only if allocated */
            if (s->m_W_in) { free(s->m_W_in); free(s->v_W_in); }
            if (s->m_W_out) { free(s->m_W_out); free(s->v_W_out); }
            if (s->m_A_log) { free(s->m_A_log); free(s->v_A_log); }
            if (s->m_W_B) { free(s->m_W_B); if (s->v_W_B) free(s->v_W_B); }
            if (s->m_W_C) { free(s->m_W_C); if (s->v_W_C) free(s->v_W_C); }
            if (s->m_b_B) { free(s->m_b_B); if (s->v_b_B) free(s->v_b_B); }
            if (s->m_b_C) { free(s->m_b_C); if (s->v_b_C) free(s->v_b_C); }
            if (s->m_delta_proj) { free(s->m_delta_proj); if (s->v_delta_proj) free(s->v_delta_proj); }
            if (s->m_lambda_proj) { free(s->m_lambda_proj); if (s->v_lambda_proj) free(s->v_lambda_proj); }
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
            size_t dim   = block->config.dim;
            size_t state = block->config.state_size;
            size_t R     = _mimo_R(&block->config);
            size_t size_in  = R * dim;
            size_t size_out = dim * R;
            size_t size_bc  = state * R * dim;
            memset(s->g_W_in,  0, size_in  * sizeof(float));
            memset(s->g_W_out, 0, size_out * sizeof(float));
            memset(s->g_A_log, 0, state    * sizeof(float));
            memset(s->g_W_B,   0, size_bc  * sizeof(float));
            memset(s->g_W_C,   0, size_bc  * sizeof(float));
            memset(s->g_b_B,   0, state * R * sizeof(float));
            memset(s->g_b_C,   0, state * R * sizeof(float));
            memset(s->g_delta_proj,  0, dim * sizeof(float));
            memset(s->g_lambda_proj, 0, dim * sizeof(float));
            { size_t ts = state / 2; if (ts == 0) ts = 1; memset(s->g_theta, 0, ts * sizeof(float)); }
            return;
        }
    }
}

float mamba_block_grad_sqnorm(const MambaBlock *block) {
    MBOptimState *s;
    size_t dim;
    size_t state;
    size_t R;
    size_t size_in;
    size_t size_out;
    size_t size_bc;
    size_t theta_size;
    double acc = 0.0;

    if (!block) return 0.0f;

    s = _find_opt((MambaBlock *)block);
    if (!s) return 0.0f;

    dim = block->config.dim;
    state = block->config.state_size;
    R = _mimo_R(&block->config);

    size_in = R * dim;
    size_out = dim * R;
    size_bc = state * R * dim;
    theta_size = state / 2;
    if (theta_size == 0) theta_size = 1;

    acc += sum_squares_f32(s->g_W_in,        size_in);
    acc += sum_squares_f32(s->g_W_out,       size_out);
    acc += sum_squares_f32(s->g_A_log,       state);
    acc += sum_squares_f32(s->g_W_B,         size_bc);
    acc += sum_squares_f32(s->g_W_C,         size_bc);
    acc += sum_squares_f32(s->g_b_B,         state * R);
    acc += sum_squares_f32(s->g_b_C,         state * R);
    acc += sum_squares_f32(s->g_delta_proj,  dim);
    acc += sum_squares_f32(s->g_lambda_proj, dim);
    acc += sum_squares_f32(s->g_theta,       theta_size);

    return (float)acc;
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
    size_t R = _mimo_R(&block->config);
    size_t size_in = R * dim; size_t size_out = dim * R;
    size_t size_bc = state * R * dim;

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
    ADAM_CLIP_UPDATE(block->W_B.data, s->g_W_B, s->m_W_B, s->v_W_B, size_bc);
    ADAM_CLIP_UPDATE(block->W_C.data, s->g_W_C, s->m_W_C, s->v_W_C, size_bc);
    ADAM_CLIP_UPDATE(block->b_B, s->g_b_B, s->m_b_B, s->v_b_B, state * R);
    ADAM_CLIP_UPDATE(block->b_C, s->g_b_C, s->m_b_C, s->v_b_C, state * R);
    ADAM_CLIP_UPDATE(block->delta_proj.data,  s->g_delta_proj,  s->m_delta_proj,  s->v_delta_proj,  dim);
    ADAM_CLIP_UPDATE(block->lambda_proj.data, s->g_lambda_proj, s->m_lambda_proj, s->v_lambda_proj, dim);
    { size_t ts = state/2; if (ts==0) ts=1;
      ADAM_CLIP_UPDATE(block->theta, s->g_theta, s->m_theta, s->v_theta, ts); }

#undef ADAM_CLIP_UPDATE
#endif
}

static void adamw_update(MambaBlock *block, MBOptimState *s, const MBOptimConfig *conf) {
    size_t dim = block->config.dim; size_t state = block->config.state_size;
    size_t R = _mimo_R(&block->config);
    size_t size_in = R * dim; size_t size_out = dim * R;
    size_t size_bc = state * R * dim;
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
    ADAMW_UPDATE(block->W_B.data, s->g_W_B, s->m_W_B, s->v_W_B, size_bc);
    ADAMW_UPDATE(block->W_C.data, s->g_W_C, s->m_W_C, s->v_W_C, size_bc);
    ADAMW_UPDATE(block->b_B, s->g_b_B, s->m_b_B, s->v_b_B, state * R);
    ADAMW_UPDATE(block->b_C, s->g_b_C, s->m_b_C, s->v_b_C, state * R);
    ADAMW_UPDATE(block->delta_proj.data,  s->g_delta_proj,  s->m_delta_proj,  s->v_delta_proj,  dim);
    ADAMW_UPDATE(block->lambda_proj.data, s->g_lambda_proj, s->m_lambda_proj, s->v_lambda_proj, dim);
    { size_t ts = state/2; if (ts==0) ts=1;
      ADAMW_UPDATE(block->theta, s->g_theta, s->m_theta, s->v_theta, ts); }

#undef ADAMW_UPDATE
}

static void sgd_update(MambaBlock *block, MBOptimState *s, const MBOptimConfig *conf) {
    size_t dim = block->config.dim; size_t state = block->config.state_size;
    size_t R = _mimo_R(&block->config);
    size_t size_in = R * dim; size_t size_out = dim * R;
    size_t size_bc = state * R * dim;
    float lr = conf->lr; float mu = conf->mu; float wd = conf->weight_decay;

#define SGD_UPDATE(param, grad, m, N) do { \
    for (size_t _i=0; _i < (N); _i++) { float g = grad[_i] + wd * param[_i]; \
        m[_i] = mu * m[_i] + g; \
        param[_i] -= lr * m[_i]; } \
    } while (0)

    SGD_UPDATE(block->W_in.data, s->g_W_in, s->m_W_in, size_in);
    SGD_UPDATE(block->W_out.data, s->g_W_out, s->m_W_out, size_out);
    SGD_UPDATE(block->A_log.data, s->g_A_log, s->m_A_log, state);
    SGD_UPDATE(block->W_B.data, s->g_W_B, s->m_W_B, size_bc);
    SGD_UPDATE(block->W_C.data, s->g_W_C, s->m_W_C, size_bc);
    SGD_UPDATE(block->b_B, s->g_b_B, s->m_b_B, state * R);
    SGD_UPDATE(block->b_C, s->g_b_C, s->m_b_C, state * R);
    SGD_UPDATE(block->delta_proj.data,  s->g_delta_proj,  s->m_delta_proj,  dim);
    SGD_UPDATE(block->lambda_proj.data, s->g_lambda_proj, s->m_lambda_proj, dim);
    { size_t ts = state/2; if (ts==0) ts=1;
      SGD_UPDATE(block->theta, s->g_theta, s->m_theta, ts); }

#undef SGD_UPDATE
}

/* MUON : matrices (W_in, W_out) — Newton-Schulz + momentum + clipping */
static void muon_update(MambaBlock *block, MBOptimState *s, const MBOptimConfig *conf) {
    size_t dim = block->config.dim; size_t state = block->config.state_size;
    size_t R = _mimo_R(&block->config);

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

    MUON_UPDATE_MAT(block->W_in.data,  s->g_W_in,  s->m_W_in,  R,         dim);
    MUON_UPDATE_MAT(block->W_out.data, s->g_W_out, s->m_W_out, dim,       R);
    MUON_UPDATE_VEC(block->A_log.data,       s->g_A_log,       s->m_A_log,       state);
    MUON_UPDATE_MAT(block->W_B.data,   s->g_W_B,   s->m_W_B,   state * R, dim);
    MUON_UPDATE_MAT(block->W_C.data,   s->g_W_C,   s->m_W_C,   state * R, dim);
    MUON_UPDATE_VEC(block->b_B,        s->g_b_B,   s->m_b_B,   state * R);
    MUON_UPDATE_VEC(block->b_C,        s->g_b_C,   s->m_b_C,   state * R);
    MUON_UPDATE_VEC(block->delta_proj.data,  s->g_delta_proj,  s->m_delta_proj,  dim);
    MUON_UPDATE_VEC(block->lambda_proj.data, s->g_lambda_proj, s->m_lambda_proj, dim);
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
 * Per-thread local gradient helpers (for lock-free parallel backward)
 * ============================================================================ */

MBOptimState* mamba_local_grad_alloc(const MambaBlock *block) {
    if (!block) return NULL;
    MBOptimState *s = (MBOptimState *)calloc(1, sizeof(MBOptimState));
    if (!s) return NULL;
    size_t dim   = block->config.dim;
    size_t state = block->config.state_size;
    size_t R     = _mimo_R(&block->config);
    size_t NR    = state * R;
    size_t theta_size = state / 2; if (theta_size == 0) theta_size = 1;

    s->g_W_in        = (float *)calloc(R * dim,    sizeof(float));
    s->g_W_out       = (float *)calloc(dim * R,    sizeof(float));
    s->g_A_log       = (float *)calloc(state,      sizeof(float));
    s->g_W_B         = (float *)calloc(NR * dim,   sizeof(float));
    s->g_W_C         = (float *)calloc(NR * dim,   sizeof(float));
    s->g_b_B         = (float *)calloc(state * R,  sizeof(float));
    s->g_b_C         = (float *)calloc(state * R,  sizeof(float));
    s->g_delta_proj  = (float *)calloc(dim,        sizeof(float));
    s->g_lambda_proj = (float *)calloc(dim,        sizeof(float));
    s->g_theta       = (float *)calloc(theta_size, sizeof(float));
    s->type = OPTIMIZER_SGD; /* sentinel: not used for stepping */
    return s;
}

void mamba_local_grad_reduce(MambaBlock *block, const MBOptimState *local) {
    if (!local) return;
    MBOptimState *s = _find_opt(block);
    if (!s) return;
    size_t dim   = block->config.dim;
    size_t state = block->config.state_size;
    size_t R     = _mimo_R(&block->config);
    size_t NR    = state * R;
    size_t theta_size = state / 2; if (theta_size == 0) theta_size = 1;
#define REDUCE(field, n) do { \
    if (s->field && local->field) \
        for (size_t _i = 0; _i < (n); _i++) s->field[_i] += local->field[_i]; \
} while(0)
    REDUCE(g_W_in,        R * dim);
    REDUCE(g_W_out,       dim * R);
    REDUCE(g_A_log,       state);
    REDUCE(g_W_B,         NR * dim);
    REDUCE(g_W_C,         NR * dim);
    REDUCE(g_b_B,         state * R);
    REDUCE(g_b_C,         state * R);
    REDUCE(g_delta_proj,  dim);
    REDUCE(g_lambda_proj, dim);
    REDUCE(g_theta,       theta_size);
#undef REDUCE
}

void mamba_local_grad_free(MBOptimState *local) {
    if (!local) return;
    free(local->g_W_in); free(local->g_W_out); free(local->g_A_log);
    free(local->g_W_B);  free(local->g_W_C);
    free(local->g_b_B);  free(local->g_b_C);
    free(local->g_delta_proj);
    free(local->g_lambda_proj);
    free(local->g_theta);
    free(local);
}

/* ============================================================================
 * Forward scan with storage for backward
 * ============================================================================ */

/* B_seq: per-timestep B vectors [seq_len x N*R] (BCNorm'd output of W_B)
 * input_u: [seq_len x R] SiLU(W_in @ x_t) — MIMO input vectors
 * input_flat: [seq_len x dim] needed for lambda_proj computation
 * lambda_proj_param: MBMatrix [1 x dim]
 * R_rank: MIMO rank (1 = SISO compat)
 */
static void selective_scan_forward_store(ForwardStore *store, float *state,
                    const float *input_u,     /* [seq_len x R] SiLU(W_in x) */
                    const float *delta,
                    const MBMatrix *A_bar, const float *B_seq,  /* [seq_len x N*R] */
                    const float *C_seq,                          /* [seq_len x N*R] */
                    const MBMatrix *C, float D_unused,
                    const float *theta,
                    const float *input_flat,  /* [seq_len x dim] raw embeddings */
                    const MBMatrix *lambda_proj_param,
                    size_t seq_len, size_t state_size, size_t dim, size_t R_rank) {
    if (!store || !state) return;
    (void)C; (void)D_unused;

    store->x      = (float *)calloc(seq_len * state_size, sizeof(float));
    store->A_diag = (float *)calloc(seq_len * state_size, sizeof(float));
    store->B_bar  = (float *)calloc(seq_len * state_size, sizeof(float));
    /* u_seq stored as [seq_len x R] */
    store->u_seq  = (float *)calloc(seq_len * R_rank, sizeof(float));
    store->C_seq  = (float *)calloc(seq_len * state_size * R_rank, sizeof(float));
    store->h_rot  = (float *)calloc(seq_len * state_size, sizeof(float));
    /* Bu stored as [seq_len x N]: Bu_t[n] = sum_r B_t[n,r]*u_t[r] */
    store->Bu     = (float *)calloc(seq_len * state_size, sizeof(float));
    store->lambda = (float *)calloc(seq_len, sizeof(float));
    store->alpha  = (float *)calloc(seq_len * state_size, sizeof(float));

    memset(state, 0, state_size * sizeof(float));

    float *zero_prev = (float *)calloc(state_size, sizeof(float));
    float *h_rot_tmp = (float *)malloc(state_size * sizeof(float));
    float *prev_Bu   = (float *)calloc(state_size, sizeof(float));
    float *lam_buf   = (float *)malloc(sizeof(float));

    for (size_t t = 0; t < seq_len; t++) {
        const float *u_t = &input_u[t * R_rank];  /* u_t ∈ R^R */
        float dt_t = delta[t];

        /* Store u_t in [seq_len x R] layout */
        for (size_t r = 0; r < R_rank; r++) store->u_seq[t * R_rank + r] = u_t[r];
        if (store->C_seq && C_seq) {
            memcpy(&store->C_seq[t * state_size * R_rank],
                   &C_seq[t * state_size * R_rank],
                   state_size * R_rank * sizeof(float));
        }

        /* Compute lambda_t = sigmoid(lambda_proj · x_t) */
        float lam_t = 0.5f;
        if (lambda_proj_param && lambda_proj_param->data && input_flat) {
            mb_matrix_vec_mult(lam_buf, lambda_proj_param, &input_flat[t * dim]);
            lam_t = 1.0f / (1.0f + expf(-lam_buf[0]));
        }
        store->lambda[t] = lam_t;

        /* MIMO Bu_t[n] = sum_r B_t[n,r] * u_t[r]
         * B_seq layout: [L x N*R] where B_seq[t*N*R + r*N + n] = B_t[n,r] */
        const float *b_t = &B_seq[t * state_size * R_rank];
        for (size_t i = 0; i < state_size; i++) {
            float a_val = A_bar->data[i * state_size + i];
            if (a_val > -1e-5f) a_val = -1e-5f;
            float a_diag_t = expf(dt_t * a_val);
            store->A_diag[t * state_size + i] = a_diag_t;
            store->alpha[t * state_size + i]  = a_diag_t;
            /* B_bar: first rank slice * dt (for backward compat; full MIMO uses Bu directly) */
            store->B_bar[t * state_size + i]  = dt_t * b_t[i]; /* rank-0 slice */

            /* Bu_t[i] = sum_r B_t[i,r] * u_t[r] */
            float bu = 0.0f;
            for (size_t r = 0; r < R_rank; r++)
                bu += b_t[r * state_size + i] * u_t[r];
            store->Bu[t * state_size + i] = bu;
        }

        /* Apply R(θ) to h_prev → h_rot_tmp */
        float *h_prev = (t == 0) ? zero_prev : &store->x[(t-1)*state_size];
        if (theta) {
            for (size_t i = 0; i + 1 < state_size; i += 2) {
                float th = theta[i >> 1];
                float c = cosf(th), sv = sinf(th);
                float h0 = h_prev[i], h1 = h_prev[i+1];
                h_rot_tmp[i]   = c*h0 - sv*h1;
                h_rot_tmp[i+1] = sv*h0 + c*h1;
            }
            if (state_size & 1) h_rot_tmp[state_size-1] = h_prev[state_size-1];
        } else {
            memcpy(h_rot_tmp, h_prev, state_size * sizeof(float));
        }
        /* Store h_rot for dA gradient in backward */
        memcpy(&store->h_rot[t * state_size], h_rot_tmp, state_size * sizeof(float));

        float *x_t = &store->x[t * state_size];
        for (size_t i = 0; i < state_size; i++) {
            float a_diag_t  = store->A_diag[t * state_size + i];
            float beta_t    = (1.0f - lam_t) * dt_t * a_diag_t;
            float gamma_t   = lam_t * dt_t;
            float bu_t      = store->Bu[t * state_size + i];
            x_t[i] = a_diag_t * h_rot_tmp[i] + beta_t * prev_Bu[i] + gamma_t * bu_t;
            state[i] = x_t[i];
            prev_Bu[i] = bu_t;
        }
    }

    free(zero_prev);
    free(h_rot_tmp);
    free(prev_Bu);
    free(lam_buf);
}

/* ============================================================================
 * Backward through stored forward trace
 * ============================================================================ */

static void selective_scan_backward(ForwardStore *store, MambaBlock *block,
                                    const MambaBlockWorkspace *ws,
                                    const float *dY, const float *input_flat,
                                    float *d_input_out,
                                    const float *theta,
                                    size_t seq_len, size_t state_size, size_t R_rank,
                                    MBOptimState *s_ext) {
    if (!store || !block || !ws) return;
    size_t dim = block->config.dim;

    MBOptimState *s = s_ext ? s_ext : _find_opt(block);
    if (!s) return;

    /* MIMO: scan_out[t] ∈ R^R, adj_y ∈ R^R per timestep
     * W_out: [dim x R], so:
     *   g_W_out += dY^T @ scan_out_R   (dim x L) @ (L x R) → (dim x R) ✓
     *   adj_y_R = dY @ W_out^T          (L x dim) @ (dim x R) → (L x R)
     * Then dh_t = adj_y_R_t @ C_t      backprop through y_t = C_t^T h_t
     */
    /* scan_out_R: [L x R] — y_t vectors for W_out gradient */
    float *scan_out  = (float *)calloc(seq_len * R_rank, sizeof(float));
    float *dY_T      = (float *)calloc(dim * seq_len, sizeof(float));
    /* adj_y_R: [L x R] — gradient dL/dy_t before broadcasting to state */
    float *adj_y_R   = (float *)calloc(seq_len * R_rank, sizeof(float));
    /* adj_y: [L x N] — gradient dL/dh_t from C output side */
    float *adj_y     = (float *)calloc(seq_len * state_size, sizeof(float));
    float *scan_du   = (float *)calloc(seq_len * R_rank, sizeof(float));
    float *scan_dA   = (float *)calloc(state_size, sizeof(float));
    float *scan_ddelta = (float *)calloc(seq_len, sizeof(float));
    float *contrib_T = (float *)calloc(R_rank * seq_len, sizeof(float));
    size_t z_size = (R_rank > state_size) ? R_rank : state_size;
    float *z         = (float *)malloc(z_size * sizeof(float));
    float *adj_h     = (float *)calloc(state_size, sizeof(float));
    /* Per-timestep B/C gradients: dB_t [seq_len x N*R], dC_t [seq_len x N*R] */
    float *dB_seq    = (float *)calloc(seq_len * state_size * R_rank, sizeof(float));
    float *dC_seq    = (float *)calloc(seq_len * state_size * R_rank, sizeof(float));

    if (!scan_out || !dY_T || !adj_y_R || !adj_y || !scan_du || !scan_dA ||
        !scan_ddelta || !contrib_T || !z || !adj_h || !dB_seq || !dC_seq) {
        free(scan_out); free(dY_T); free(adj_y_R); free(adj_y); free(scan_du); free(scan_dA);
        free(scan_ddelta); free(contrib_T); free(z); free(adj_h);
        free(dB_seq); free(dC_seq);
        return;
    }

    /* Recompute scan_out_R[t,r] = sum_n C_t[n,r] * h_t[n] (MIMO: C^T @ h) for W_out grad.
     * Use the same BCNorm'd C_t that the forward path used.
     */
    for (size_t t = 0; t < seq_len; t++) {
        const float *c_t = store->C_seq ? &store->C_seq[t * state_size * R_rank] : NULL;
        if (!c_t) continue;
        /* c_t is [N*R] with layout [r*N + n]; scan_out[t,r] = sum_n c_t[r*N+n] * h[n] */
        for (size_t r = 0; r < R_rank; r++) {
            float yr = 0.0f;
            for (size_t n = 0; n < state_size; n++)
                yr += c_t[r * state_size + n] * store->x[t * state_size + n];
            scan_out[t * R_rank + r] = yr;
        }
    }
    transpose_row_major(dY, dY_T, seq_len, dim);

    /* g_W_out += dY^T @ scan_out_R  — (dim x L) @ (L x R) → (dim x R) */
    gemm_rowmajor(dY_T, scan_out, s->g_W_out,
                  (int)dim, (int)seq_len, (int)R_rank);

    /* adj_y_R = dY @ W_out^T  — (L x dim) @ (dim x R) = (L x R)
     * W_out is [dim x R], W_out^T is [R x dim]. We need (L x dim) @ (dim x R).
     * Using gemm_rowmajor(A, B, C, M, K, N): C[M,N] = A[M,K] @ B[K,N] */
    gemm_rowmajor((float *)dY, block->W_out.data, adj_y_R,
                  (int)seq_len, (int)dim, (int)R_rank);

    /* From adj_y_R [L x R] and BCNorm'd C_t [N*R], compute adj_y [L x N] = C_t @ adj_y_R_t:
     * adj_y[t,n] = sum_r C_t[n,r] * adj_y_R[t,r]    (adjoint of y_t = C_t^T @ h_t) */
    for (size_t t = 0; t < seq_len; t++) {
        const float *c_t = store->C_seq ? &store->C_seq[t * state_size * R_rank] : NULL;
        if (!c_t) continue;
        /* c_t[r*N + n] = C_t[n,r]; adj_y[t,n] = sum_r C_t[n,r] * adj_y_R[t,r] */
        for (size_t n = 0; n < state_size; n++) {
            float a = 0.0f;
            for (size_t r = 0; r < R_rank; r++)
                a += c_t[r * state_size + n] * adj_y_R[t * R_rank + r];
            adj_y[t * state_size + n] = a;
        }
    }

    /* ------------------------------------------------------------------ *
     * Inline backward through the SSM scan.                              *
     * 3-term recurrence (exp-trapezoidal + complex SSM):                 *
     * h_t = alpha_t * R(θ)*h_{t-1} + beta_t * Bu_{t-1} + gamma_t * Bu_t *
     * where: alpha_t  = exp(dt*A)                                        *
     *        lambda_t = sigmoid(lambda_proj · x_t)  [stored in store]   *
     *        beta_t   = (1-lambda_t) * dt * alpha_t                     *
     *        gamma_t  = lambda_t * dt                                    *
     *        Bu_t     = B_t * u_t   [stored in store->Bu]               *
     * y_t = C_t * h_t       (adj_y is adjoint of y)                     *
     * ------------------------------------------------------------------ */
    memset(scan_dA,     0, state_size * sizeof(float));
    memset(scan_ddelta, 0, seq_len    * sizeof(float));
    memset(adj_h,       0, state_size * sizeof(float));

    /* adj_prev_Bu: adjoint of prev_Bu that we need to carry backward */
    float *adj_prev_Bu = (float *)calloc(state_size, sizeof(float));
    float *scan_dlambda = (float *)calloc(seq_len, sizeof(float)); /* d_loss/d_lambda_t */
    /* d_h_rot: scratch buffer reused across timesteps (avoids malloc in tight loop) */
    float *d_h_rot = (float *)malloc(state_size * sizeof(float));
    if (!adj_prev_Bu || !scan_dlambda || !d_h_rot) {
        free(adj_prev_Bu); free(scan_dlambda); free(d_h_rot);
        goto cleanup_bwd;
    }

    for (long t = (long)seq_len - 1; t >= 0; t--) {
        float dt_t    = ws->delta[t];
        float lam_t   = store->lambda ? store->lambda[t] : 0.5f;

        /* acc for d_lambda_t (scalar per timestep, across dims) */
        float d_lambda_t = 0.0f;

        for (size_t d = 0; d < state_size; d++) {
            size_t td = (size_t)t * state_size + d;
            float h_t_d   = store->x[td];
            float a_diag  = store->alpha ? store->alpha[td] : store->A_diag[td];
            float h_rot_d = store->h_rot[td];
            float bu_t    = store->Bu ? store->Bu[td] : 0.0f;
            float bu_prev = (t > 0 && store->Bu) ? store->Bu[((size_t)t-1)*state_size+d] : 0.0f;
            float a_log_d = block->A_log.data[d];
            if (a_log_d > -1e-5f) a_log_d = -1e-5f;

            float beta_t  = (1.0f - lam_t) * dt_t * a_diag;
            float gamma_t = lam_t * dt_t;

            /* adj_h[d] = future-state grad + C output grad (already accumulated in adj_y) */
            float ah_actual = adj_h[d] + adj_y[td];

            /* Gradient for C_t: MIMO dC_t[n,r] = adj_y_R[t,r] * h_t[n]
             * stored in dC_seq[t * N*R + r*N + d] */
            for (size_t r = 0; r < R_rank; r++)
                dC_seq[t * state_size * R_rank + r * state_size + d] =
                    adj_y_R[t * R_rank + r] * h_t_d;

            /* Gradient for Bu_t: gamma term (h_t) + beta contribution from h_{t+1}
             * adj_prev_Bu[d] holds ah_{t+1} * beta_{t+1} from the previous iteration */
            float d_bu_t = ah_actual * gamma_t + adj_prev_Bu[d];

            /* MIMO dB_t[n,r] = d_bu_t * u_t[r],   scan_du[t,r] += d_bu_t * B_t[n,r]
             * B_seq[t * N*R + r*N + d] = B_t[d,r] */
            for (size_t r = 0; r < R_rank; r++) {
                float b_nr = ws->scan_B[t * state_size * R_rank + r * state_size + d];
                float u_r  = store->u_seq[t * R_rank + r];
                dB_seq[t * state_size * R_rank + r * state_size + d] = d_bu_t * u_r;
                scan_du[t * R_rank + r] += d_bu_t * b_nr;
            }

            /* Gradient for A_log[d] via alpha_t */
            scan_dA[d] += ah_actual * dt_t * a_diag * h_rot_d;

            /* d_lambda_t: lambda affects both beta and gamma */
            d_lambda_t += ah_actual * ((-dt_t * a_diag) * bu_prev + dt_t * bu_t);

            /* Gradient for delta_t (through alpha_t and beta_t and gamma_t) */
            scan_ddelta[(size_t)t] += ah_actual * (a_log_d * a_diag * h_rot_d
                                       + (1.0f - lam_t) * a_diag * bu_prev
                                       + (1.0f - lam_t) * dt_t * a_log_d * a_diag * bu_prev
                                       + lam_t * bu_t);

            /* Propagate adj_Bu_{t-1}: contribution from "beta_t * Bu_{t-1}" in h_t */
            adj_prev_Bu[d] = ah_actual * beta_t;

            /* d_h_rot[d] = ah_actual * alpha_t (for theta gradient + R^T adj propagation) */
            d_h_rot[d] = ah_actual * a_diag;
        }

        /* accumulate d_lambda into scan_dlambda */
        scan_dlambda[t] = d_lambda_t;

        /* Gradient for theta and adj_h through R(θ) */
        {
            float *h_prev_vec = (t > 0) ? &store->x[((size_t)t - 1) * state_size] : NULL;

            for (size_t i = 0; i + 1 < state_size; i += 2) {
                float th = theta ? theta[i >> 1] : 0.0f;
                float c = cosf(th), s_val = sinf(th);
                float hp0 = h_prev_vec ? h_prev_vec[i]   : 0.0f;
                float hp1 = h_prev_vec ? h_prev_vec[i+1] : 0.0f;
                float dr0 = d_h_rot[i], dr1 = d_h_rot[i+1];

                if (theta && s && s->g_theta)
                    s->g_theta[i >> 1] += dr0 * (-s_val * hp0 - c * hp1)
                                        + dr1 * (c * hp0 - s_val * hp1);

                /* adj_h = R^T * d_h_rot */
                adj_h[i]   = c * dr0 + s_val * dr1;
                adj_h[i+1] = -s_val * dr0 + c * dr1;
            }
            if (state_size & 1) adj_h[state_size-1] = d_h_rot[state_size-1];
        }
    }
    free(d_h_rot);

    /* Gradient for lambda_proj: g_lambda_proj += sum_t d_lambda_t * sigmoid'(raw_t) * x_t^T */
    {
        if (s && s->g_lambda_proj && block->lambda_proj.data) {
            float lam_raw_val;
            float *lam_raw = &lam_raw_val;
            for (size_t t = 0; t < seq_len; t++) {
                const float *x_t = &input_flat[t * dim];
                mb_matrix_vec_mult(lam_raw, &block->lambda_proj, x_t);
                float sig = 1.0f / (1.0f + expf(-lam_raw[0]));
                float dsig = sig * (1.0f - sig);
                float d_raw = scan_dlambda[t] * dsig;
                for (size_t k = 0; k < dim; k++)
                    s->g_lambda_proj[k] += d_raw * x_t[k];
                /* d_input[t] += d_raw * lambda_proj  (gradient through lambda path) */
                if (d_input_out)
                    for (size_t k = 0; k < dim; k++)
                        d_input_out[t * dim + k] += d_raw * block->lambda_proj.data[k];
            }
        }
    }

    cleanup_bwd:
    free(adj_prev_Bu);
    free(scan_dlambda);

    /* Accumulate A gradient */
    for (size_t i = 0; i < state_size; i++)
        s->g_A_log[i] += scan_dA[i];

    /* ---- Backward BCNorm pour B et C (per rank-slice) -------------------- *
     * MIMO: W_B is [N*R x dim]; BCNorm applied per rank-slice of N.          *
     * dB_seq layout: [L x N*R] with dB_seq[t*N*R + r*N + n] = dB_t[n,r]     *
     * -------------------------------------------------------------------- */
    {
        const float eps = 1e-6f;
        size_t NR = state_size * R_rank;
        /* Allocate z_B/z_C once outside the loop — reused each timestep */
        float *z_B = (float *)malloc(NR * sizeof(float));
        float *z_C = (float *)malloc(NR * sizeof(float));
        if (z_B && z_C) for (size_t t = 0; t < seq_len; t++) {
            /* Recompute z_B = W_B · x_t  [N*R] and  z_C = W_C · x_t [N*R] */
            mb_matrix_vec_mult(z_B, &block->W_B, &input_flat[t * dim]);
            mb_matrix_vec_mult(z_C, &block->W_C, &input_flat[t * dim]);

            /* Per rank-slice BCNorm backward */
            for (size_t r = 0; r < R_rank; r++) {
                float *zb_r  = z_B + r * state_size;
                float *dB_r  = dB_seq + t * NR + r * state_size;
                float *zc_r  = z_C + r * state_size;
                float *dC_r  = dC_seq + t * NR + r * state_size;

                /* g_b_B[r*N..] += dB_r */
                for (size_t d = 0; d < state_size; d++)
                    s->g_b_B[r * state_size + d] += dB_r[d];

                /* Backward RMSNorm pour B slice r */
                float rms_b = 0.0f;
                for (size_t d = 0; d < state_size; d++) rms_b += zb_r[d] * zb_r[d];
                rms_b = 1.0f / sqrtf(rms_b / (float)state_size + eps);
                float dot_b = 0.0f;
                for (size_t d = 0; d < state_size; d++) dot_b += dB_r[d] * zb_r[d];
                dot_b /= (float)state_size;
                for (size_t d = 0; d < state_size; d++)
                    dB_r[d] = dB_r[d] * rms_b - zb_r[d] * (rms_b*rms_b*rms_b) * dot_b;

                /* g_b_C[r*N..] += dC_r */
                for (size_t d = 0; d < state_size; d++)
                    s->g_b_C[r * state_size + d] += dC_r[d];

                /* Backward RMSNorm pour C slice r */
                float rms_c = 0.0f;
                for (size_t d = 0; d < state_size; d++) rms_c += zc_r[d] * zc_r[d];
                rms_c = 1.0f / sqrtf(rms_c / (float)state_size + eps);
                float dot_c = 0.0f;
                for (size_t d = 0; d < state_size; d++) dot_c += dC_r[d] * zc_r[d];
                dot_c /= (float)state_size;
                for (size_t d = 0; d < state_size; d++)
                    dC_r[d] = dC_r[d] * rms_c - zc_r[d] * (rms_c*rms_c*rms_c) * dot_c;
            }
        }
        free(z_B); free(z_C);
    }

    /* g_W_C += sum_t dC_t @ x_t^T  — dC_seq [L x N*R]^T @ input [L x dim] → [N*R x dim] */
    {
        size_t NR = state_size * R_rank;
        float *dC_seq_T = (float *)malloc(NR * seq_len * sizeof(float));
        if (dC_seq_T) {
            transpose_row_major(dC_seq, dC_seq_T, seq_len, NR);
            gemm_rowmajor(dC_seq_T, (float *)input_flat, s->g_W_C,
                          (int)NR, (int)seq_len, (int)dim);
            free(dC_seq_T);
        }
    }

    /* g_W_B += sum_t dB_t @ x_t^T — dB_seq [L x N*R]^T @ input [L x dim] → [N*R x dim] */
    {
        size_t NR = state_size * R_rank;
        float *dB_seq_T = (float *)malloc(NR * seq_len * sizeof(float));
        if (dB_seq_T) {
            transpose_row_major(dB_seq, dB_seq_T, seq_len, NR);
            gemm_rowmajor(dB_seq_T, (float *)input_flat, s->g_W_B,
                          (int)NR, (int)seq_len, (int)dim);
            free(dB_seq_T);
        }
    }

    /* d_input += dB_seq @ W_B + dC_seq @ W_C
     * x_t feeds W_B and W_C; their pre-BCNorm gradients (stored in dB_seq/dC_seq
     * after the BCNorm backward above) must be propagated back to d_input.
     * dB_seq [L x N*R] @ W_B [N*R x dim] → [L x dim]  (+=)
     * dC_seq [L x N*R] @ W_C [N*R x dim] → [L x dim]  (+=) */
    if (d_input_out) {
        size_t NR = state_size * R_rank;
        gemm_rowmajor(dB_seq, block->W_B.data, d_input_out,
                      (int)seq_len, (int)NR, (int)dim);
        gemm_rowmajor(dC_seq, block->W_C.data, d_input_out,
                      (int)seq_len, (int)NR, (int)dim);
    }

    /* Delta and W_in gradients
     * scan_du: [L x R] — grad w.r.t. u_t ∈ R^R
     * W_in: [R x dim], W_in grad: scan_du_silu^T @ input [R x dim]
     */
    for (size_t t = 0; t < seq_len; t++) {
        const float *x_input_t = &input_flat[t * dim];
        float ddt_t = scan_ddelta[t];

        /* delta_proj gradient + d_input contribution */
        {
            float raw_t = 0.0f;
            for (size_t k = 0; k < dim; k++)
                raw_t += block->delta_proj.data[k] * x_input_t[k];
            float draw = ddt_t * scalar_sigmoid(raw_t);
            for (size_t k = 0; k < dim; k++)
                s->g_delta_proj[k] += draw * x_input_t[k];
            /* d_input[t] += draw * delta_proj  (gradient through delta path) */
            if (d_input_out)
                for (size_t k = 0; k < dim; k++)
                    d_input_out[t * dim + k] += draw * block->delta_proj.data[k];
        }

        /* W_in gradient: z = W_in @ x_t, u_t = SiLU(z), du = scan_du[t] ∈ R^R
         * dz[r] = scan_du[t,r] * SiLU'(z[r])
         * scan_out reused as [L x R] buffer for dz */
        mb_matrix_vec_mult(z, &block->W_in, x_input_t);
        for (size_t r = 0; r < R_rank; r++) {
            float sig = scalar_sigmoid(z[r]);
            float dz = sig * (1.0f + z[r] * (1.0f - sig));
            scan_out[t * R_rank + r] = scan_du[t * R_rank + r] * dz;
        }
    }

    /* g_W_in += scan_out^T @ input  — [R x L] @ [L x dim] → [R x dim] */
    transpose_row_major(scan_out, contrib_T, seq_len, R_rank);
    gemm_rowmajor(contrib_T, (float *)input_flat, s->g_W_in,
                  (int)R_rank, (int)seq_len, (int)dim);

    /* d_input = scan_out @ W_in  — [L x R] @ [R x dim] → [L x dim] */
    if (d_input_out) {
        gemm_rowmajor(scan_out, block->W_in.data, d_input_out,
                      (int)seq_len, (int)R_rank, (int)dim);
        /* Residual gradient: d_input += dY (identity path) */
        for (size_t i = 0; i < seq_len * dim; i++)
            d_input_out[i] += dY[i];
    }

    free(scan_out); free(dY_T); free(z); free(adj_y_R); free(adj_y);
    free(scan_du); free(scan_dA); free(scan_ddelta); free(contrib_T);
    free(adj_h); free(dB_seq); free(dC_seq);
}

/* ============================================================================
 * Backward entrypoint
 * ============================================================================ */

static void mamba_backward_ws_impl(MambaBlock *block, MambaBlockWorkspace *ws,
                                   const float *dY, const float *input,
                                   float *d_input, MBOptimState *s_ext);

void mamba_backward_ws(MambaBlock *block, MambaBlockWorkspace *ws,
                       const float *dY, const float *input,
                       float *d_input, size_t batch_index) {
    (void)batch_index;
    mamba_backward_ws_impl(block, ws, dY, input, d_input, NULL);
}

void mamba_backward_ws_local(MambaBlock *block, MambaBlockWorkspace *ws,
                              const float *dY, const float *input,
                              float *d_input, size_t batch_index,
                              MBOptimState *local_grad) {
    (void)batch_index;
    mamba_backward_ws_impl(block, ws, dY, input, d_input, local_grad);
}

static void mamba_backward_ws_impl(MambaBlock *block, MambaBlockWorkspace *ws,
                                   const float *dY, const float *input,
                                   float *d_input, MBOptimState *s_ext) {
    size_t seq_len = block->config.seq_len;
    size_t state_size = block->config.state_size;

    if (!block || !ws || !workspace_matches_block(block, ws)) return;

    MBMatrix *A_bar = mb_matrix_create(state_size, state_size);
    for (size_t i = 0; i < state_size; i++) A_bar->data[i * state_size + i] = block->A_log.data[i];

    ForwardStore store;
    memset(&store, 0, sizeof(store));

    size_t dim    = block->config.dim;
    size_t R_rank = _mimo_R(&block->config);

    /* u_seq: [L x R] — MIMO input from W_in */
    float *u_seq  = (float *)calloc(seq_len * R_rank, sizeof(float));
    /* B_seq/C_seq: [L x N*R] — BCNorm'd B/C matrices per timestep */
    float *B_seq  = (float *)malloc(seq_len * state_size * R_rank * sizeof(float));
    float *C_seq  = (float *)malloc(seq_len * state_size * R_rank * sizeof(float));
    if (!u_seq || !B_seq || !C_seq) {
        free(u_seq); free(B_seq); free(C_seq); mb_matrix_free(A_bar); return;
    }

    float *tmp_delta = (float *)calloc(block->delta_proj.rows ? block->delta_proj.rows : 1,
                                        sizeof(float));
    size_t z_size = (R_rank > state_size) ? R_rank : state_size;
    float *z = (float *)malloc(z_size * sizeof(float));
    if (!tmp_delta || !z) {
        free(z); free(tmp_delta); free(u_seq); free(B_seq);
        mb_matrix_free(A_bar);
        return;
    }

    {
        const float eps = 1e-6f;
        size_t NR = state_size * R_rank;
        for (size_t t = 0; t < seq_len; t++) {
            const float *x_t = &input[t * dim];
            project_controller(block, x_t, z, &u_seq[t * R_rank]);
            ws->delta[t] = project_delta_value(block, x_t, tmp_delta, t, seq_len);

            /* B_seq[t] = BCNorm(W_B · x_t) per rank-slice [N*R] */
            float *b_t = &B_seq[t * NR];
            float *c_t = &C_seq[t * NR];
            mb_matrix_vec_mult(b_t, &block->W_B, x_t);
            mb_matrix_vec_mult(c_t, &block->W_C, x_t);
            for (size_t r = 0; r < R_rank; r++) {
                float *b_r = b_t + r * state_size;
                float *c_r = c_t + r * state_size;
                float rms = 0.0f;
                float rms_c = 0.0f;
                for (size_t d = 0; d < state_size; d++) {
                    rms += b_r[d] * b_r[d];
                    rms_c += c_r[d] * c_r[d];
                }
                rms = 1.0f / sqrtf(rms / (float)state_size + eps);
                rms_c = 1.0f / sqrtf(rms_c / (float)state_size + eps);
                for (size_t d = 0; d < state_size; d++)
                    b_r[d] = b_r[d] * rms + block->b_B[r * state_size + d];
                for (size_t d = 0; d < state_size; d++)
                    c_r[d] = c_r[d] * rms_c + block->b_C[r * state_size + d];
            }
        }
    }
    free(z);
    free(tmp_delta);

    selective_scan_forward_store(&store, ws->hidden, u_seq, ws->delta,
                                A_bar, B_seq, C_seq, &block->W_C, 0.0f,
                                block->theta,
                                input, &block->lambda_proj,
                                seq_len, state_size, dim, R_rank);

    selective_scan_backward(&store, block, ws, dY, input, d_input,
                            block->theta, seq_len, state_size, R_rank, s_ext);

    free(store.x); free(store.A_diag); free(store.B_bar); free(store.u_seq);
    if (store.h_rot)  free(store.h_rot);
    if (store.Bu)     free(store.Bu);
    if (store.C_seq)  free(store.C_seq);
    if (store.lambda) free(store.lambda);
    if (store.alpha)  free(store.alpha);
    free(u_seq); free(B_seq); free(C_seq);
    mb_matrix_free(A_bar);
}

void mamba_backward(MambaBlock *block, const float *dY, const float *input,
                    float *d_input, size_t batch_index) {
    MambaBlockWorkspace ws;
    if (!block) return;
    bind_default_workspace(&ws, block);
    mamba_backward_ws(block, &ws, dY, input, d_input, batch_index);
}

/* ============================================================================
 * Forward pass — 1D
 * ============================================================================ */

void mamba_block_forward_ws(MambaBlock *block, MambaBlockWorkspace *ws,
                            float *output, const float *input,
                            size_t batch_size) {
    if (!block || !ws || !output || !input || !workspace_matches_block(block, ws)) return;

    size_t seq_len = block->config.seq_len;
    size_t dim = block->config.dim;
    size_t state_size = block->config.state_size;

    for (size_t b = 0; b < batch_size; b++) {
        const float *batch_input = &input[b * seq_len * dim];
        float *batch_output = &output[b * seq_len * dim];

        size_t R_rank = _mimo_R(&block->config);
        /* u_seq: [L x R] — MIMO input vectors (SiLU(W_in @ x_t) ∈ R^R) */
        float *u_seq = (float *)calloc(seq_len * R_rank, sizeof(float));
        /* z: scratch buffer of size max(state_size, R) for project_controller */
        float *z = (float *)malloc((R_rank > state_size ? R_rank : state_size) * sizeof(float));
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
            /* W_in is [R x dim] → output is R-dimensional */
            project_controller(block, x_t, z, &u_seq[t * R_rank]);
            ws->delta[t] = project_delta_value(block, x_t, tmp_delta, t, seq_len);
        }
        free(z);
        free(tmp_delta);
        /* scan_out: [L x R] — MIMO output (y_t ∈ R^R per timestep) */
        float *scan_out = (float *)malloc(seq_len * R_rank * sizeof(float));
        if (!scan_out) { free(u_seq); continue; }

        long L = (long)seq_len, D = (long)state_size;

        /* Data-dependent B/C with BCNorm + biases (Mamba-3)
         * MIMO: scan_B[t] ∈ R^{N*R}, scan_C[t] ∈ R^{N*R}
         * u_seq[t] ∈ R^R  (from W_in: [R x dim])
         * The BCNorm is applied independently to each rank-slice of B/C.
         */
        {
            const float eps = 1e-6f;
            long NR = (long)(state_size * R_rank);
            for (long t = 0; t < L; t++) {
                const float *x_t = &batch_input[t * dim];
                float *b_out = &ws->scan_B[t * NR];
                float *c_out = &ws->scan_C[t * NR];

                /* B_flat[N*R] = W_B[N*R x dim] @ x_t */
                mb_matrix_vec_mult(b_out, &block->W_B, x_t);
                mb_matrix_vec_mult(c_out, &block->W_C, x_t);

                /* RMSNorm + bias per rank-slice of B */
                for (size_t r = 0; r < R_rank; r++) {
                    float *b_r = b_out + r * state_size;
                    float rms_b = 0.0f;
                    for (long d = 0; d < D; d++) rms_b += b_r[d] * b_r[d];
                    rms_b = 1.0f / sqrtf(rms_b / (float)state_size + eps);
                    for (long d = 0; d < D; d++) b_r[d] = b_r[d] * rms_b + block->b_B[r * state_size + d];
                }
                /* RMSNorm + bias per rank-slice of C */
                for (size_t r = 0; r < R_rank; r++) {
                    float *c_r = c_out + r * state_size;
                    float rms_c = 0.0f;
                    for (long d = 0; d < D; d++) rms_c += c_r[d] * c_r[d];
                    rms_c = 1.0f / sqrtf(rms_c / (float)state_size + eps);
                    for (long d = 0; d < D; d++) c_r[d] = c_r[d] * rms_c + block->b_C[r * state_size + d];
                }

                for (long d = 0; d < D; d++)
                    ws->scan_delta[t*D + d] = ws->delta[t];
            }
        }
        memset(ws->scan_h, 0, (size_t)D * sizeof(float));

        /* Complex SSM scan with R(θ) rotation + exp-trapezoidal discretization
         * (Mamba-3 §3.1 + §3.2) with MIMO (rank R)
         * h_t = alpha_t * R(θ) * h_{t-1} + beta_t * Bu_{t-1} + gamma_t * Bu_t
         * Bu_t[n] = sum_r B_t[n,r] * u_t[r]     (MIMO: reduce over rank)
         * y_t[r]  = sum_n C_t[n,r] * h_t[n]     (MIMO: project to rank space)
         * output  = W_out[dim x R] @ y_t         (project back to dim)
         */
        {
            long NR = (long)(state_size * R_rank);
            float *h_cur    = ws->scan_h;
            float *h_rot    = (float *)malloc((size_t)D * sizeof(float));
            float *prev_Bu  = (float *)calloc((size_t)D, sizeof(float));
            float *Bu_cur   = (float *)malloc((size_t)D * sizeof(float));
            float *lam_buf  = (float *)malloc(sizeof(float));
            /* scan_out: [L x R] — y_t vectors before W_out projection */
            float *y_rank   = (float *)malloc(seq_len * R_rank * sizeof(float));
            if (!h_rot || !prev_Bu || !lam_buf || !Bu_cur || !y_rank) {
                free(h_rot); free(prev_Bu); free(lam_buf); free(Bu_cur); free(y_rank);
                free(u_seq); free(scan_out); continue;
            }

            for (long t = 0; t < L; t++) {
                float dt_t = ws->delta[t];
                const float *x_t = &batch_input[t * dim];
                const float *u_t = &u_seq[t * R_rank];     /* u_t ∈ R^R */
                const float *b_t = &ws->scan_B[t * NR]; /* B_t[N*R] */
                const float *c_t = &ws->scan_C[t * NR]; /* C_t[N*R] */

                /* lambda_t = sigmoid(lambda_proj · x_t) */
                mb_matrix_vec_mult(lam_buf, &block->lambda_proj, x_t);
                float lam_t = scalar_sigmoid(lam_buf[0]);

                /* Apply R(θ) to h_cur → h_rot */
                for (long i = 0; i + 1 < D; i += 2) {
                    float th = block->theta[i >> 1];
                    float c = cosf(th), sv = sinf(th);
                    float h0 = h_cur[i], h1 = h_cur[i+1];
                    h_rot[i]   = c*h0 - sv*h1;
                    h_rot[i+1] = sv*h0 + c*h1;
                }
                if (D & 1) h_rot[D-1] = h_cur[D-1];

                /* MIMO Bu_t[n] = sum_r B_t[n,r] * u_t[r]
                 * b_t layout: [R slices of N] i.e. b_t[r*N + n] = B_t[n,r] */
                for (long n = 0; n < D; n++) {
                    float bu = 0.0f;
                    for (size_t r = 0; r < R_rank; r++)
                        bu += b_t[r * state_size + n] * u_t[r];
                    Bu_cur[n] = bu;
                }

                /* 3-term recurrence */
                for (long d = 0; d < D; d++) {
                    float a = block->A_log.data[d];
                    if (a > -1e-5f) a = -1e-5f;
                    float alpha  = expf(dt_t * a);
                    float beta   = (1.0f - lam_t) * dt_t * alpha;
                    float gamma_ = lam_t * dt_t;
                    h_cur[d] = alpha * h_rot[d] + beta * prev_Bu[d] + gamma_ * Bu_cur[d];
                    prev_Bu[d] = Bu_cur[d];
                }

                /* MIMO y_t[r] = sum_n C_t[n,r] * h_t[n]
                 * c_t layout: [R slices of N] i.e. c_t[r*N + n] = C_t[n,r] */
                for (size_t r = 0; r < R_rank; r++) {
                    float yr = 0.0f;
                    for (long n = 0; n < D; n++)
                        yr += c_t[r * state_size + n] * h_cur[n];
                    y_rank[t * R_rank + r] = yr;
                }
                /* Also store in scan_out (first-R) for backward compat with W_out gradient */
                for (size_t r = 0; r < R_rank; r++)
                    scan_out[t * R_rank + r] = y_rank[t * R_rank + r];
            }
            free(h_rot); free(prev_Bu); free(lam_buf); free(Bu_cur); free(y_rank);
        }
        memcpy(ws->hidden, ws->scan_h, (size_t)D * sizeof(float));

        float *ybuf = (float *)malloc(dim * sizeof(float));
        if (ybuf) {
            for (size_t t = 0; t < seq_len; t++) {
                /* y_t ∈ R^R, project to dim via W_out[dim x R] */
                const float *y_t = &scan_out[t * R_rank];
                mb_matrix_vec_mult(ybuf, &block->W_out, y_t);
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

void mamba_block_forward(MambaBlock *block, float *output, const float *input,
                         size_t batch_size) {
    MambaBlockWorkspace ws;
    if (!block) return;
    bind_default_workspace(&ws, block);
    mamba_block_forward_ws(block, &ws, output, input, batch_size);
}
