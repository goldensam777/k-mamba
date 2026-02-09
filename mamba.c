#include "mamba.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Forward storage for training (per timestep) */
typedef struct {
    real_t *x;         /* seq_len x state_size */
    real_t *A_diag;    /* seq_len x state_size */
    real_t *B_bar;     /* seq_len x state_size */
    real_t *u_seq;     /* seq_len x state_size (controller vectors per timestep) */
} ForwardStore;

/* Global registry mapping MambaBlock -> OptimState */
static OptimState *g_opt_states[256];
static MambaBlock *g_opt_blocks[256];
static size_t g_opt_n = 0;
/* forward declarations */
static void _mamba_free_opt_for(MambaBlock *block);
static OptimState* _find_opt(MambaBlock *block);

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

Matrix* matrix_create(size_t rows, size_t cols) {
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    m->data = (real_t *)calloc(rows * cols, sizeof(real_t));
    
    if (!m->data) {
        free(m);
        return NULL;
    }
    
    return m;
}

void matrix_free(Matrix *m) {
    if (!m) return;
    if (m->data) free(m->data);
    free(m);
}

void matrix_zero(Matrix *m) {
    if (!m || !m->data) return;
    memset(m->data, 0, m->rows * m->cols * sizeof(real_t));
}

void matrix_copy(Matrix *dst, const Matrix *src) {
    if (!dst || !src || !dst->data || !src->data) return;
    if (dst->rows != src->rows || dst->cols != src->cols) return;
    
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(real_t));
}

void matrix_print(const Matrix *m) {
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

void matrix_vec_mult(real_t *out, const Matrix *m, const real_t *v) {
    if (!out || !m || !v || !m->data) return;
    
    memset(out, 0, m->rows * sizeof(real_t));
    
    for (size_t i = 0; i < m->rows; i++) {
        real_t sum = 0.0;
        for (size_t j = 0; j < m->cols; j++) {
            sum += m->data[i * m->cols + j] * v[j];
        }
        out[i] = sum;
    }
}

void vec_add(real_t *y, const real_t *x, size_t n) {
    if (!y || !x) return;
    for (size_t i = 0; i < n; i++) {
        y[i] += x[i];
    }
}

void vec_scale(real_t *v, real_t alpha, size_t n) {
    if (!v) return;
    for (size_t i = 0; i < n; i++) {
        v[i] *= alpha;
    }
}

/* ============================================================================
 * Activation Functions
 * ============================================================================ */

real_t softplus(real_t x) {
    if (x > 20.0f) return x;  /* Avoid overflow */
    if (x < -20.0f) return 0.0f;  /* Avoid underflow */
    return logf(1.0f + expf(x));
}

real_t sigmoid(real_t x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

real_t relu(real_t x) {
    return x > 0.0f ? x : 0.0f;
}

/* ============================================================================
 * Discretization Functions
 * ============================================================================ */

/* Compute matrix exponential using Taylor series for small matrices
 * For diagonal matrices: exp(dt*A) = diag(exp(dt*a_ii))
 * Assumes A is provided as diagonal elements
 */
void discretize_A(Matrix *A_bar, const Matrix *A, real_t dt) {
    if (!A_bar || !A) return;
    
    /* For a diagonal state matrix (common in SSMs) */
    for (size_t i = 0; i < A_bar->rows && i < A->rows; i++) {
        real_t a_ii = A->data[i * A->cols + i];
        A_bar->data[i * A_bar->cols + i] = expf(dt * a_ii);
    }
}

/* Compute discretized B using forward Euler for simplicity
 * More accurate: B_bar = integral_0^dt exp(tau*A) d(tau) * B
 * Simplified: B_bar = dt * B (for small dt)
 */
void discretize_B(real_t *B_bar, const Matrix *A, const real_t *B, 
                  real_t dt, size_t state_size) {
    if (!B_bar || !A || !B) return;
    
    /* Simple approximation: scale B by dt */
    for (size_t i = 0; i < state_size; i++) {
        B_bar[i] = dt * B[i];
    }
}

/* ============================================================================
 * Selective Scan - Core Mamba Operation
 * ============================================================================ */

void selective_scan(real_t *output, real_t *state, 
                   const real_t *input, const real_t *delta,
                   const Matrix *A_bar, const real_t *B_bar,
                   const Matrix *C, real_t D,
                   size_t seq_len, size_t state_size) {
    
    if (!output || !state || !input || !delta || !A_bar || !B_bar || !C) {
        return;
    }
    (void)C; (void)D; /* currently unused: readout handled outside */
    
    /* Initialize state to zero */
    memset(state, 0, state_size * sizeof(real_t));

    /* Allocate temporaries once to avoid per-timestep malloc/free */
    real_t *temp_state = (real_t *)malloc(state_size * sizeof(real_t));
    real_t *A_diag_t = (real_t *)malloc(state_size * sizeof(real_t));
    real_t *B_bar_t = (real_t *)malloc(state_size * sizeof(real_t));
    if (!temp_state || !A_diag_t || !B_bar_t) {
        free(temp_state); free(A_diag_t); free(B_bar_t);
        return;
    }

    /* Process each timestep; input is flattened controller vectors (seq_len x state_size) */
    for (size_t t = 0; t < seq_len; t++) {
        const real_t *u_t = &input[t * state_size];
        real_t dt_t = delta[t];

        /* Compute A_diag_t and B_bar_t in parallel across state dimensions */
#pragma omp parallel for
        for (size_t i = 0; i < state_size; i++) {
            real_t a_val = A_bar->data[i * state_size + i];
            real_t a_diag = expf(dt_t * a_val);
            A_diag_t[i] = a_diag;
            if (fabsl(a_val) < 1e-8) {
                B_bar_t[i] = dt_t * B_bar[i];
            } else {
                B_bar_t[i] = (a_diag - 1.0f) / a_val * B_bar[i];
            }
        }

        memcpy(temp_state, state, state_size * sizeof(real_t));

        /* Update state elementwise in parallel: x_t[i] = A_diag_t[i] * x_{t-1}[i] + B_bar_t[i] * u_t[i] */
#pragma omp parallel for
        for (size_t i = 0; i < state_size; i++) {
            state[i] = A_diag_t[i] * temp_state[i] + B_bar_t[i] * u_t[i];
        }

        /* Write state vector into output buffer (flattened) */
#pragma omp parallel for
        for (size_t i = 0; i < state_size; i++) output[t * state_size + i] = state[i];
    }

    free(A_diag_t); free(B_bar_t); free(temp_state);
}

/* ============================================================================
 * Mamba Block Operations
 * ============================================================================ */

MambaBlock* mamba_block_create(const MambaConfig *config) {
    if (!config) return NULL;
    
    MambaBlock *block = (MambaBlock *)malloc(sizeof(MambaBlock));
    if (!block) return NULL;
    
    block->config = *config;
    
    /* Allocate matrices */
     /* W_in: state_size x dim  (maps input -> controller vector)
         W_out: dim x state_size (maps state -> output) */
     block->W_in = *matrix_create(config->state_size, config->dim);
     block->W_out = *matrix_create(config->dim, config->state_size);
    
    block->A_log = *matrix_create(config->state_size, 1);
    block->B_mat = *matrix_create(config->state_size, 1);
    block->C_mat = *matrix_create(config->state_size, 1);
    
    block->delta_proj = *matrix_create(1, config->dim);
    
    /* Allocate temporary buffers */
    block->hidden = (real_t *)calloc(config->state_size, sizeof(real_t));
    block->delta = (real_t *)calloc(config->seq_len, sizeof(real_t));
    
    if (!block->W_in.data || !block->W_out.data || !block->A_log.data ||
        !block->B_mat.data || !block->C_mat.data || !block->delta_proj.data ||
        !block->hidden || !block->delta) {
        mamba_block_free(block);
        return NULL;
    }
    
    return block;
}

void mamba_block_free(MambaBlock *block) {
    if (!block) return;
    
    if (block->W_in.data) free(block->W_in.data);
    if (block->W_out.data) free(block->W_out.data);
    if (block->A_log.data) free(block->A_log.data);
    if (block->B_mat.data) free(block->B_mat.data);
    if (block->C_mat.data) free(block->C_mat.data);
    if (block->delta_proj.data) free(block->delta_proj.data);
    if (block->hidden) free(block->hidden);
    if (block->delta) free(block->delta);
    /* free optimizer/grad buffers if present */
    /* Those are managed by mamba_free_optimizer through block pointer if used */
    
    free(block);
}

/* Initialize Mamba block with standard parameters */
void mamba_block_init(MambaBlock *block) {
    if (!block) return;
    
    /* Initialize A with stable values (negative real parts for stability) */
    for (size_t i = 0; i < block->config.state_size; i++) {
        /* Logarithmically spaced values for numerical stability */
        real_t spacing = (real_t)(i + 1) / (real_t)block->config.state_size;
        block->A_log.data[i] = -expf(spacing * logf(block->config.dt_scale));
    }
    
    /* Initialize B uniformly */
    for (size_t i = 0; i < block->config.state_size; i++) {
        block->B_mat.data[i] = 1.0f / sqrtf((real_t)block->config.state_size);
    }
    
    /* Initialize C uniformly */
    for (size_t i = 0; i < block->config.state_size; i++) {
        block->C_mat.data[i] = 1.0f / sqrtf((real_t)block->config.state_size);
    }
    
    /* Initialize projections with small random values */
    /* initialize W_in (state_size x dim) */
    for (size_t i = 0; i < block->W_in.rows * block->W_in.cols; i++) {
        block->W_in.data[i] = ((real_t)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    /* initialize W_out (dim x state_size) */
    for (size_t i = 0; i < block->W_out.rows * block->W_out.cols; i++) {
        block->W_out.data[i] = ((real_t)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    for (size_t i = 0; i < block->delta_proj.rows * block->delta_proj.cols; i++) {
        block->delta_proj.data[i] = ((real_t)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
}

void compute_delta(real_t *delta_out, const MambaBlock *block, 
                   const real_t *delta_in, size_t seq_len) {
    if (!delta_out || !block || !delta_in) return;
    
    /* Apply softplus to ensure positive deltas */
    for (size_t i = 0; i < seq_len; i++) {
        real_t delta_val = delta_in[i];
        delta_val = softplus(delta_val);
        
        /* Clamp to valid range */
        if (delta_val < block->config.dt_min) {
            delta_val = block->config.dt_min;
        }
        if (delta_val > block->config.dt_max) {
            delta_val = block->config.dt_max;
        }
        
        delta_out[i] = delta_val;
    }
}

/* Allocate and attach optimizer state to block */
void mamba_attach_optimizer(MambaBlock *block, const OptimConfig *optconf) {
    if (!block) return;
    OptimState *s = (OptimState *)malloc(sizeof(OptimState));
    size_t dim = block->config.dim;
    size_t state = block->config.state_size;
    size_t size_in = state * dim;       /* W_in size */
    size_t size_out = dim * state;      /* W_out size */
    memset(s, 0, sizeof(OptimState));

    /* allocate gradient buffers */
    s->g_W_in = (real_t *)calloc(size_in, sizeof(real_t));
    s->g_W_out = (real_t *)calloc(size_out, sizeof(real_t));
    s->g_A_log = (real_t *)calloc(state, sizeof(real_t));
    s->g_B_mat = (real_t *)calloc(state, sizeof(real_t));
    s->g_C_mat = (real_t *)calloc(state, sizeof(real_t));
    s->g_delta_proj = (real_t *)calloc(dim, sizeof(real_t));

    /* optimizer moments */
    s->m_W_in = (real_t *)calloc(size_in, sizeof(real_t)); s->v_W_in = (real_t *)calloc(size_in, sizeof(real_t));
    s->m_W_out = (real_t *)calloc(size_out, sizeof(real_t)); s->v_W_out = (real_t *)calloc(size_out, sizeof(real_t));
    s->m_A_log = (real_t *)calloc(state, sizeof(real_t)); s->v_A_log = (real_t *)calloc(state, sizeof(real_t));
    s->m_B_mat = (real_t *)calloc(state, sizeof(real_t)); s->v_B_mat = (real_t *)calloc(state, sizeof(real_t));
    s->m_C_mat = (real_t *)calloc(state, sizeof(real_t)); s->v_C_mat = (real_t *)calloc(state, sizeof(real_t));
    s->m_delta_proj = (real_t *)calloc(dim, sizeof(real_t)); s->v_delta_proj = (real_t *)calloc(dim, sizeof(real_t));

    s->step = 0;
    s->step = 0;
    /* register in global registry */
    if (g_opt_n < 256) { g_opt_blocks[g_opt_n] = block; g_opt_states[g_opt_n] = s; g_opt_n++; }
    else free(s);
    (void)optconf; /* unused for now */
}

/* free optimizer state (best-effort) */
void mamba_free_optimizer(MambaBlock *block) {
    /* find in global registry and free */
    _mamba_free_opt_for(block);
}

/* internal helper to free optimizer map entry */
void _mamba_free_opt_for(MambaBlock *block) {
    for (size_t i = 0; i < g_opt_n; i++) {
        if (g_opt_blocks[i] == block) {
            OptimState *s = g_opt_states[i];
            if (!s) return;
            free(s->g_W_in); free(s->g_W_out); free(s->g_A_log); free(s->g_B_mat); free(s->g_C_mat); free(s->g_delta_proj);
            free(s->m_W_in); free(s->v_W_in); free(s->m_W_out); free(s->v_W_out);
            free(s->m_A_log); free(s->v_A_log); free(s->m_B_mat); free(s->v_B_mat);
            free(s->m_C_mat); free(s->v_C_mat); free(s->m_delta_proj); free(s->v_delta_proj);
            free(s);
            /* remove entry */
            for (size_t j = i; j + 1 < g_opt_n; j++) { g_opt_blocks[j] = g_opt_blocks[j+1]; g_opt_states[j] = g_opt_states[j+1]; }
            g_opt_n--;
            return;
        }
    }
}

/* zero gradients (best-effort using global map) */
void mamba_zero_grads(MambaBlock *block) {
    for (size_t i = 0; i < g_opt_n; i++) {
        if (g_opt_blocks[i] == block) {
            OptimState *s = g_opt_states[i];
            size_t dim = block->config.dim; size_t state = block->config.state_size;
            size_t size_in = state * dim; size_t size_out = dim * state;
            memset(s->g_W_in, 0, size_in * sizeof(real_t)); memset(s->g_W_out, 0, size_out * sizeof(real_t));
            memset(s->g_A_log, 0, state * sizeof(real_t)); memset(s->g_B_mat, 0, state * sizeof(real_t));
            memset(s->g_C_mat, 0, state * sizeof(real_t)); memset(s->g_delta_proj, 0, dim * sizeof(real_t));
            return;
        }
    }
}

/* Helper: find optimizer state for block */
OptimState* _find_opt(MambaBlock *block) {
    for (size_t i = 0; i < g_opt_n; i++) if (g_opt_blocks[i] == block) return g_opt_states[i];
    return NULL;
}

/* Simple MUONCLIP optimizer step operating on each parameter buffer using its stored moments */
void mamba_optimizer_step(MambaBlock *block, const OptimConfig *conf) {
    OptimState *s = _find_opt(block);
    if (!s) return;
    s->step += 1;
    real_t lr = conf->lr; real_t mu = conf->mu; real_t beta2 = conf->beta2; real_t eps = conf->eps; real_t clip = conf->clip_norm; real_t wd = conf->weight_decay;

    size_t dim = block->config.dim; size_t state = block->config.state_size;
    size_t size_in = state * dim; size_t size_out = dim * state;

    /* helper macro to update a parameter buffer */
#define MUONCLIP_UPDATE(param, grad, m, v, N) do { \
    /* compute global norm and clip */ \
    double sq = 0.0; for (size_t _i=0; _i < (N); _i++) { double g = (double)(grad[_i]); sq += g*g; } \
    double gn = sqrt(sq); double scale = 1.0; if (gn > clip && clip>0.0) scale = clip / gn; \
    for (size_t _i=0; _i < (N); _i++) { real_t g = grad[_i] * (real_t)scale + wd * param[_i]; \
        m[_i] = mu * m[_i] + (1.0f - mu) * g; \
        v[_i] = beta2 * v[_i] + (1.0f - beta2) * (g * g); \
        real_t m_hat = m[_i] / (1.0f - powf(mu, (real_t)s->step)); \
        real_t v_hat = v[_i] / (1.0f - powf(beta2, (real_t)s->step)); \
        param[_i] -= lr * m_hat / (sqrtf(v_hat) + eps); } \
    } while (0)

    MUONCLIP_UPDATE(block->W_in.data, s->g_W_in, s->m_W_in, s->v_W_in, size_in);
    MUONCLIP_UPDATE(block->W_out.data, s->g_W_out, s->m_W_out, s->v_W_out, size_out);
    MUONCLIP_UPDATE(block->A_log.data, s->g_A_log, s->m_A_log, s->v_A_log, state);
    MUONCLIP_UPDATE(block->B_mat.data, s->g_B_mat, s->m_B_mat, s->v_B_mat, state);
    MUONCLIP_UPDATE(block->C_mat.data, s->g_C_mat, s->m_C_mat, s->v_C_mat, state);
    MUONCLIP_UPDATE(block->delta_proj.data, s->g_delta_proj, s->m_delta_proj, s->v_delta_proj, dim);

#undef MUONCLIP_UPDATE
}

/* ------------------------------------------------------------------
 * Forward scan that stores per-timestep values for backward
 * ------------------------------------------------------------------ */
void selective_scan_forward_store(ForwardStore *store, real_t *state, 
                   const real_t *input, const real_t *delta,
                   const Matrix *A_bar, const real_t *B_bar,
                   const Matrix *C, real_t D,
                   size_t seq_len, size_t state_size) {
    if (!store || !state) return;
    (void)C; (void)D;
    /* allocate storage arrays */
    store->x = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
    store->A_diag = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
    store->B_bar = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
    store->u_seq = (real_t *)calloc(seq_len * state_size, sizeof(real_t));

    memset(state, 0, state_size * sizeof(real_t));

    /* allocate zero prev buffer to avoid per-step calloc/free */
    real_t *zero_prev = (real_t *)calloc(state_size, sizeof(real_t));

    for (size_t t = 0; t < seq_len; t++) {
        const real_t *u_t = &input[t * state_size];
        real_t dt_t = delta[t];
        /* copy controller vector */
#pragma omp parallel for
        for (size_t i = 0; i < state_size; i++) store->u_seq[t * state_size + i] = u_t[i];

        /* A_bar_t diagonal and B_bar per-dim (parallelizable) */
#pragma omp parallel for
        for (size_t i = 0; i < state_size; i++) {
            real_t a_val = A_bar->data[i * state_size + i];
            real_t a_diag_t = expf(dt_t * a_val);
            store->A_diag[t * state_size + i] = a_diag_t;
            if (fabsl(a_val) < 1e-8) store->B_bar[t * state_size + i] = dt_t * B_bar[i];
            else store->B_bar[t * state_size + i] = (a_diag_t - 1.0f) / a_val * B_bar[i];
        }

        /* update state (elementwise) */
        real_t *x_prev = (t == 0) ? zero_prev : &store->x[(t-1)*state_size];
        real_t *x_t = &store->x[t * state_size];
#pragma omp parallel for
        for (size_t i = 0; i < state_size; i++) {
            real_t a_diag_t = store->A_diag[t * state_size + i];
            real_t bbar = store->B_bar[t * state_size + i];
            x_t[i] = a_diag_t * x_prev[i] + bbar * u_t[i];
            state[i] = x_t[i];
        }
    }

    free(zero_prev);
}

/* Backward through stored forward trace. dY is scalar gradient per timestep. input is original inputs (batch's flattened seq*dim), and u_pooling assumed mean pooling done outside. */
void selective_scan_backward(ForwardStore *store, MambaBlock *block, const real_t *dY,
                             const real_t *input_flat, size_t seq_len, size_t state_size) {
    if (!store || !block) return;
    size_t dim = block->config.dim;
    /* find optimizer state */
    OptimState *s = _find_opt(block);
    if (!s) return;

    /* initialize adjoints for x_t */
    real_t *adj_x = (real_t *)calloc(seq_len * state_size, sizeof(real_t));

    /* First, map output gradients (dY) back into state space via W_out and accumulate grads for W_out */
    for (size_t t = 0; t < seq_len; t++) {
        for (size_t j = 0; j < dim; j++) {
            real_t dy = dY[t * dim + j];
            /* accumulate grad W_out: row j, cols over state_size */
            for (size_t i = 0; i < state_size; i++) {
                s->g_W_out[j * state_size + i] += dy * store->x[t * state_size + i];
                /* propagate to adjoint of state */
                adj_x[t * state_size + i] += dy * block->W_out.data[j * state_size + i];
            }
        }
    }

    /* Backprop through time (B and A) */
    for (int t = (int)seq_len - 1; t >= 0; t--) {
        real_t dt_t = block->delta[t];

        for (size_t i = 0; i < state_size; i++) {
            real_t ax = adj_x[t*state_size + i];
            if (ax == 0.0f) continue;

            /* grad B: B_mat (original B) receives ax * bcoef * u_i where bcoef = discretization coeff */
            real_t u_i = store->u_seq[t * state_size + i];
            real_t bbar = store->B_bar[t * state_size + i];
            real_t Bparam = block->B_mat.data[i];
            real_t bcoef = (fabsl(Bparam) > 0.0f) ? (bbar / Bparam) : dt_t;
            s->g_B_mat[i] += ax * bcoef * u_i;

            /* grad A_log: chain through A_bar_t and A_bar */
            real_t A_bar_i = expf(block->A_log.data[i]);
            real_t A_bar_t_i = store->A_diag[t * state_size + i];
            real_t x_prev = (t == 0) ? 0.0f : store->x[(t-1)*state_size + i];
            s->g_A_log[i] += ax * dt_t * A_bar_t_i * A_bar_i * x_prev;

            /* propagate to previous state */
            if (t > 0) adj_x[(t-1)*state_size + i] += ax * A_bar_t_i;
        }
    }

    /* Backprop into W_in from controller u_seq: each controller dimension j affects only state index j (elementwise) */
    for (size_t t = 0; t < seq_len; t++) {
        const real_t *x_input_t = &input_flat[t * dim];

        /* reconstruct pre-activation z = W_in @ x_input_t (length state_size) */
        real_t *z = (real_t *)malloc(state_size * sizeof(real_t));
        matrix_vec_mult(z, &block->W_in, x_input_t);

        for (size_t j = 0; j < state_size; j++) {
            /* derivative of loss wrt controller u_j */
            real_t bbar = store->B_bar[t * state_size + j]; /* dt_t * B[j] */
            real_t du = adj_x[t * state_size + j] * bbar; /* dL/du_j = ax * bbar */

            /* silu'(z) */
            real_t sig = sigmoid(z[j]);
            real_t dz = sig * (1.0f + z[j] * (1.0f - sig));
            real_t contrib = du * dz; /* no mean pooling */

            /* accumulate gradient into W_in row j */
            for (size_t k = 0; k < block->W_in.cols; k++) {
                s->g_W_in[j * block->W_in.cols + k] += contrib * x_input_t[k];
            }
        }
        free(z);
    }

    free(adj_x);
}

/* Backward entrypoint: compute gradients for a single batch element (batch_index unused here as we perform batch_size=1 in examples) */
void mamba_backward(MambaBlock *block, const real_t *dY, const real_t *input, size_t batch_index) {
    (void)batch_index;
    size_t seq_len = block->config.seq_len;
    size_t state_size = block->config.state_size;

    /* Re-run forward to capture stored traces (we could have kept them during forward in realistic implementation) */
    /* For simplicity call selective_scan_forward_store here using A_bar built from A_log and B from B_mat */
    Matrix *A_bar = matrix_create(state_size, state_size);
    for (size_t i = 0; i < state_size; i++) A_bar->data[i * state_size + i] = expf(block->A_log.data[i]);
    real_t *B_bar = (real_t *)malloc(state_size * sizeof(real_t));
    for (size_t i = 0; i < state_size; i++) B_bar[i] = block->B_mat.data[i];

    ForwardStore store;
    memset(&store, 0, sizeof(store));

    /* compute u_seq from inputs: vector controller of size state_size per timestep */
    size_t dim = block->config.dim;
    real_t *u_seq = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
    if (!u_seq) { matrix_free(A_bar); free(B_bar); return; }

    /* temporary buffer for delta projection (delta_proj has rows=1) */
    real_t *tmp_delta = (real_t *)calloc(block->delta_proj.rows, sizeof(real_t));

    for (size_t t = 0; t < seq_len; t++) {
        const real_t *x_t = &input[t * dim];
        /* z = W_in @ x_t  (length state_size) */
        real_t *z = (real_t *)calloc(state_size, sizeof(real_t));
        matrix_vec_mult(z, &block->W_in, x_t);
        for (size_t j = 0; j < state_size; j++) {
            real_t s = z[j] * sigmoid(z[j]);
            u_seq[t * state_size + j] = s;
        }
        free(z);

        /* compute delta from delta_proj: scalar projection then softplus+clamp */
        if (block->delta_proj.rows > 0) {
            matrix_vec_mult(tmp_delta, &block->delta_proj, x_t); /* tmp_delta length = delta_proj.rows (1) */
            real_t dval = softplus(tmp_delta[0]);
            if (dval < block->config.dt_min) dval = block->config.dt_min;
            if (dval > block->config.dt_max) dval = block->config.dt_max;
            block->delta[t] = dval;
        } else {
            block->delta[t] = block->config.dt_scale * ((real_t)t / (real_t)seq_len + 1.0f);
        }
    }
    free(tmp_delta);

    selective_scan_forward_store(&store, block->hidden, u_seq, block->delta, A_bar, B_bar, &block->C_mat, 0.0f, seq_len, state_size);

    /* call backward on stored trace */
    selective_scan_backward(&store, block, dY, input, seq_len, state_size);

    /* accumulate gradients from W_out if used: in forward we projected scan_out into output[0] slot only; handle W_out as mapping from scan outputs to output vector — for simplicity we compute gradient for W_out as zeros here */

    /* free store */
    free(store.x); free(store.A_diag); free(store.B_bar); free(store.u_seq); free(u_seq);
    matrix_free(A_bar); free(B_bar);
}

void mamba_forward(MambaBlock *block, real_t *output, const real_t *input, 
                   size_t batch_size) {
    if (!block || !output || !input) return;
    
    size_t seq_len = block->config.seq_len;
    size_t dim = block->config.dim;
    size_t state_size = block->config.state_size;
    
    /* Process each sequence in batch */
    for (size_t b = 0; b < batch_size; b++) {
        const real_t *batch_input = &input[b * seq_len * dim];
        real_t *batch_output = &output[b * seq_len * dim];
        
        /* Project input: compute vector controller u_seq (seq_len x state_size) */
        real_t *u_seq = (real_t *)calloc(seq_len * state_size, sizeof(real_t));
        if (!u_seq) continue;

        /* temporary for delta projection */
        real_t *tmp_delta = (real_t *)calloc(block->delta_proj.rows, sizeof(real_t));

        for (size_t t = 0; t < seq_len; t++) {
            const real_t *x_t = &batch_input[t * dim];
            /* z = W_in @ x_t -> length state_size */
            real_t *z = (real_t *)malloc(state_size * sizeof(real_t));
            matrix_vec_mult(z, &block->W_in, x_t);
            for (size_t j = 0; j < state_size; j++) {
                real_t s = z[j] * sigmoid(z[j]);
                u_seq[t * state_size + j] = s;
            }
            free(z);

            /* Compute delta for this timestep using delta_proj -> softplus + clamp */
            if (block->delta_proj.rows > 0) {
                matrix_vec_mult(tmp_delta, &block->delta_proj, x_t);
                real_t dval = softplus(tmp_delta[0]);
                if (dval < block->config.dt_min) dval = block->config.dt_min;
                if (dval > block->config.dt_max) dval = block->config.dt_max;
                block->delta[t] = dval;
            } else {
                block->delta[t] = block->config.dt_scale * ((real_t)t / (real_t)seq_len + 1.0f);
            }
        }
        free(tmp_delta);

        /* Selective scan across sequence */
        real_t *scan_out = (real_t *)malloc(seq_len * state_size * sizeof(real_t));
        if (!scan_out) {
            free(u_seq);
            continue;
        }
        
        /* Create A_bar matrix from A_log */
        Matrix *A_bar = matrix_create(state_size, state_size);
        real_t *B_bar = (real_t *)malloc(state_size * sizeof(real_t));
        
        if (!A_bar || !A_bar->data || !B_bar) {
            free(u_seq);
            free(scan_out);
            if (A_bar) matrix_free(A_bar);
            free(B_bar);
            continue;
        }
        
        /* Initialize A_bar from A_log */
        for (size_t i = 0; i < state_size; i++) {
            A_bar->data[i * state_size + i] = expf(block->A_log.data[i]);
        }
        
        /* Initialize B_bar from B */
        for (size_t i = 0; i < state_size; i++) {
            B_bar[i] = block->B_mat.data[i];
        }
        
        /* Run selective scan */
        selective_scan(scan_out, block->hidden, u_seq, block->delta,
                  A_bar, B_bar, &block->C_mat, 0.0f,
                  seq_len, state_size);
        
        /* Output projection: y_t = W_out @ state_t (scan_out contains state vectors) */
        real_t *ybuf = (real_t *)malloc(dim * sizeof(real_t));
        for (size_t t = 0; t < seq_len; t++) {
            const real_t *state_t = &scan_out[t * state_size];
            matrix_vec_mult(ybuf, &block->W_out, state_t); /* ybuf length = dim */
            for (size_t j = 0; j < dim; j++) batch_output[t * dim + j] = ybuf[j];
        }
        free(ybuf);
        
        free(u_seq);
        free(scan_out);
        matrix_free(A_bar);
        free(B_bar);
    }
}
