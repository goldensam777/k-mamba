#ifndef KMAMBA_H
#define KMAMBA_H

#include <stddef.h>
#include <stdint.h>
#include "scan.h"
#include "scan_nd.h"
#include "wavefront_plan.h"
#include "wavefront_nd.h"

#define KMAMBA_MAX_NDIMS 8

/* ============================================================================
 * Optimizer Config (previously from optimatrix.h)
 * ============================================================================ */
typedef struct {
    float lr;
    float mu;
    float beta2;
    float eps;
    float clip_norm;
    float weight_decay;
} MBOptimConfig;

/* ============================================================================
 * ConvND Types (previously from optimatrix.h)
 * ============================================================================ */
typedef struct {
    long ndims;
    long total_floats;
    float **inter;       /* [ndims+1][total_floats] intermediates */
} ConvNDWorkspace;

typedef struct {
    float *input;        /* Input tensor (read/write for backward) */
    const float *kernel; /* Kernel [ndims * K * D] */
    const float *bias;   /* Bias [D] or NULL */
    float *output;       /* Output tensor */
    float *dy;           /* Gradient w.r.t. output (for backward) */
    float *dinput;       /* Gradient w.r.t. input (output) */
    float *dkernel;      /* Gradient w.r.t. kernel [ndims * K * D] */
    float *dbias;        /* Gradient w.r.t. bias [D] or NULL */
    const long *dims;    /* Shape [ndims] */
    long ndims;          /* Number of dimensions */
    long D;              /* Depth/channels */
    long K;              /* Kernel size */
} ConvNDParams;

typedef enum {
    CONVND_FORWARD = 1,   /* Forward pass only */
    CONVND_BACKWARD = 2,  /* Backward pass only */
    CONVND_COMPLETE = 3   /* Forward + Backward */
} ConvNDMode;

/* ConvND API */
ConvNDWorkspace* convnd_workspace_create(const ConvNDParams *p);
void convnd_workspace_free(ConvNDWorkspace *ws);
void convnd(ConvNDParams *p, ConvNDMode mode, ConvNDWorkspace *ws);

/* Full ConvND depthwise reference API.
 * Kernel layout is [K^ndims, D] with the last axis varying fastest.
 * The support is causal on every axis, matching the existing separable ConvND. */
typedef struct {
    const float *input;   /* Input tensor [prod(dims), D] */
    const float *kernel;  /* Full kernel [K^ndims, D] */
    const float *bias;    /* Bias [D] or NULL */
    float *output;        /* Output tensor [prod(dims), D] */
    const float *dy;      /* Gradient w.r.t. output [prod(dims), D] */
    float *dinput;        /* Gradient w.r.t. input [prod(dims), D] */
    float *dkernel;       /* Gradient w.r.t. kernel [K^ndims, D] */
    float *dbias;         /* Gradient w.r.t. bias [D] or NULL */
    const long *dims;     /* Spatial shape [ndims] */
    long ndims;           /* Number of spatial dimensions */
    long D;               /* Depth/channels */
    long K;               /* conv kernel_size ; distinct du state_size du scan */
} ConvNDFullParams;

long convnd_full_kernel_volume(long ndims, long K);
void convnd_full_ref(ConvNDFullParams *p, ConvNDMode mode);
void convnd_full_ref_with_plan(ConvNDFullParams *p,
                               const KMWavefrontPlan *plan,
                               ConvNDMode mode);

/* ============================================================================
 * Basic Matrix type
 * ============================================================================ */
typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} MBMatrix;

/* ============================================================================
 * MambaBlock Configuration
 * ============================================================================ */
typedef struct {
    size_t dim;         /* model dimension */
    size_t state_size;  /* mamba state size (N) */
    size_t seq_len;     /* context length */
    size_t mimo_rank;   /* MIMO rank R (default 1 = SISO, B∈R^{NxR}, C∈R^{NxR}, u∈R^R) */

    float dt_scale;
    float dt_min;
    float dt_max;

    /* Shared ND topology for scanND / convND.
     * spatial_ndims == 0 means the implicit 1D shape [seq_len]. */
    long   spatial_ndims;
    long   spatial_dims[KMAMBA_MAX_NDIMS];

    /* ConvND parameters */
    int    use_convnd;     /* 0 = disable, 1 = enable ConvND locale */
    long   convnd_K;       /* Conv kernel_size (K>=1), distinct du state_size */
    long   convnd_ndims;   /* 0 => dérivé de spatial_ndims ; sinon doit matcher */
} MBConfig;

/* ============================================================================
 * MambaBlock - Single SSM layer
 * ============================================================================ */
typedef struct {
    MBConfig config;
    
    /* Parameters */
    MBMatrix W_in;        /* [R x dim]     — projects x_t → u_t ∈ R^R  (MIMO input) */
    MBMatrix W_out;       /* [dim x R]     — projects y_t ∈ R^R → output ∈ R^dim */
    MBMatrix A_log;       /* [state_size]  — diagonal log(-A), always negative */
    MBMatrix W_B;         /* [N*R x dim]   — data-dependent B∈R^{NxR} projection */
    MBMatrix W_C;         /* [N*R x dim]   — data-dependent C∈R^{NxR} projection */
    MBMatrix delta_proj;  /* [1 x dim] */

    /* BCNorm biases (Mamba-3) */
    float *b_B;           /* [state_size] — bias after RMSNorm(W_B·x) */
    float *b_C;           /* [state_size] — bias after RMSNorm(W_C·x) */

    /* Complex SSM / RoPE angles (Mamba-3 §3.2) */
    float *theta;         /* [state_size/2] — learned rotation angles per pair */

    /* Exp-Trapezoidal discretization (Mamba-3 §3.1) */
    MBMatrix lambda_proj; /* [1 x dim] — projects x_t -> scalar lambda_t (sigmoid -> [0,1]) */

    /* Shared ND execution topology reused by scanND / convND. */
    KMWavefrontPlan *wavefront_plan;
    
    /* ConvND parameters */
    float *convnd_kernel;  /* [convnd_ndims * convnd_K * dim] */
    float *convnd_bias;    /* [dim] */
    ConvNDWorkspace *convnd_ws;
    
    /* Runtime buffers */
    float *hidden;         /* [state_size] — SSM state at last timestep */
    float *delta;          /* [seq_len] */
    float *scan_B;         /* [seq_len x N*R] — normalized B_t columns, R=mimo_rank */
    float *scan_C;         /* [seq_len x N*R] — normalized C_t columns, R=mimo_rank */
    float *scan_delta;     /* [seq_len x state_size] */
    float *scan_h;         /* [state_size] */
} MambaBlock;

typedef struct {
    float *hidden;         /* [state_size] */
    float *delta;          /* [seq_len] */
    float *scan_B;         /* [seq_len x N*R] */
    float *scan_C;         /* [seq_len x N*R] */
    float *scan_delta;     /* [seq_len x state_size] */
    float *scan_h;         /* [state_size] */
} MambaBlockWorkspace;

/* ============================================================================
 * Optimizer Types
 * ============================================================================ */
typedef enum {
    OPTIMIZER_ADAM_CLIP,    /* Current implementation (AdamW + gradient clipping) */
    OPTIMIZER_MUON,          /* MUON with Newton-Schulz (future implementation) */
    OPTIMIZER_SGD,           /* Vanilla SGD with momentum */
    OPTIMIZER_ADAMW          /* Standard AdamW */
} OptimizerType;

/* MBOptimConfig est défini dans optimatrix.h */

/* ============================================================================
 * Optimizer State (modular)
 * ============================================================================ */
typedef struct {
    OptimizerType type;
    size_t step;
    
    /* Gradients */
    float *g_W_in;
    float *g_W_out;
    float *g_A_log;
    float *g_W_B;
    float *g_W_C;
    float *g_b_B;
    float *g_b_C;
    float *g_delta_proj;
    float *g_theta;       /* [state_size/2] */
    float *g_lambda_proj; /* [dim] */

    /* Moments (used by ADAM-based optimizers) */
    float *m_W_in;
    float *v_W_in;
    float *m_W_out;
    float *v_W_out;
    float *m_A_log;
    float *v_A_log;
    float *m_W_B;
    float *v_W_B;
    float *m_W_C;
    float *v_W_C;
    float *m_b_B;
    float *v_b_B;
    float *m_b_C;
    float *v_b_C;
    float *m_delta_proj;
    float *v_delta_proj;
    float *m_theta;       /* [state_size/2] */
    float *v_theta;       /* [state_size/2] — only for Adam-based */
    float *m_lambda_proj; /* [dim] */
    float *v_lambda_proj; /* [dim] — only for Adam-based */
} MBOptimState;

/* ============================================================================
 * KMamba Configuration
 * ============================================================================ */
typedef struct {
    size_t vocab_size;   /* default: 256 (byte-level) */
    size_t dim;          /* model dimension */
    size_t state_size;   /* mamba state size (N) */
    size_t seq_len;      /* context length */
    size_t n_layers;     /* number of stacked MambaBlocks */
    size_t mimo_rank;    /* MIMO rank R (0 or 1 = SISO; otherwise full MIMO) */

    float dt_scale;
    float dt_min;
    float dt_max;

    /* Shared ND topology for scanND / convND.
     * spatial_ndims == 0 means the implicit 1D shape [seq_len]. */
    long   spatial_ndims;
    long   spatial_dims[KMAMBA_MAX_NDIMS];

    /* ConvND parameters (optionnel) */
    int    use_convnd;     /* 0 = disable, 1 = enable ConvND locale */
    long   convnd_K;       /* Conv kernel_size (K>=1), distinct du state_size */
    long   convnd_ndims;   /* 0 => dérivé de spatial_ndims ; sinon doit matcher */
} KMambaConfig;

typedef struct {
    KMambaConfig cfg;

    /* Parameters */
    float *embedding; /* [vocab_size, dim] row-major */
    float *head;      /* [dim, vocab_size] row-major */

    /* Stack */
    MambaBlock **layers; /* [n_layers] */

    /* Training */
    int for_training;
    MBOptimConfig opt_blocks;
    float lr_embed_head;
    float weight_decay;

    /* Adam state pour embedding et head */
    float  *m_embedding;
    float  *v_embedding;
    float  *m_head;
    float  *v_head;
    size_t  step_embed_head;
    float   last_grad_norm;
    float   last_grad_over_clip;
    int     last_grad_would_clip;
} KMamba;

/* ============================================================================
 * MambaBlock API
 * ============================================================================ */
MambaBlock* mamba_block_create(const MBConfig *config);
void        mamba_block_free(MambaBlock *block);
void        mamba_block_init(MambaBlock *block);
MambaBlockWorkspace* mamba_block_workspace_create(const MambaBlock *block);
void                 mamba_block_workspace_free(MambaBlockWorkspace *ws);

void mamba_block_forward(MambaBlock *block, float *output, const float *input,
                        size_t batch_size);
void mamba_block_forward_ws(MambaBlock *block, MambaBlockWorkspace *ws,
                            float *output, const float *input, size_t batch_size);

/* Training functions */
void mamba_attach_optimizer(MambaBlock *block, OptimizerType type, const MBOptimConfig *optconf);
void mamba_free_optimizer(MambaBlock *block);
void mamba_zero_grads(MambaBlock *block);
void mamba_optimizer_step(MambaBlock *block, const MBOptimConfig *conf);
float mamba_block_grad_sqnorm(const MambaBlock *block);

void mamba_backward(MambaBlock *block, const float *dY, const float *input,
                    float *d_input, size_t batch_index);
void mamba_backward_ws(MambaBlock *block, MambaBlockWorkspace *ws,
                       const float *dY, const float *input,
                       float *d_input, size_t batch_index);
void mamba_backward_ws_local(MambaBlock *block, MambaBlockWorkspace *ws,
                              const float *dY, const float *input,
                              float *d_input, size_t batch_index,
                              MBOptimState *local_grad);

/* Per-thread local gradient helpers for lock-free parallel backward */
MBOptimState* mamba_local_grad_alloc(const MambaBlock *block);
void          mamba_local_grad_reduce(MambaBlock *block, const MBOptimState *local);
void          mamba_local_grad_free(MBOptimState *local);

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */
MBMatrix* mb_matrix_create(size_t rows, size_t cols);
void      mb_matrix_free(MBMatrix *m);
void      mb_matrix_zero(MBMatrix *m);
void      mb_matrix_copy(MBMatrix *dst, const MBMatrix *src);
void      mb_matrix_print(const MBMatrix *m);

void mb_matrix_vec_mult(float *out, const MBMatrix *m, const float *v);
void mb_vec_add(float *y, const float *x, size_t n);
void mb_vec_scale(float *v, float alpha, size_t n);

void mb_compute_delta(float *delta_out, const MambaBlock *block,
                      const float *delta_in, size_t seq_len);

/* ============================================================================
 * KMamba API
 * ============================================================================ */
KMamba* kmamba_create(const KMambaConfig *cfg);
void        kmamba_free(KMamba *m);

int  kmamba_init(KMamba *m, uint32_t seed);
int  kmamba_enable_training(KMamba *m, const MBOptimConfig *opt_blocks,
                                float lr_embed_head, float weight_decay);
int  kmamba_enable_training_with_optimizer(KMamba *m, OptimizerType opt_type,
                                          const MBOptimConfig *opt_blocks,
                                          float lr_embed_head, float weight_decay);

int         kmamba_save(const KMamba *m, const char *path);
KMamba* kmamba_load(const char *path, int for_training,
                            const MBOptimConfig *opt_blocks,
                            float lr_embed_head, float weight_decay);

/* Inference: tokens length must equal cfg.seq_len. logits_out: [seq_len, vocab_size]. */
int kmamba_forward(KMamba *m, const uint8_t *tokens, float *logits_out);

/* One training step on one sequence. */
float kmamba_train_step(KMamba *m, const uint8_t *tokens_plus1);

/* Batch training: B sequences of (seq_len+1) bytes each. Returns mean loss. */
float kmamba_train_batch(KMamba *m, const uint8_t *batch_tokens, size_t batch_size);

#endif /* KMAMBA_H */
