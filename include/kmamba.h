#ifndef KMAMBA_H
#define KMAMBA_H

#include <stddef.h>
#include <stdint.h>
#include "optimatrix.h"
#include "scan.h"

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
    size_t state_size;  /* mamba state size */
    size_t seq_len;     /* context length */
    
    float dt_scale;
    float dt_min;
    float dt_max;

    /* ConvND parameters */
    int    use_convnd;     /* 0 = disable, 1 = enable ConvND avant scan */
    long   convnd_K;       /* Taille du noyau ConvND (K>=1) */
    long   convnd_ndims;   /* Nombre de dimensions spatiales (1, 2, ou 3) */
} MBConfig;

/* ============================================================================
 * MambaBlock - Single SSM layer
 * ============================================================================ */
typedef struct {
    MBConfig config;
    
    /* Parameters */
    MBMatrix W_in;        /* [state_size x dim] */
    MBMatrix W_out;       /* [dim x state_size] */
    MBMatrix A_log;       /* [state_size] diagonal */
    MBMatrix W_B;         /* [state_size x dim] — data-dependent B projection */
    MBMatrix W_C;         /* [state_size x dim] — data-dependent C projection */
    MBMatrix delta_proj;  /* [1 x dim] */

    /* BCNorm biases (Mamba-3) */
    float *b_B;           /* [state_size] — bias after RMSNorm(W_B·x) */
    float *b_C;           /* [state_size] — bias after RMSNorm(W_C·x) */

    /* Complex SSM / RoPE angles (Mamba-3 §3.2) */
    float *theta;         /* [state_size/2] — learned rotation angles per pair */
    
    /* ConvND parameters */
    float *convnd_kernel;  /* [convnd_ndims * convnd_K * dim] */
    float *convnd_bias;    /* [dim] */
    ConvNDWorkspace *convnd_ws;
    
    /* Runtime buffers */
    float *hidden;         /* [dim] */
    float *delta;          /* [seq_len] */
    float *scan_B;         /* [seq_len x state_size] */
    float *scan_C;         /* [seq_len x state_size] */
    float *scan_delta;     /* [seq_len x state_size] */
    float *scan_h;         /* [state_size] */
} MambaBlock;

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
} MBOptimState;

/* ============================================================================
 * KMamba Configuration
 * ============================================================================ */
typedef struct {
    size_t vocab_size;   /* default: 256 (byte-level) */
    size_t dim;          /* model dimension */
    size_t state_size;   /* mamba state size */
    size_t seq_len;      /* context length */
    size_t n_layers;     /* number of stacked MambaBlocks */

    float dt_scale;
    float dt_min;
    float dt_max;
    
    /* ConvND parameters (optionnel) */
    int    use_convnd;     /* 0 = disable, 1 = enable ConvND avant scan */
    long   convnd_K;       /* Taille du noyau ConvND (K>=1) */
    long   convnd_ndims;   /* Nombre de dimensions spatiales (1, 2, ou 3) */
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
} KMamba;

/* ============================================================================
 * MambaBlock API
 * ============================================================================ */
MambaBlock* mamba_block_create(const MBConfig *config);
void        mamba_block_free(MambaBlock *block);
void        mamba_block_init(MambaBlock *block);

void mamba_block_forward(MambaBlock *block, float *output, const float *input,
                        size_t batch_size);

/* Training functions */
void mamba_attach_optimizer(MambaBlock *block, OptimizerType type, const MBOptimConfig *optconf);
void mamba_free_optimizer(MambaBlock *block);
void mamba_zero_grads(MambaBlock *block);
void mamba_optimizer_step(MambaBlock *block, const MBOptimConfig *conf);

void mamba_backward(MambaBlock *block, const float *dY, const float *input,
                    float *d_input, size_t batch_index);

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
