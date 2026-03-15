#ifndef KMAMBA_H
#define KMAMBA_H

#include <stddef.h>
#include <stdint.h>
#include "optimatrix.h"

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
    float dt_rank;
    float dt_init;
    
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
    MBMatrix B_mat;       /* [state_size] shared */
    MBMatrix C_mat;       /* [state_size] shared */
    MBMatrix delta_proj;   /* [1 x dim] */
    
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
 * Optimizer Configuration
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
 * Optimizer State (MUONCLIP)
 * ============================================================================ */
typedef struct {
    size_t step;
    
    /* Gradients */
    float *g_W_in;
    float *g_W_out;
    float *g_A_log;
    float *g_B_mat;
    float *g_C_mat;
    float *g_delta_proj;
    
    /* Moments */
    float *m_W_in;
    float *v_W_in;
    float *m_W_out;
    float *v_W_out;
    float *m_A_log;
    float *v_A_log;
    float *m_B_mat;
    float *v_B_mat;
    float *m_C_mat;
    float *v_C_mat;
    float *m_delta_proj;
    float *v_delta_proj;
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
} KMamba;

/* ============================================================================
 * MambaBlock API
 * ============================================================================ */
MambaBlock* mamba_block_create(const MBConfig *config);
void        mamba_block_free(MambaBlock *block);
void        mamba_block_init(MambaBlock *block);

void mamba_block_forward(MambaBlock *block, float *output, const float *input,
                        size_t batch_size);
void mamba_block_forward_2d(MambaBlock *block, float *output, const float *input,
                            size_t d1, size_t d2);

/* Training functions */
void mamba_attach_optimizer(MambaBlock *block, const MBOptimConfig *optconf);
void mamba_free_optimizer(MambaBlock *block);
void mamba_zero_grads(MambaBlock *block);
void mamba_optimizer_step(MambaBlock *block, const MBOptimConfig *conf);

void mamba_backward(MambaBlock *block, const float *dY, const float *input,
                    float *d_input, size_t batch_index);
void mamba_backward_2d(MambaBlock *block, const float *dY, const float *input,
                       float *d_input, size_t d1, size_t d2);

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
