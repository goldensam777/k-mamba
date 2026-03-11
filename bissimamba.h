#ifndef BISSIMAMBA_H
#define BISSIMAMBA_H

#include <stddef.h>
#include <stdint.h>
#include "optimatrix.h"

typedef struct {
    size_t vocab_size;   /* default: 256 (byte-level) */
    size_t dim;          /* model dimension */
    size_t state_size;   /* mamba state size */
    size_t seq_len;      /* context length */
    size_t n_layers;     /* number of stacked MambaBlocks */

    float dt_scale;
    float dt_min;
    float dt_max;
} BissiMambaConfig;

typedef struct {
    BissiMambaConfig cfg;

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
} BissiMamba;

BissiMamba* bissimamba_create(const BissiMambaConfig *cfg);
void        bissimamba_free(BissiMamba *m);

int  bissimamba_init(BissiMamba *m, uint32_t seed);
int  bissimamba_enable_training(BissiMamba *m, const MBOptimConfig *opt_blocks,
                                float lr_embed_head, float weight_decay);

int         bissimamba_save(const BissiMamba *m, const char *path);
BissiMamba* bissimamba_load(const char *path, int for_training,
                            const MBOptimConfig *opt_blocks,
                            float lr_embed_head, float weight_decay);

/* Inference: tokens length must equal cfg.seq_len. logits_out: [seq_len, vocab_size]. */
int bissimamba_forward(BissiMamba *m, const uint8_t *tokens, float *logits_out);

/* One training step on one sequence. */
float bissimamba_train_step(BissiMamba *m, const uint8_t *tokens_plus1);

/* Batch training: B sequences of (seq_len+1) bytes each. Returns mean loss. */
float bissimamba_train_batch(BissiMamba *m, const uint8_t *batch_tokens, size_t batch_size);

#endif /* BISSIMAMBA_H */
