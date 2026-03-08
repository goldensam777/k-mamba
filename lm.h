#ifndef LM_H
#define LM_H

#include "mamba.h"
#include <stddef.h>
#include <stdio.h>

/* ===================================================================
 * LM — Character-level Language Model on top of Mamba SSM
 *
 * Architecture:
 *   input tokens (int[seq_len])
 *       -> EmbeddingTable  [vocab_size x dim]
 *       -> MambaBlock      [dim -> dim]
 *       -> LMHead          [dim -> vocab_size]
 *       -> softmax -> cross-entropy loss
 * =================================================================== */

/* -----------------------------------------------------------------
 * Configuration
 * ----------------------------------------------------------------- */
typedef struct {
    size_t vocab_size;   /* token vocabulary size (256 = byte-level)  */
    size_t dim;          /* embedding + model dimension               */
    size_t state_size;   /* Mamba SSM state dimension                 */
    size_t seq_len;      /* context window (tokens)                   */
    size_t max_gen_len;  /* max tokens to generate in one call        */
} LMConfig;

/* Shared default config for the conversational LM. */
LMConfig lm_default_config(void);

/* Total trainable parameters for the current config. */
size_t   lm_num_parameters(const LMConfig *cfg);

/* -----------------------------------------------------------------
 * Embedding Table
 * table[vocab_size x dim] — one row per token, learnable.
 * ----------------------------------------------------------------- */
typedef struct {
    real_t *table;       /* [vocab_size x dim], row-major             */
    real_t *g_table;     /* gradient buffer, same layout              */
    size_t  vocab_size;
    size_t  dim;
} EmbeddingTable;

/* -----------------------------------------------------------------
 * LM Head  (linear projection + softmax)
 * W[vocab_size x dim], bias[vocab_size]
 * ----------------------------------------------------------------- */
typedef struct {
    real_t *W;           /* [vocab_size x dim]                        */
    real_t *bias;        /* [vocab_size]                              */
    real_t *g_W;         /* gradient of W                            */
    real_t *g_bias;      /* gradient of bias                         */
    real_t *m_W;         /* Adam first moment  — W                   */
    real_t *v_W;         /* Adam second moment — W                   */
    real_t *m_b;         /* Adam first moment  — bias                */
    real_t *v_b;         /* Adam second moment — bias                */
    size_t  vocab_size;
    size_t  dim;
    size_t  step;        /* optimizer step counter (bias correction)  */
} LMHead;

/* -----------------------------------------------------------------
 * Full Language Model
 * ----------------------------------------------------------------- */
typedef struct {
    LMConfig       config;
    EmbeddingTable emb;
    MambaBlock    *mamba;
    LMHead         head;
    /* Adam moments for embedding */
    real_t        *m_emb;
    real_t        *v_emb;
    size_t         emb_step;
} LM;

/* -----------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------- */

/* Allocate, zero-init, and Xavier-init all weights. Returns NULL on OOM. */
LM   *lm_create(const LMConfig *cfg);

/* Free all heap memory owned by the LM. */
void  lm_free(LM *lm);

/* -----------------------------------------------------------------
 * Training
 * ----------------------------------------------------------------- */

/* One supervised step over a sequence of length cfg->seq_len.
 * in_seq[t]  = input  token at timestep t
 * tgt_seq[t] = target token at timestep t  (= in_seq[t+1] typically)
 * Returns average cross-entropy loss over the sequence.
 */
real_t lm_train_step(LM *lm,
                     const int *in_seq,
                     const int *tgt_seq,
                     const OptimConfig *opt);

/* -----------------------------------------------------------------
 * Generation
 * ----------------------------------------------------------------- */

/* Autoregressive generation.
 * Feeds prompt chars into the context, then generates max_tokens
 * additional characters and writes them to stdout (flushed per char).
 * temperature > 1 => more random; < 1 => more focused; 0.8 is a good default.
 */
void  lm_generate(LM *lm,
                  const char *prompt,
                  size_t      max_tokens,
                  real_t      temperature);

/* -----------------------------------------------------------------
 * Checkpoint
 * ----------------------------------------------------------------- */

/* Write binary snapshot. Returns 0 on success, -1 on error. */
int   lm_save(const LM *lm, const char *path);

/* Load binary snapshot. LM must have been created with matching config.
 * Returns 0 on success, -1 on error. */
int   lm_load(LM *lm, const char *path);

/* Checkpoint file magic word */
#define LM_MAGIC 0x4C4D3030u   /* "LM00" */

#endif /* LM_H */
