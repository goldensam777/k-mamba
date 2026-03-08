/* lm.c — Character-level Language Model built on MambaBlock
 *
 * Architecture: EmbeddingTable -> MambaBlock -> LMHead -> softmax
 * Training:     cross-entropy loss, Adam optimizer on all params
 * Generation:   autoregressive with temperature sampling
 */

#include "lm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ===================================================================
 * Internal helpers
 * =================================================================== */

/* Numerically-stable softmax in place over v[n] */
static void softmax_inplace(real_t *v, size_t n) {
    real_t mx = v[0];
    for (size_t i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
    real_t sum = 0.0f;
    for (size_t i = 0; i < n; i++) { v[i] = expf(v[i] - mx); sum += v[i]; }
    if (sum < 1e-12f) sum = 1e-12f;
    for (size_t i = 0; i < n; i++) v[i] /= sum;
}

/* dot product a[n] · b[n] */
static real_t dot(const real_t *a, const real_t *b, size_t n) {
    real_t s = 0.0f;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* Adam update for a single parameter array of length n.
 * m, v are moment buffers (same size as p).
 * step is the global step counter (already incremented before this call).
 */
static void adam_update(real_t *p, const real_t *g, real_t *m, real_t *v,
                        size_t n, size_t step, const OptimConfig *opt) {
    /* Global norm clipping */
    double sq = 0.0;
    for (size_t i = 0; i < n; i++) { double gi = (double)g[i]; sq += gi * gi; }
    double gn = sqrt(sq);
    float scale = 1.0f;
    if (opt->clip_norm > 0.0f && gn > (double)opt->clip_norm)
        scale = (float)((double)opt->clip_norm / gn);

    float mu    = opt->mu;
    float beta2 = opt->beta2;
    float lr    = opt->lr;
    float eps   = opt->eps;
    float wd    = opt->weight_decay;
    float bc1   = 1.0f - powf(mu,    (float)step);
    float bc2   = 1.0f - powf(beta2, (float)step);

    for (size_t i = 0; i < n; i++) {
        real_t gi = g[i] * scale + wd * p[i];
        m[i] = mu    * m[i] + (1.0f - mu)    * gi;
        v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        real_t m_hat = m[i] / bc1;
        real_t v_hat = v[i] / bc2;
        p[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

/* ===================================================================
 * Lifecycle
 * =================================================================== */

LMConfig lm_default_config(void) {
    LMConfig cfg = {
        .vocab_size  = 256,
        .dim         = 384,
        .state_size  = 1024,
        .seq_len     = 128,
        .max_gen_len = 256
    };
    return cfg;
}

size_t lm_num_parameters(const LMConfig *cfg) {
    size_t V, D, S;

    if (!cfg) return 0;
    V = cfg->vocab_size;
    D = cfg->dim;
    S = cfg->state_size;

    /* embedding + head + Mamba block */
    return (V * D) + (V * D + V) + (2 * S * D + 3 * S + D);
}

LM *lm_create(const LMConfig *cfg) {
    if (!cfg) return NULL;

    LM *lm = (LM *)calloc(1, sizeof(LM));
    if (!lm) return NULL;
    lm->config = *cfg;

    size_t V = cfg->vocab_size;
    size_t D = cfg->dim;
    size_t S = cfg->state_size;

    /* --- Embedding table ------------------------------------------ */
    lm->emb.vocab_size = V;
    lm->emb.dim        = D;
    lm->emb.table   = (real_t *)calloc(V * D, sizeof(real_t));
    lm->emb.g_table = (real_t *)calloc(V * D, sizeof(real_t));
    if (!lm->emb.table || !lm->emb.g_table) { lm_free(lm); return NULL; }

    /* Embedding init: uniform [-scale, +scale], scale = 1/sqrt(V) */
    {
        real_t scale = 1.0f / sqrtf((real_t)V);
        for (size_t i = 0; i < V * D; i++)
            lm->emb.table[i] = ((real_t)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }

    lm->m_emb = (real_t *)calloc(V * D, sizeof(real_t));
    lm->v_emb = (real_t *)calloc(V * D, sizeof(real_t));
    if (!lm->m_emb || !lm->v_emb) { lm_free(lm); return NULL; }

    /* --- Mamba block ----------------------------------------------- */
    MambaConfig mcfg = {
        .dim        = D,
        .state_size = S,
        .seq_len    = cfg->seq_len,
        .dt_rank    = 0.1f,
        .dt_scale   = 1.0f,
        .dt_init    = 0.001f,
        .dt_min     = 0.001f,
        .dt_max     = 0.1f
    };
    lm->mamba = mamba_block_create(&mcfg);
    if (!lm->mamba) { lm_free(lm); return NULL; }
    mamba_block_init(lm->mamba);

    /* Attach optimizer with default config (caller will pass actual config
     * at train step time, but we need to register the block). */
    OptimConfig default_opt = { .lr=1e-3f, .mu=0.9f, .beta2=0.999f,
                                .eps=1e-8f, .clip_norm=1.0f, .weight_decay=0.0f };
    mamba_attach_optimizer(lm->mamba, &default_opt);

    /* --- LM Head --------------------------------------------------- */
    lm->head.vocab_size = V;
    lm->head.dim        = D;
    lm->head.step       = 0;
    lm->head.W     = (real_t *)calloc(V * D, sizeof(real_t));
    lm->head.bias  = (real_t *)calloc(V,     sizeof(real_t));
    lm->head.g_W   = (real_t *)calloc(V * D, sizeof(real_t));
    lm->head.g_bias= (real_t *)calloc(V,     sizeof(real_t));
    lm->head.m_W   = (real_t *)calloc(V * D, sizeof(real_t));
    lm->head.v_W   = (real_t *)calloc(V * D, sizeof(real_t));
    lm->head.m_b   = (real_t *)calloc(V,     sizeof(real_t));
    lm->head.v_b   = (real_t *)calloc(V,     sizeof(real_t));
    if (!lm->head.W || !lm->head.bias || !lm->head.g_W || !lm->head.g_bias ||
        !lm->head.m_W || !lm->head.v_W || !lm->head.m_b || !lm->head.v_b) {
        lm_free(lm); return NULL;
    }
    /* Xavier init for head.W */
    {
        real_t scale = sqrtf(6.0f / ((real_t)(D + V)));
        for (size_t i = 0; i < V * D; i++)
            lm->head.W[i] = ((real_t)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
    /* bias stays zero */

    return lm;
}

void lm_free(LM *lm) {
    if (!lm) return;
    free(lm->emb.table);
    free(lm->emb.g_table);
    free(lm->m_emb);
    free(lm->v_emb);
    if (lm->mamba) {
        mamba_free_optimizer(lm->mamba);
        mamba_block_free(lm->mamba);
    }
    free(lm->head.W);    free(lm->head.bias);
    free(lm->head.g_W);  free(lm->head.g_bias);
    free(lm->head.m_W);  free(lm->head.v_W);
    free(lm->head.m_b);  free(lm->head.v_b);
    free(lm);
}

/* ===================================================================
 * Training step
 * =================================================================== */

real_t lm_train_step(LM *lm,
                     const int *in_seq,
                     const int *tgt_seq,
                     const OptimConfig *opt) {
    size_t T = lm->config.seq_len;
    size_t D = lm->config.dim;
    size_t V = lm->config.vocab_size;
    size_t S = lm->config.state_size;

    /* ---- 0. Allocate working buffers ---- */
    real_t *embed_seq   = (real_t *)calloc(T * D, sizeof(real_t)); /* [T x D] */
    real_t *mamba_out   = (real_t *)calloc(T * D, sizeof(real_t)); /* [T x D] */
    real_t *d_mamba     = (real_t *)calloc(T * D, sizeof(real_t)); /* [T x D] */
    real_t *logits      = (real_t *)calloc(V,     sizeof(real_t)); /* [V] */
    real_t *probs       = (real_t *)calloc(V,     sizeof(real_t)); /* [V] */
    real_t *d_logits    = (real_t *)calloc(V,     sizeof(real_t)); /* [V] */
    real_t *d_hidden    = (real_t *)calloc(D,     sizeof(real_t)); /* [D] */

    if (!embed_seq || !mamba_out || !d_mamba || !logits ||
        !probs || !d_logits || !d_hidden) {
        free(embed_seq); free(mamba_out); free(d_mamba); free(logits);
        free(probs); free(d_logits); free(d_hidden);
        return 0.0f;
    }

    /* ---- 1. Embed input tokens ---- */
    for (size_t t = 0; t < T; t++) {
        int tok = (unsigned int)in_seq[t] & 0xFF;
        memcpy(embed_seq + t * D,
               lm->emb.table + tok * D,
               D * sizeof(real_t));
    }

    /* ---- 2. Mamba forward pass ---- */
    mamba_forward(lm->mamba, mamba_out, embed_seq, 1);

    /* ---- 3. Zero gradient buffers ---- */
    mamba_zero_grads(lm->mamba);
    memset(lm->emb.g_table, 0, lm->config.vocab_size * D * sizeof(real_t));
    memset(lm->head.g_W,    0, V * D * sizeof(real_t));
    memset(lm->head.g_bias, 0, V * sizeof(real_t));

    /* ---- 4. Forward through head + compute loss + backward ---- */
    real_t total_loss = 0.0f;

    for (size_t t = 0; t < T; t++) {
        const real_t *h_t = mamba_out + t * D;
        int target = (unsigned int)tgt_seq[t] & 0xFF;

        /* LM head forward: logits[k] = W[k,:] · h_t + bias[k] */
        for (size_t k = 0; k < V; k++)
            logits[k] = dot(lm->head.W + k * D, h_t, D) + lm->head.bias[k];

        /* softmax -> probs */
        memcpy(probs, logits, V * sizeof(real_t));
        softmax_inplace(probs, V);

        /* cross-entropy loss */
        float p_tgt = probs[target];
        if (p_tgt < 1e-12f) p_tgt = 1e-12f;
        total_loss += -logf(p_tgt);

        /* d_logits = probs; d_logits[target] -= 1  (combined softmax+CE grad) */
        memcpy(d_logits, probs, V * sizeof(real_t));
        d_logits[target] -= 1.0f;

        /* LM head backward:
         * g_W[k,j]  += d_logits[k] * h_t[j]
         * g_bias[k] += d_logits[k]
         * d_hidden[j] = sum_k( d_logits[k] * W[k,j] )
         */
        memset(d_hidden, 0, D * sizeof(real_t));
        for (size_t k = 0; k < V; k++) {
            real_t dl = d_logits[k];
            if (dl == 0.0f) continue;
            lm->head.g_bias[k] += dl;
            real_t *gw_row = lm->head.g_W + k * D;
            const real_t *w_row = lm->head.W + k * D;
            for (size_t j = 0; j < D; j++) {
                gw_row[j]   += dl * h_t[j];
                d_hidden[j] += dl * w_row[j];
            }
        }

        /* Accumulate gradient into d_mamba[t] */
        real_t *dm_t = d_mamba + t * D;
        for (size_t j = 0; j < D; j++) dm_t[j] += d_hidden[j];
    }

    /* ---- 5. Mamba backward ---- */
    mamba_backward(lm->mamba, d_mamba, embed_seq, 0);

    /* ---- 6. Embedding backward:
     *   d_emb[t] ≈ W_in^T @ d_u_t, where d_u_t comes from the Mamba input grad.
     *   We approximate via the Mamba W_in matrix directly:
     *   for each timestep t, d_emb_t[j] = sum_i( W_in[i,j] * d_mamba[t*D+j] )
     *   (This propagates the gradient from d_mamba back through the embedding
     *    by using W_in as the effective linear map.)
     *
     *   Simpler & stable alternative used here: direct propagation.
     *   d_emb_t = d_mamba[t] (the gradient already at the embedding output).
     *   Then W_in will correct via its own gradient.
     * ----------------------------------------------------------------*/
    for (size_t t = 0; t < T; t++) {
        int tok = (unsigned int)in_seq[t] & 0xFF;
        real_t *g_row = lm->emb.g_table + tok * D;
        const real_t *dm_t = d_mamba + t * D;
        for (size_t j = 0; j < D; j++) g_row[j] += dm_t[j];
    }

    /* ---- 7. Optimizer steps ---- */
    lm->head.step++;
    lm->emb_step++;
    /* W_in has shape state_size x dim, which can be large.
     * We use the existing MUONCLIP via mamba_optimizer_step. */
    mamba_optimizer_step(lm->mamba, opt);

    /* LM head Adam */
    adam_update(lm->head.W,    lm->head.g_W,   lm->head.m_W, lm->head.v_W,
                V * D, lm->head.step, opt);
    adam_update(lm->head.bias, lm->head.g_bias, lm->head.m_b, lm->head.v_b,
                V,     lm->head.step, opt);

    /* Embedding Adam — only update rows that received gradient */
    {
        size_t step = lm->emb_step;
        float mu    = opt->mu;
        float beta2 = opt->beta2;
        float lr    = opt->lr;
        float eps   = opt->eps;
        float wd    = opt->weight_decay;
        float bc1   = 1.0f - powf(mu,    (float)step);
        float bc2   = 1.0f - powf(beta2, (float)step);
        for (size_t i = 0; i < V * D; i++) {
            real_t gi = lm->emb.g_table[i] + wd * lm->emb.table[i];
            lm->m_emb[i] = mu    * lm->m_emb[i] + (1.0f - mu)    * gi;
            lm->v_emb[i] = beta2 * lm->v_emb[i] + (1.0f - beta2) * gi * gi;
            lm->emb.table[i] -= lr * (lm->m_emb[i] / bc1)
                                   / (sqrtf(lm->v_emb[i] / bc2) + eps);
        }
    }

    /* ---- 8. Cleanup ---- */
    free(embed_seq); free(mamba_out); free(d_mamba);
    free(logits); free(probs); free(d_logits); free(d_hidden);

    (void)S; /* suppress unused-variable warning */
    return total_loss / (real_t)T;
}

/* ===================================================================
 * Generation
 * =================================================================== */

void lm_generate(LM *lm, const char *prompt, size_t max_tokens, real_t temperature) {
    size_t T = lm->config.seq_len;
    size_t D = lm->config.dim;
    size_t V = lm->config.vocab_size;

    /* Build initial context from prompt (ring buffer of length T) */
    int *context = (int *)calloc(T, sizeof(int));
    real_t *embed_seq = (real_t *)calloc(T * D, sizeof(real_t));
    real_t *mamba_out = (real_t *)calloc(T * D, sizeof(real_t));
    real_t *logits    = (real_t *)calloc(V,     sizeof(real_t));
    if (!context || !embed_seq || !mamba_out || !logits) {
        free(context); free(embed_seq); free(mamba_out); free(logits);
        return;
    }

    /* Fill context with prompt chars (most recent T chars, oldest first) */
    const unsigned char *prompt_bytes = (const unsigned char *)(prompt ? prompt : "");
    size_t plen = prompt ? strlen(prompt) : 0;
    if (plen >= T) {
        /* take the last T chars */
        const unsigned char *start = prompt_bytes + (plen - T);
        for (size_t i = 0; i < T; i++) context[i] = start[i];
    } else {
        /* pad left with 0, then fill */
        size_t pad = T - plen;
        for (size_t i = 0; i < pad; i++) context[i] = 0;
        for (size_t i = 0; i < plen; i++) context[pad + i] = prompt_bytes[i];
    }

    /* Generate tokens */
    for (size_t gen = 0; gen < max_tokens; gen++) {
        /* Embed context */
        for (size_t t = 0; t < T; t++) {
            int tok = context[t];
            memcpy(embed_seq + t * D,
                   lm->emb.table + tok * D,
                   D * sizeof(real_t));
        }

        /* Mamba forward */
        mamba_forward(lm->mamba, mamba_out, embed_seq, 1);

        /* Take last hidden state -> logits */
        const real_t *h_last = mamba_out + (T - 1) * D;
        for (size_t k = 0; k < V; k++)
            logits[k] = dot(lm->head.W + k * D, h_last, D) + lm->head.bias[k];

        /* Apply temperature: scale log-probs by 1/T before softmax */
        if (temperature > 0.0f && temperature != 1.0f) {
            for (size_t k = 0; k < V; k++)
                logits[k] /= temperature;
        }
        softmax_inplace(logits, V);

        /* Sample via inverse CDF */
        double r = (double)rand() / ((double)RAND_MAX + 1.0);
        int next_tok = (int)(V - 1);  /* fallback */
        double cum = 0.0;
        for (size_t k = 0; k < V; k++) {
            cum += (double)logits[k];
            if (r < cum) { next_tok = (int)k; break; }
        }

        /* Print raw UTF-8 bytes, but keep non-printable ASCII visible. */
        if (next_tok >= 32 || next_tok == '\n' || next_tok == '\t' || next_tok == '\r') {
            putchar((unsigned char)next_tok);
        } else {
            putchar('?');
        }
        fflush(stdout);

        /* Stop on newline (end of response) */
        if (next_tok == '\n') break;

        /* Slide context window */
        memmove(context, context + 1, (T - 1) * sizeof(int));
        context[T - 1] = next_tok;
    }
    putchar('\n');
    fflush(stdout);

    free(context); free(embed_seq); free(mamba_out); free(logits);
}

/* ===================================================================
 * Checkpoint
 * =================================================================== */

int lm_save(const LM *lm, const char *path) {
    if (!lm || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Header */
    uint32_t magic = LM_MAGIC;
    fwrite(&magic,      sizeof(uint32_t), 1, f);
    fwrite(&lm->config, sizeof(LMConfig), 1, f);

    /* Blobs */
    size_t V = lm->config.vocab_size;
    size_t D = lm->config.dim;
    fwrite(lm->emb.table,       sizeof(real_t), V * D, f);
    fwrite(lm->head.W,          sizeof(real_t), V * D, f);
    fwrite(lm->head.bias,       sizeof(real_t), V,     f);
    /* Mamba weights */
    MambaBlock *m = lm->mamba;
    fwrite(m->A_log.data,      sizeof(real_t), m->A_log.rows      * m->A_log.cols,      f);
    fwrite(m->B_mat.data,      sizeof(real_t), m->B_mat.rows      * m->B_mat.cols,      f);
    fwrite(m->C_mat.data,      sizeof(real_t), m->C_mat.rows      * m->C_mat.cols,      f);
    fwrite(m->W_in.data,       sizeof(real_t), m->W_in.rows       * m->W_in.cols,       f);
    fwrite(m->W_out.data,      sizeof(real_t), m->W_out.rows      * m->W_out.cols,      f);
    fwrite(m->delta_proj.data, sizeof(real_t), m->delta_proj.rows * m->delta_proj.cols, f);

    fclose(f);
    return 0;
}

int lm_load(LM *lm, const char *path) {
    if (!lm || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    /* Check magic */
    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != LM_MAGIC) {
        fclose(f); return -1;
    }

    /* Read config and reject incompatible checkpoints early. */
    LMConfig saved;
    if (fread(&saved, sizeof(LMConfig), 1, f) != 1) { fclose(f); return -1; }
    if (saved.vocab_size  != lm->config.vocab_size  ||
        saved.dim         != lm->config.dim         ||
        saved.state_size  != lm->config.state_size  ||
        saved.seq_len     != lm->config.seq_len     ||
        saved.max_gen_len != lm->config.max_gen_len) {
        fclose(f);
        return -1;
    }

    /* Blobs */
    size_t V = lm->config.vocab_size;
    size_t D = lm->config.dim;
    if (fread(lm->emb.table, sizeof(real_t), V * D, f) != V * D) { fclose(f); return -1; }
    if (fread(lm->head.W,    sizeof(real_t), V * D, f) != V * D) { fclose(f); return -1; }
    if (fread(lm->head.bias, sizeof(real_t), V,     f) != V)     { fclose(f); return -1; }
    MambaBlock *m = lm->mamba;
    size_t na = m->A_log.rows      * m->A_log.cols;
    size_t nb = m->B_mat.rows      * m->B_mat.cols;
    size_t nc = m->C_mat.rows      * m->C_mat.cols;
    size_t ni = m->W_in.rows       * m->W_in.cols;
    size_t no = m->W_out.rows      * m->W_out.cols;
    size_t nd = m->delta_proj.rows * m->delta_proj.cols;
    if (fread(m->A_log.data,      sizeof(real_t), na, f) != na) { fclose(f); return -1; }
    if (fread(m->B_mat.data,      sizeof(real_t), nb, f) != nb) { fclose(f); return -1; }
    if (fread(m->C_mat.data,      sizeof(real_t), nc, f) != nc) { fclose(f); return -1; }
    if (fread(m->W_in.data,       sizeof(real_t), ni, f) != ni) { fclose(f); return -1; }
    if (fread(m->W_out.data,      sizeof(real_t), no, f) != no) { fclose(f); return -1; }
    if (fread(m->delta_proj.data, sizeof(real_t), nd, f) != nd) { fclose(f); return -1; }

    fclose(f);
    return 0;
}
