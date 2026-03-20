#include "../include/kmamba.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ========= utils ========= */

/* Adam update pour embedding et head (mêmes hyperparams que opt_blocks) */
#define ADAM_EH_UPDATE(km, param, grad_arr, m_arr, v_arr, N) do {          \
    float _mu    = (km)->opt_blocks.mu;                                      \
    float _b2    = (km)->opt_blocks.beta2;                                   \
    float _eps   = (km)->opt_blocks.eps;                                     \
    float _lr    = (km)->lr_embed_head;                                      \
    float _wd    = (km)->weight_decay;                                       \
    float _bc1   = 1.0f - powf(_mu, (float)(km)->step_embed_head);          \
    float _bc2   = 1.0f - powf(_b2, (float)(km)->step_embed_head);          \
    for (size_t _i = 0; _i < (N); _i++) {                                   \
        float _g = (grad_arr)[_i] + _wd * (param)[_i];                      \
        (m_arr)[_i] = _mu * (m_arr)[_i] + (1.0f - _mu) * _g;               \
        (v_arr)[_i] = _b2 * (v_arr)[_i] + (1.0f - _b2) * _g * _g;         \
        float _mh = (m_arr)[_i] / _bc1;                                      \
        float _vh = (v_arr)[_i] / _bc2;                                      \
        (param)[_i] -= _lr * _mh / (sqrtf(_vh) + _eps);                     \
    }                                                                         \
} while (0)

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) { fprintf(stderr, "OOM\n"); abort(); }
    return p;
}

static float frand_uniform(void) {
    return (float)rand() / (float)RAND_MAX;
}

static void xavier_uniform(float *w, size_t fan_in, size_t fan_out, size_t n) {
    float scale = sqrtf(6.0f / ((float)fan_in + (float)fan_out));
    for (size_t i = 0; i < n; i++)
        w[i] = (frand_uniform() * 2.0f - 1.0f) * scale;
}

static void transpose(const float *src, float *dst, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

static void softmax(float *probs, const float *logits, size_t n) {
    float maxv = logits[0];
    for (size_t i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = expf(logits[i] - maxv);
        probs[i] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++) probs[i] *= inv;
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

/* ========= embedding ========= */

static void embed_lookup(const KMamba *m, float *out, const uint8_t *tokens) {
    size_t D = m->cfg.dim;
    for (size_t t = 0; t < m->cfg.seq_len; t++)
        memcpy(&out[t * D], &m->embedding[(size_t)tokens[t] * D], D * sizeof(float));
}

/* ========= create / free ========= */

KMamba* kmamba_create(const KMambaConfig *cfg) {
    if (!cfg || !cfg->vocab_size || !cfg->dim || !cfg->seq_len || !cfg->n_layers)
        return NULL;

    KMamba *m = (KMamba *)xcalloc(1, sizeof(KMamba));
    m->cfg = *cfg;

    m->embedding = (float *)xcalloc(cfg->vocab_size * cfg->dim, sizeof(float));
    m->head      = (float *)xcalloc(cfg->dim * cfg->vocab_size, sizeof(float));

    m->layers = (MambaBlock **)xcalloc(cfg->n_layers, sizeof(MambaBlock *));
    for (size_t i = 0; i < cfg->n_layers; i++) {
        MBConfig bc = {
            .dim        = cfg->dim,
            .state_size = cfg->state_size,
            .seq_len    = cfg->seq_len,
            .mimo_rank  = cfg->mimo_rank > 0 ? cfg->mimo_rank : 1,
            .dt_scale   = cfg->dt_scale,
            .dt_min     = cfg->dt_min,
            .dt_max     = cfg->dt_max
        };
        m->layers[i] = mamba_block_create(&bc);
        if (!m->layers[i]) { kmamba_free(m); return NULL; }
    }

    return m;
}

void kmamba_free(KMamba *m) {
    if (!m) return;
    if (m->layers) {
        for (size_t i = 0; i < m->cfg.n_layers; i++) {
            if (m->layers[i]) {
                if (m->for_training) mamba_free_optimizer(m->layers[i]);
                mamba_block_free(m->layers[i]);
            }
        }
        free(m->layers);
    }
    free(m->embedding);
    free(m->head);
    free(m->m_embedding);
    free(m->v_embedding);
    free(m->m_head);
    free(m->v_head);
    free(m);
}

int kmamba_init(KMamba *m, uint32_t seed) {
    if (!m) return -1;
    srand((unsigned)seed);
    xavier_uniform(m->embedding, m->cfg.vocab_size, m->cfg.dim, m->cfg.vocab_size * m->cfg.dim);
    xavier_uniform(m->head, m->cfg.dim, m->cfg.vocab_size, m->cfg.dim * m->cfg.vocab_size);
    for (size_t i = 0; i < m->cfg.n_layers; i++)
        mamba_block_init(m->layers[i]);
    return 0;
}

int kmamba_enable_training(KMamba *m, const MBOptimConfig *opt_blocks,
                               float lr_embed_head, float weight_decay) {
    return kmamba_enable_training_with_optimizer(m, OPTIMIZER_ADAM_CLIP, opt_blocks, 
                                                 lr_embed_head, weight_decay);
}

int kmamba_enable_training_with_optimizer(KMamba *m, OptimizerType opt_type,
                                          const MBOptimConfig *opt_blocks,
                                          float lr_embed_head, float weight_decay) {
    if (!m || !opt_blocks) return -1;
    m->for_training = 1;
    m->opt_blocks = *opt_blocks;
    m->lr_embed_head = lr_embed_head;
    m->weight_decay = weight_decay;
    for (size_t i = 0; i < m->cfg.n_layers; i++)
        mamba_attach_optimizer(m->layers[i], opt_type, opt_blocks);

    size_t VD = m->cfg.vocab_size * m->cfg.dim;
    m->m_embedding     = (float *)xcalloc(VD, sizeof(float));
    m->v_embedding     = (float *)xcalloc(VD, sizeof(float));
    m->m_head          = (float *)xcalloc(VD, sizeof(float));
    m->v_head          = (float *)xcalloc(VD, sizeof(float));
    m->step_embed_head = 0;
    return 0;
}

/* ========= checkpoint IO ========= */

typedef struct {
    char magic[8];       /* "KMAMBA\0\0" */
    uint32_t version;
    uint32_t reserved;
    uint64_t vocab_size;
    uint64_t dim;
    uint64_t state_size;
    uint64_t seq_len;
    uint64_t n_layers;
    uint64_t mimo_rank;  /* v4: MIMO rank R (0/1 = SISO) */
    float dt_scale;
    float dt_min;
    float dt_max;
} CheckpointHeader;

static int write_floats(FILE *f, const float *p, size_t n) {
    return fwrite(p, sizeof(float), n, f) == n ? 0 : -1;
}

static int read_floats(FILE *f, float *p, size_t n) {
    return fread(p, sizeof(float), n, f) == n ? 0 : -1;
}

int kmamba_save(const KMamba *m, const char *path) {
    if (!m || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    CheckpointHeader h = {0};
    memcpy(h.magic, "KMAMBA", 6);
    h.version    = 4; /* v4: adds mimo_rank (MIMO B/C rank) */
    h.vocab_size = (uint64_t)m->cfg.vocab_size;
    h.dim        = (uint64_t)m->cfg.dim;
    h.state_size = (uint64_t)m->cfg.state_size;
    h.seq_len    = (uint64_t)m->cfg.seq_len;
    h.n_layers   = (uint64_t)m->cfg.n_layers;
    h.mimo_rank  = (uint64_t)(m->cfg.mimo_rank > 0 ? m->cfg.mimo_rank : 1);
    h.dt_scale   = m->cfg.dt_scale;
    h.dt_min     = m->cfg.dt_min;
    h.dt_max     = m->cfg.dt_max;

    if (fwrite(&h, sizeof(h), 1, f) != 1) { fclose(f); return -1; }
    if (write_floats(f, m->embedding, m->cfg.vocab_size * m->cfg.dim)) { fclose(f); return -1; }
    if (write_floats(f, m->head, m->cfg.dim * m->cfg.vocab_size)) { fclose(f); return -1; }

    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        const MambaBlock *b = m->layers[i];
        size_t theta_size = b->config.state_size / 2; if (theta_size == 0) theta_size = 1;
        if (write_floats(f, b->W_in.data,       b->W_in.rows * b->W_in.cols)            ||
            write_floats(f, b->W_out.data,      b->W_out.rows * b->W_out.cols)           ||
            write_floats(f, b->A_log.data,      b->A_log.rows * b->A_log.cols)           ||
            write_floats(f, b->W_B.data,        b->W_B.rows * b->W_B.cols)               ||
            write_floats(f, b->W_C.data,        b->W_C.rows * b->W_C.cols)               ||
            write_floats(f, b->b_B,             b->W_B.rows)                              ||
            write_floats(f, b->b_C,             b->W_C.rows)                              ||
            write_floats(f, b->delta_proj.data,  b->delta_proj.rows * b->delta_proj.cols) ||
            write_floats(f, b->lambda_proj.data, b->lambda_proj.rows * b->lambda_proj.cols)||
            write_floats(f, b->theta,           theta_size)) {
            fclose(f); return -1;
        }
    }

    fclose(f);
    return 0;
}

KMamba* kmamba_load(const char *path, int for_training,
                            const MBOptimConfig *opt_blocks,
                            float lr_embed_head, float weight_decay) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    CheckpointHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); return NULL; }
    if (memcmp(h.magic, "KMAMBA", 6) != 0 || h.version > 4) { fclose(f); return NULL; }

    KMambaConfig cfg = {
        .vocab_size  = (size_t)h.vocab_size,
        .dim         = (size_t)h.dim,
        .state_size  = (size_t)h.state_size,
        .seq_len     = (size_t)h.seq_len,
        .n_layers    = (size_t)h.n_layers,
        .mimo_rank   = (h.version >= 4 && h.mimo_rank > 0) ? (size_t)h.mimo_rank : 1,
        .dt_scale    = h.dt_scale,
        .dt_min      = h.dt_min,
        .dt_max      = h.dt_max
    };

    KMamba *m = kmamba_create(&cfg);
    if (!m) { fclose(f); return NULL; }
    if (for_training) {
        if (!opt_blocks) { kmamba_free(m); fclose(f); return NULL; }
        if (kmamba_enable_training(m, opt_blocks, lr_embed_head, weight_decay)) {
            kmamba_free(m); fclose(f); return NULL;
        }
    }

    if (read_floats(f, m->embedding, cfg.vocab_size * cfg.dim) ||
        read_floats(f, m->head, cfg.dim * cfg.vocab_size)) {
        kmamba_free(m); fclose(f); return NULL;
    }

    for (size_t i = 0; i < cfg.n_layers; i++) {
        MambaBlock *b = m->layers[i];
        size_t theta_size = b->config.state_size / 2; if (theta_size == 0) theta_size = 1;
        if (read_floats(f, b->W_in.data,       b->W_in.rows * b->W_in.cols)            ||
            read_floats(f, b->W_out.data,      b->W_out.rows * b->W_out.cols)           ||
            read_floats(f, b->A_log.data,      b->A_log.rows * b->A_log.cols)           ||
            read_floats(f, b->W_B.data,        b->W_B.rows * b->W_B.cols)               ||
            read_floats(f, b->W_C.data,        b->W_C.rows * b->W_C.cols)               ||
            read_floats(f, b->b_B,             b->W_B.rows)                              ||
            read_floats(f, b->b_C,             b->W_C.rows)                              ||
            read_floats(f, b->delta_proj.data,  b->delta_proj.rows * b->delta_proj.cols) ||
            read_floats(f, b->lambda_proj.data, b->lambda_proj.rows * b->lambda_proj.cols)||
            read_floats(f, b->theta,           theta_size)) {
            kmamba_free(m); fclose(f); return NULL;
        }
    }

    fclose(f);
    return m;
}

/* ========= forward ========= */

int kmamba_forward(KMamba *m, const uint8_t *tokens, float *logits_out) {
    if (!m || !tokens || !logits_out) return -1;

    size_t L = m->cfg.seq_len;
    size_t D = m->cfg.dim;

    float *buf0 = (float *)xcalloc(L * D, sizeof(float));
    float *buf1 = (float *)xcalloc(L * D, sizeof(float));

    embed_lookup(m, buf0, tokens);

    const float *cur = buf0;
    float *next = buf1;
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        mamba_block_forward(m->layers[i], next, cur, 1);
        const float *tmp = cur; cur = next; next = (float *)tmp;
    }

    gemm_avx2((float *)cur, m->head, logits_out,
              (long)L, (long)D, (long)m->cfg.vocab_size);

    free(buf0);
    free(buf1);
    return 0;
}

/* ========= training ========= */

float kmamba_train_step(KMamba *m, const uint8_t *tokens_plus1) {
    if (!m || !tokens_plus1 || !m->for_training) return NAN;

    size_t V = m->cfg.vocab_size;
    size_t L = m->cfg.seq_len;
    size_t D = m->cfg.dim;

    const uint8_t *tokens_in  = tokens_plus1;
    const uint8_t *tokens_tgt = tokens_plus1 + 1;

    float *acts    = (float *)xcalloc((m->cfg.n_layers + 1) * L * D, sizeof(float));
    float *d_hidden = (float *)xcalloc(L * D, sizeof(float));
    float *logits  = (float *)xcalloc(L * V, sizeof(float));
    float *probs   = (float *)xcalloc(V, sizeof(float));
    float *dlogits = (float *)xcalloc(L * V, sizeof(float));
    float *head_T  = (float *)xcalloc(V * D, sizeof(float));
    float *hidden_T = (float *)xcalloc(D * L, sizeof(float));
    float *g_head  = (float *)xcalloc(D * V, sizeof(float));
    float *g_embed = (float *)xcalloc(V * D, sizeof(float));

    embed_lookup(m, &acts[0], tokens_in);
    for (size_t i = 0; i < m->cfg.n_layers; i++)
        mamba_block_forward(m->layers[i], &acts[(i + 1) * L * D], &acts[i * L * D], 1);

    const float *hidden = &acts[m->cfg.n_layers * L * D];
    gemm_avx2((float *)hidden, m->head, logits, (long)L, (long)D, (long)V);

    float loss = 0.0f;
    float invL = 1.0f / (float)L;
    for (size_t t = 0; t < L; t++) {
        softmax(probs, &logits[t * V], V);
        float p = probs[(size_t)tokens_tgt[t]];
        if (p < 1e-20f) p = 1e-20f;
        loss += -logf(p);

        float *dl = &dlogits[t * V];
        for (size_t i = 0; i < V; i++) dl[i] = probs[i] * invL;
        dl[(size_t)tokens_tgt[t]] -= invL;
    }
    loss *= invL;

    transpose(m->head, head_T, D, V);
    gemm_avx2(dlogits, head_T, d_hidden, (long)L, (long)V, (long)D);

    transpose(hidden, hidden_T, L, D);
    gemm_avx2(hidden_T, dlogits, g_head, (long)D, (long)L, (long)V);

    for (size_t i = 0; i < m->cfg.n_layers; i++) mamba_zero_grads(m->layers[i]);

    float *d_buf = (float *)xcalloc(L * D, sizeof(float));
    float *dcur  = d_hidden;
    float *dnext = d_buf;
    for (size_t li = m->cfg.n_layers; li-- > 0;) {
        memset(dnext, 0, L * D * sizeof(float));
        mamba_backward(m->layers[li], dcur, &acts[li * L * D], dnext, 0);
        float *tmp = dcur; dcur = dnext; dnext = tmp;
    }

    for (size_t t = 0; t < L; t++) {
        float *g = &g_embed[(size_t)tokens_in[t] * D];
        const float *d = &dcur[t * D];
        for (size_t j = 0; j < D; j++) g[j] += d[j];
    }

    {
        double grad_sq = sum_squares_f32(g_head, D * V)
                       + sum_squares_f32(g_embed, V * D);
        for (size_t i = 0; i < m->cfg.n_layers; i++)
            grad_sq += (double)mamba_block_grad_sqnorm(m->layers[i]);
        m->last_grad_norm = sqrtf((float)grad_sq);
        m->last_grad_over_clip = (m->opt_blocks.clip_norm > 0.0f)
                               ? (m->last_grad_norm / m->opt_blocks.clip_norm)
                               : 0.0f;
        m->last_grad_would_clip = (m->opt_blocks.clip_norm > 0.0f &&
                                   m->last_grad_norm > m->opt_blocks.clip_norm);
    }

    for (size_t i = 0; i < m->cfg.n_layers; i++)
        mamba_optimizer_step(m->layers[i], &m->opt_blocks);

    m->step_embed_head++;
    ADAM_EH_UPDATE(m, m->embedding, g_embed, m->m_embedding, m->v_embedding, V * D);
    ADAM_EH_UPDATE(m, m->head,      g_head,  m->m_head,      m->v_head,      D * V);

    free(acts); free(d_hidden); free(logits); free(probs);
    free(dlogits); free(head_T); free(hidden_T);
    free(g_head); free(g_embed); free(d_buf);

    return loss;
}

float kmamba_train_batch(KMamba *m, const uint8_t *batch_tokens, size_t batch_size) {
    if (!m || !batch_tokens || !m->for_training || batch_size == 0) return NAN;

    size_t V = m->cfg.vocab_size;
    size_t L = m->cfg.seq_len;
    size_t D = m->cfg.dim;
    size_t Lp1 = L + 1;
    size_t n_layers = m->cfg.n_layers;
    float invB = 1.0f / (float)batch_size;
    float invL = 1.0f / (float)L;
    float invBL = invB * invL;

    float *g_head  = (float *)xcalloc(D * V, sizeof(float));
    float *g_embed = (float *)xcalloc(V * D, sizeof(float));
    float *head_T  = (float *)xcalloc(V * D, sizeof(float));
    MambaBlockWorkspace ***thread_ws = NULL;
    int n_threads = 1;

    if (!g_head || !g_embed || !head_T) {
        free(g_head); free(g_embed); free(head_T);
        return NAN;
    }

    transpose(m->head, head_T, D, V);

    for (size_t i = 0; i < n_layers; i++) mamba_zero_grads(m->layers[i]);

#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    thread_ws = (MambaBlockWorkspace ***)xcalloc((size_t)n_threads, sizeof(*thread_ws));
    for (int t = 0; t < n_threads; t++) {
        thread_ws[t] = (MambaBlockWorkspace **)xcalloc(n_layers, sizeof(*thread_ws[t]));
        for (size_t li = 0; li < n_layers; li++) {
            thread_ws[t][li] = mamba_block_workspace_create(m->layers[li]);
            if (!thread_ws[t][li]) {
                fprintf(stderr, "OOM\n");
                abort();
            }
        }
    }

    float total_loss = 0.0f;

#pragma omp parallel reduction(+:total_loss)
    {
        int tid = 0;
        MambaBlockWorkspace **layer_ws;
        float *acts     = (float *)xcalloc((n_layers + 1) * L * D, sizeof(float));
        float *logits   = (float *)xcalloc(L * V, sizeof(float));
        float *probs    = (float *)xcalloc(V, sizeof(float));
        float *dlogits  = (float *)xcalloc(L * V, sizeof(float));
        float *d_hidden = (float *)xcalloc(L * D, sizeof(float));
        float *d_buf    = (float *)xcalloc(L * D, sizeof(float));
        float *hidden_T = (float *)xcalloc(D * L, sizeof(float));
        float *g_head_b = (float *)xcalloc(D * V, sizeof(float));

#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        layer_ws = thread_ws[tid];

#pragma omp for schedule(dynamic)
        for (size_t b = 0; b < batch_size; b++) {
            const uint8_t *seq     = &batch_tokens[b * Lp1];
            const uint8_t *tok_in  = seq;
            const uint8_t *tok_tgt = seq + 1;

            embed_lookup(m, &acts[0], tok_in);
            for (size_t i = 0; i < n_layers; i++) {
                mamba_block_forward_ws(m->layers[i], layer_ws[i],
                                       &acts[(i + 1) * L * D],
                                       &acts[i * L * D], 1);
            }

            const float *hidden = &acts[n_layers * L * D];
            memset(logits, 0, L * V * sizeof(float));
            memset(dlogits, 0, L * V * sizeof(float));
            gemm_avx2((float *)hidden, m->head, logits, (long)L, (long)D, (long)V);

            float sample_loss = 0.0f;
            for (size_t t = 0; t < L; t++) {
                softmax(probs, &logits[t * V], V);
                float p = probs[(size_t)tok_tgt[t]];
                if (p < 1e-20f) p = 1e-20f;
                sample_loss += -logf(p);
                float *dl = &dlogits[t * V];
                for (size_t i = 0; i < V; i++) dl[i] = probs[i] * invBL;
                dl[(size_t)tok_tgt[t]] -= invBL;
            }
            total_loss += sample_loss * invL;

            memset(d_hidden, 0, L * D * sizeof(float));
            gemm_avx2(dlogits, head_T, d_hidden, (long)L, (long)V, (long)D);

            transpose(hidden, hidden_T, L, D);
            memset(g_head_b, 0, D * V * sizeof(float));
            gemm_avx2(hidden_T, dlogits, g_head_b, (long)D, (long)L, (long)V);

#pragma omp critical
            {
                for (size_t i = 0; i < D * V; i++) g_head[i] += g_head_b[i];

                float *dcur  = d_hidden;
                float *dnext = d_buf;
                for (size_t li = n_layers; li-- > 0;) {
                    memset(dnext, 0, L * D * sizeof(float));
                    mamba_backward_ws(m->layers[li], layer_ws[li],
                                      dcur, &acts[li * L * D], dnext, b);
                    float *tmp = dcur; dcur = dnext; dnext = tmp;
                }

                for (size_t t = 0; t < L; t++) {
                    float *g = &g_embed[(size_t)tok_in[t] * D];
                    const float *d = &dcur[t * D];
                    for (size_t j = 0; j < D; j++) g[j] += d[j];
                }
            }
        }

        free(acts); free(logits); free(probs); free(dlogits);
        free(d_hidden); free(d_buf); free(hidden_T); free(g_head_b);
    }

    {
        double grad_sq = sum_squares_f32(g_head, D * V)
                       + sum_squares_f32(g_embed, V * D);
        for (size_t i = 0; i < n_layers; i++)
            grad_sq += (double)mamba_block_grad_sqnorm(m->layers[i]);
        m->last_grad_norm = sqrtf((float)grad_sq);
        m->last_grad_over_clip = (m->opt_blocks.clip_norm > 0.0f)
                               ? (m->last_grad_norm / m->opt_blocks.clip_norm)
                               : 0.0f;
        m->last_grad_would_clip = (m->opt_blocks.clip_norm > 0.0f &&
                                   m->last_grad_norm > m->opt_blocks.clip_norm);
    }

    for (size_t i = 0; i < n_layers; i++)
        mamba_optimizer_step(m->layers[i], &m->opt_blocks);

    m->step_embed_head++;
    ADAM_EH_UPDATE(m, m->embedding, g_embed, m->m_embedding, m->v_embedding, V * D);
    ADAM_EH_UPDATE(m, m->head,      g_head,  m->m_head,      m->v_head,      D * V);

    for (int t = 0; t < n_threads; t++) {
        if (!thread_ws || !thread_ws[t]) continue;
        for (size_t li = 0; li < n_layers; li++)
            mamba_block_workspace_free(thread_ws[t][li]);
        free(thread_ws[t]);
    }
    free(thread_ws);

    free(g_head); free(g_embed); free(head_T);

    return total_loss * invB;
}
