#include "../include/kmamba.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ========= utils ========= */

static void *xcalloc(size_t n, size_t sz) {
    size_t total = n * sz;
    void *p = calloc(1, total);
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

/* Softmax inline: probs[i] = exp(x[i]) / sum(exp(x)) */
static inline void softmax_f32(const float *x, float *probs, int rows, int cols) {
    (void)rows;
    float max_val = x[0];
    for (int i = 1; i < cols; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        probs[i] = expf(x[i] - max_val);
        sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < cols; i++) probs[i] *= inv_sum;
}

/* GEMM variants for backward pass */
static inline void gemm_f32_ABt(const float *A, const float *B, float *C,
                                 int M, int N, int K) {
    /* C[M,N] = A[M,K] @ B^T[N,K] (B is [N,K], we use it transposed) */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

static inline void gemm_f32_AtB(const float *A, const float *B, float *C,
                                 int M, int N, int K) {
    /* C[M,N] = A^T[M,K] @ B[K,N] (A is [K,M], we use it transposed) */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
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

    

    KMamba *m = (KMamba *)calloc(1, sizeof(KMamba));
    if (!m) return NULL;

    m->cfg = *cfg;
    if (km_normalize_spatial_topology(&m->cfg.spatial_ndims,
                                      m->cfg.spatial_dims,
                                      m->cfg.seq_len,
                                      m->cfg.use_convnd,
                                      &m->cfg.convnd_ndims,
                                      m->cfg.convnd_K) != 0) {
        free(m);
        return NULL;
    }

    m->embedding = (float *)xcalloc(m->cfg.vocab_size * m->cfg.dim, sizeof(float));
    m->head      = (float *)xcalloc(m->cfg.dim * m->cfg.vocab_size, sizeof(float));

    m->layers = (MambaBlock **)calloc(m->cfg.n_layers, sizeof(MambaBlock *));
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        MBConfig bc = {
            .dim          = m->cfg.dim,
            .state_size   = m->cfg.state_size,
            .seq_len      = m->cfg.seq_len,
            .mimo_rank    = m->cfg.mimo_rank > 0 ? m->cfg.mimo_rank : 1,
            .dt_scale     = m->cfg.dt_scale,
            .dt_min       = m->cfg.dt_min,
            .dt_max       = m->cfg.dt_max,
            .spatial_ndims= m->cfg.spatial_ndims,
            .use_convnd   = m->cfg.use_convnd,
            .convnd_K     = m->cfg.convnd_K,
            .convnd_ndims = m->cfg.convnd_ndims
        };
        memcpy(bc.spatial_dims, m->cfg.spatial_dims, sizeof(bc.spatial_dims));
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
    uint64_t mimo_rank;
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
    h.version    = 4;
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
    size_t N = m->cfg.state_size;

    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        memset(m->layers[i]->hidden, 0, N * sizeof(float));
        memset(m->layers[i]->scan_h, 0, N * sizeof(float));
    }

    float *buf0 = (float *)xcalloc(L * D, sizeof(float));
    float *buf1 = (float *)xcalloc(L * D, sizeof(float));

    embed_lookup(m, buf0, tokens);

    const float *cur = buf0;
    float *next = buf1;
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        mamba_block_forward(m->layers[i], next, cur, 1);
        const float *tmp = cur; cur = next; next = (float *)tmp;
    }

    gemm_f32(cur, m->head, logits_out, (int)L, (int)D, (int)m->cfg.vocab_size);

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
    float *g_head  = (float *)xcalloc(D * V, sizeof(float));
    float *g_embed = (float *)xcalloc(V * D, sizeof(float));

    embed_lookup(m, &acts[0], tokens_in);
    for (size_t i = 0; i < m->cfg.n_layers; i++)
        mamba_block_forward(m->layers[i], &acts[(i + 1) * L * D], &acts[i * L * D], 1);

    const float *hidden = &acts[m->cfg.n_layers * L * D];
    gemm_f32(hidden, m->head, logits, (int)L, (int)D, (int)V);

    float loss = 0.0f;
    float invL = 1.0f / (float)L;
    for (size_t t = 0; t < L; t++) {
        softmax_f32(&logits[t * V], probs, 1, (int)V);
        float p = probs[(size_t)tokens_tgt[t]];
        if (p < 1e-20f) p = 1e-20f;
        loss += -logf(p);

        float *dl = &dlogits[t * V];
        for (size_t i = 0; i < V; i++) dl[i] = probs[i] * invL;
        dl[(size_t)tokens_tgt[t]] -= invL;
    }
    loss *= invL;

    /* d_hidden = dlogits @ head */
    gemm_f32(dlogits, m->head, d_hidden, (int)L, (int)V, (int)D); /* This needs to be checked, head is [D, V] */
    /* Re-check: dlogits [L, V], head [D, V]. We want dlogits [L, V] @ head^T [V, D] = [L, D] */
    gemm_f32_ABt(dlogits, m->head, d_hidden, (int)L, (int)D, (int)V);

    /* g_head = hidden^T @ dlogits = [D, L] @ [L, V] = [D, V] */
    gemm_f32_AtB(hidden, dlogits, g_head, (int)D, (int)V, (int)L);

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
        float gn_h = gradient_norm_f32(g_head, (int)(D * V));
        float gn_e = gradient_norm_f32(g_embed, (int)(V * D));
        double grad_sq = (double)gn_h * gn_h + (double)gn_e * gn_e;
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
    adamw_step_f32(m->embedding, g_embed, m->m_embedding, m->v_embedding, m->opt_blocks.lr, 0.9f, 0.999f, m->opt_blocks.eps, m->opt_blocks.weight_decay, V * D, m->step_embed_head);
    adamw_step_f32(m->head,      g_head,  m->m_head,      m->v_head,      m->opt_blocks.lr, 0.9f, 0.999f, m->opt_blocks.eps, m->opt_blocks.weight_decay, D * V, m->step_embed_head);

    free(acts); free(d_hidden); free(logits); free(probs);
    free(dlogits); free(g_head); free(g_embed); free(d_buf);

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
    
    MambaBlockWorkspace ***thread_ws = NULL;
    MBOptimState ***thread_local_grads = NULL;
    float **thread_g_head  = NULL;
    float **thread_g_embed = NULL;
    int n_threads = 1;

#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    thread_ws = (MambaBlockWorkspace ***)calloc((size_t)n_threads, sizeof(*thread_ws));
    thread_local_grads = (MBOptimState ***)calloc((size_t)n_threads, sizeof(*thread_local_grads));
    thread_g_head  = (float **)calloc((size_t)n_threads, sizeof(float *));
    thread_g_embed = (float **)calloc((size_t)n_threads, sizeof(float *));

    for (int t = 0; t < n_threads; t++) {
        thread_ws[t] = (MambaBlockWorkspace **)calloc(n_layers, sizeof(*thread_ws[t]));
        thread_local_grads[t] = (MBOptimState **)calloc(n_layers, sizeof(*thread_local_grads[t]));
        thread_g_head[t]  = (float *)xcalloc(D * V, sizeof(float));
        thread_g_embed[t] = (float *)xcalloc(V * D, sizeof(float));
        for (size_t li = 0; li < n_layers; li++) {
            thread_local_grads[t][li] = mamba_local_grad_alloc(m->layers[li]);
            thread_ws[t][li] = mamba_block_workspace_create(m->layers[li]);
        }
    }

    for (size_t i = 0; i < n_layers; i++) mamba_zero_grads(m->layers[i]);

    float total_loss = 0.0f;

#pragma omp parallel reduction(+:total_loss)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        MambaBlockWorkspace **layer_ws = thread_ws[tid];
        float *my_g_head  = thread_g_head[tid];
        float *my_g_embed = thread_g_embed[tid];
        MBOptimState **my_local_grads = thread_local_grads[tid];

        float *acts     = (float *)xcalloc((n_layers + 1) * L * D, sizeof(float));
        float *logits   = (float *)xcalloc(L * V, sizeof(float));
        float *probs    = (float *)xcalloc(V, sizeof(float));
        float *dlogits  = (float *)xcalloc(L * V, sizeof(float));
        float *d_hidden = (float *)xcalloc(L * D, sizeof(float));
        float *d_buf    = (float *)xcalloc(L * D, sizeof(float));

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
            gemm_f32(hidden, m->head, logits, (int)L, (int)D, (int)V);

            float sample_loss = 0.0f;
            for (size_t t = 0; t < L; t++) {
                softmax_f32(&logits[t * V], probs, 1, (int)V);
                float p = probs[(size_t)tok_tgt[t]];
                if (p < 1e-20f) p = 1e-20f;
                sample_loss += -logf(p);
                float *dl = &dlogits[t * V];
                for (size_t i = 0; i < V; i++) dl[i] = probs[i] * invBL;
                dl[(size_t)tok_tgt[t]] -= invBL;
            }
            total_loss += sample_loss * invL;

            /* d_hidden = dlogits [L, V] @ head^T [V, D] = [L, D] */
            gemm_f32_ABt(dlogits, m->head, d_hidden, (int)L, (int)D, (int)V);

            /* my_g_head = hidden^T [D, L] @ dlogits [L, V] = [D, V] */
            gemm_f32_AtB(hidden, dlogits, my_g_head, (int)D, (int)V, (int)L);

            float *dcur  = d_hidden;
            float *dnext = d_buf;
            for (size_t li = n_layers; li-- > 0;) {
                memset(dnext, 0, L * D * sizeof(float));
                mamba_backward_ws_local(m->layers[li], layer_ws[li],
                                        dcur, &acts[li * L * D], dnext, (size_t)b,
                                        my_local_grads[li]);
                float *tmp = dcur; dcur = dnext; dnext = tmp;
            }

            for (size_t t = 0; t < L; t++) {
                float *g = &my_g_embed[(size_t)tok_in[t] * D];
                const float *d = &dcur[t * D];
                for (size_t j = 0; j < D; j++) g[j] += d[j];
            }
        }

        free(acts); free(logits); free(probs); free(dlogits);
        free(d_hidden); free(d_buf);
    }

    for (int t = 0; t < n_threads; t++) {
        for (size_t i = 0; i < D * V; i++) g_head[i]  += thread_g_head[t][i];
        for (size_t i = 0; i < V * D; i++) g_embed[i] += thread_g_embed[t][i];
        for (size_t li = 0; li < n_layers; li++) {
            mamba_local_grad_reduce(m->layers[li], thread_local_grads[t][li]);
            mamba_local_grad_free(thread_local_grads[t][li]);
            mamba_block_workspace_free(thread_ws[t][li]);
        }
        free(thread_local_grads[t]);
        free(thread_ws[t]);
        free(thread_g_head[t]);
        free(thread_g_embed[t]);
    }
    free(thread_local_grads);
    free(thread_ws);
    free(thread_g_head);
    free(thread_g_embed);

    {
        float gn_h = gradient_norm_f32(g_head, (int)(D * V));
        float gn_e = gradient_norm_f32(g_embed, (int)(V * D));
        double grad_sq = (double)gn_h * gn_h + (double)gn_e * gn_e;
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
    adamw_step_f32(m->embedding, g_embed, m->m_embedding, m->v_embedding, m->opt_blocks.lr, 0.9f, 0.999f, m->opt_blocks.eps, m->opt_blocks.weight_decay, V * D, m->step_embed_head);
    adamw_step_f32(m->head,      g_head,  m->m_head,      m->v_head,      m->opt_blocks.lr, 0.9f, 0.999f, m->opt_blocks.eps, m->opt_blocks.weight_decay, D * V, m->step_embed_head);

    free(g_head); free(g_embed);

    return total_loss * invB;
}
