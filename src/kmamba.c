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
    if (cols <= 0) return;
    
    float max_val = x[0];
    for (int i = 1; i < cols; i++) {
        if (x[i] > max_val && !isnan(x[i]) && !isinf(x[i])) max_val = x[i];
    }
    /* Protection contre max_val invalide */
    if (isnan(max_val) || isinf(max_val)) max_val = 0.0f;
    
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        float exp_val = expf(fmaxf(x[i] - max_val, -80.0f)); /* clamp pour éviter underflow */
        probs[i] = exp_val;
        sum += exp_val;
    }
    
    /* Protection division par zéro */
    if (sum < 1e-20f) sum = 1e-20f;
    
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

static void embed_lookup(const KMamba *m, float *out, const uint32_t *tokens) {
    size_t D = m->cfg.dim;
    size_t V = m->cfg.vocab_size;
    for (size_t t = 0; t < m->cfg.seq_len; t++) {
        uint32_t tok_id = tokens[t];
        if (tok_id < V) {
            memcpy(&out[t * D], &m->embedding[(size_t)tok_id * D], D * sizeof(float));
        } else {
            /* Out-of-vocab: zero embedding */
            memset(&out[t * D], 0, D * sizeof(float));
        }
    }
}

/* ========= create / free ========= */

KMamba* kmamba_create(const KMambaConfig *cfg) {
    /* Note: vocab_size can be 0 for continuous inputs */
    if (!cfg || !cfg->dim || !cfg->seq_len || !cfg->n_layers)
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

    /* Allocate embedding only if vocab_size > 0 (skip for continuous inputs) */
    if (m->cfg.vocab_size > 0) {
        m->embedding = (float *)xcalloc(m->cfg.vocab_size * m->cfg.dim, sizeof(float));
        /* Weight tying: head partage les poids avec embedding */
        if (m->cfg.weight_tying) {
            m->head = m->embedding;  /* Partage mémoire */
        } else {
            m->head = (float *)xcalloc(m->cfg.dim * m->cfg.vocab_size, sizeof(float));
        }
    } else {
        /* Continuous inputs: no embedding/head needed */
        m->embedding = NULL;
        m->head = NULL;
    }

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
    /* Éviter double-free si weight_tying */
    if (!m->cfg.weight_tying) {
        free(m->head);
    }
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
    /* Si pas de weight tying, initialiser head séparément */
    if (!m->cfg.weight_tying) {
        xavier_uniform(m->head, m->cfg.dim, m->cfg.vocab_size, m->cfg.dim * m->cfg.vocab_size);
    }
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

int kmamba_forward(KMamba *m, const uint32_t *tokens, float *logits_out) {
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

float kmamba_train_step(KMamba *m, const uint32_t *tokens_plus1) {
    if (!m || !tokens_plus1 || !m->for_training) return NAN;

    size_t V = m->cfg.vocab_size;
    size_t L = m->cfg.seq_len;
    size_t D = m->cfg.dim;

    const uint32_t *tokens_in  = tokens_plus1;
    const uint32_t *tokens_tgt = tokens_plus1 + 1;

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

float kmamba_train_batch(KMamba *m, const uint32_t *batch_tokens, size_t batch_size) {
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
            const uint32_t *seq     = &batch_tokens[b * Lp1];
            const uint32_t *tok_in  = seq;
            const uint32_t *tok_tgt = seq + 1;

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

/* ═════════════════════════════════════════════════════════════════════════════
 * GPU Training Implementation
 * ═════════════════════════════════════════════════════════════════════════════ */

#ifdef KMAMBA_BUILD_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Initialize GPU memory for KMamba (embedding, head, and optimizer states) */
int kmamba_gpu_init(KMamba *m) {
    if (!m) return -1;
    size_t V = m->cfg.vocab_size;
    size_t D = m->cfg.dim;

    cudaError_t err;
    err = cudaMalloc(&m->gpu.d_embedding, V * D * sizeof(float));
    if (err != cudaSuccess) return -1;
    err = cudaMalloc(&m->gpu.d_head, D * V * sizeof(float));
    if (err != cudaSuccess) { cudaFree(m->gpu.d_embedding); return -1; }

    /* Copy parameters to device */
    cudaMemcpy(m->gpu.d_embedding, m->embedding, V * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m->gpu.d_head, m->head, D * V * sizeof(float), cudaMemcpyHostToDevice);

    /* Allocate optimizer states on device */
    if (m->m_embedding && m->v_embedding) {
        err = cudaMalloc(&m->gpu.d_m_embed, V * D * sizeof(float));
        if (err == cudaSuccess) cudaMemset(m->gpu.d_m_embed, 0, V * D * sizeof(float));
        err = cudaMalloc(&m->gpu.d_v_embed, V * D * sizeof(float));
        if (err == cudaSuccess) cudaMemset(m->gpu.d_v_embed, 0, V * D * sizeof(float));
    }
    if (m->m_head && m->v_head) {
        err = cudaMalloc(&m->gpu.d_m_head, D * V * sizeof(float));
        if (err == cudaSuccess) cudaMemset(m->gpu.d_m_head, 0, D * V * sizeof(float));
        err = cudaMalloc(&m->gpu.d_v_head, D * V * sizeof(float));
        if (err == cudaSuccess) cudaMemset(m->gpu.d_v_head, 0, D * V * sizeof(float));
    }

    m->gpu.gpu_ready = 1;
    return 0;
}

/* Free GPU memory for KMamba */
void kmamba_gpu_free(KMamba *m) {
    if (!m || !m->gpu.gpu_ready) return;
    cudaFree(m->gpu.d_embedding);
    cudaFree(m->gpu.d_head);
    cudaFree(m->gpu.d_m_embed);
    cudaFree(m->gpu.d_v_embed);
    cudaFree(m->gpu.d_m_head);
    cudaFree(m->gpu.d_v_head);
    m->gpu.gpu_ready = 0;
}

/* Sync host parameters from device after training */
void kmamba_gpu_sync_to_host(KMamba *m) {
    if (!m || !m->gpu.gpu_ready) return;
    size_t V = m->cfg.vocab_size;
    size_t D = m->cfg.dim;
    cudaMemcpy(m->embedding, m->gpu.d_embedding, V * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m->head, m->gpu.d_head, D * V * sizeof(float), cudaMemcpyDeviceToHost);
}

/* Single training step on GPU (simplified: no batching, single sequence) */
float kmamba_train_step_gpu(KMamba *m, const uint32_t *tokens_plus1) {
    if (!m || !m->for_training || !tokens_plus1) return 0.0f;
    if (!m->gpu.gpu_ready) {
        if (kmamba_gpu_init(m) != 0) return 0.0f;
    }

    size_t L = m->cfg.seq_len;
    size_t Lp1 = L + 1;
    size_t V = m->cfg.vocab_size;
    size_t D = m->cfg.dim;
    size_t n_layers = m->cfg.n_layers;

    const uint32_t *tok_in = tokens_plus1;
    const uint32_t *tok_tgt = tokens_plus1 + 1;

    /* Allocate device buffers */
    float *d_acts = NULL, *d_logits = NULL, *d_probs = NULL, *d_dlogits = NULL;
    float *d_dhidden = NULL, *d_dbuf = NULL, *d_dembed = NULL;
    cudaMalloc((void**)&d_acts, (n_layers + 1) * L * D * sizeof(float));
    cudaMalloc((void**)&d_logits, L * V * sizeof(float));
    cudaMalloc((void**)&d_probs, V * sizeof(float));
    cudaMalloc((void**)&d_dlogits, L * V * sizeof(float));
    cudaMalloc((void**)&d_dhidden, L * D * sizeof(float));
    cudaMalloc((void**)&d_dbuf, L * D * sizeof(float));
    cudaMalloc((void**)&d_dembed, V * D * sizeof(float));
    cudaMemset(d_dembed, 0, V * D * sizeof(float));

    /* Embedding lookup on GPU: d_acts[0] = d_embedding[tok_in] */
    /* Simplified: do on CPU and upload for now */
    float *h_acts0 = (float *)malloc(L * D * sizeof(float));
    for (size_t t = 0; t < L; t++)
        memcpy(&h_acts0[t * D], &m->embedding[(size_t)tok_in[t] * D], D * sizeof(float));
    cudaMemcpy(d_acts, h_acts0, L * D * sizeof(float), cudaMemcpyHostToDevice);
    free(h_acts0);

    /* Forward through layers on GPU */
    for (size_t i = 0; i < n_layers; i++) {
        /* Note: mamba_block_forward will auto-dispatch to GPU if backend=GPU */
        KMambaBackend saved_backend = kmamba_backend_preference;
        kmamba_backend_preference = KMAMBA_BACKEND_GPU;
        float *h_in = (float *)malloc(L * D * sizeof(float));
        float *h_out = (float *)malloc(L * D * sizeof(float));
        cudaMemcpy(h_in, d_acts + i * L * D, L * D * sizeof(float), cudaMemcpyDeviceToHost);
        mamba_block_forward(m->layers[i], h_out, h_in, 1);
        cudaMemcpy(d_acts + (i + 1) * L * D, h_out, L * D * sizeof(float), cudaMemcpyHostToDevice);
        free(h_in); free(h_out);
        kmamba_backend_preference = saved_backend;
    }

    /* logits = acts[n_layers] @ head */
    float *h_hidden = (float *)malloc(L * D * sizeof(float));
    cudaMemcpy(h_hidden, d_acts + n_layers * L * D, L * D * sizeof(float), cudaMemcpyDeviceToHost);
    float *h_logits = (float *)malloc(L * V * sizeof(float));
    gemm_f32(h_hidden, m->head, h_logits, (int)L, (int)D, (int)V);
    cudaMemcpy(d_logits, h_logits, L * V * sizeof(float), cudaMemcpyHostToDevice);
    free(h_hidden); free(h_logits);

    /* Compute loss and dlogits on GPU */
    float *h_logits_cpu = (float *)malloc(L * V * sizeof(float));
    cudaMemcpy(h_logits_cpu, d_logits, L * V * sizeof(float), cudaMemcpyDeviceToHost);
    float *h_dlogits = (float *)calloc(L * V, sizeof(float));
    float sample_loss = 0.0f;
    float invL = 1.0f / (float)L;
    for (size_t t = 0; t < L; t++) {
        float *probs = h_dlogits + t * V; /* reuse buffer for probs */
        softmax_f32(&h_logits_cpu[t * V], probs, 1, (int)V);
        float p = probs[(size_t)tok_tgt[t]];
        if (p < 1e-20f) p = 1e-20f;
        sample_loss += -logf(p);
        for (size_t i = 0; i < V; i++) h_dlogits[t * V + i] = probs[i] * invL;
        h_dlogits[t * V + (size_t)tok_tgt[t]] -= invL;
    }
    cudaMemcpy(d_dlogits, h_dlogits, L * V * sizeof(float), cudaMemcpyHostToDevice);
    free(h_logits_cpu); free(h_dlogits);

    /* Backward: d_hidden = dlogits @ head^T */
    /* g_head = hidden^T @ dlogits */
    /* (Simplified: do on CPU for now) */
    float *h_hidden_all = (float *)malloc(L * D * sizeof(float));
    cudaMemcpy(h_hidden_all, d_acts + n_layers * L * D, L * D * sizeof(float), cudaMemcpyDeviceToHost);
    float *h_dlogits2 = (float *)malloc(L * V * sizeof(float));
    cudaMemcpy(h_dlogits2, d_dlogits, L * V * sizeof(float), cudaMemcpyDeviceToHost);

    float *h_dhidden = (float *)malloc(L * D * sizeof(float));
    gemm_f32_ABt(h_dlogits2, m->head, h_dhidden, (int)L, (int)D, (int)V);

    float *h_g_head = (float *)calloc(D * V, sizeof(float));
    gemm_f32_AtB(h_hidden_all, h_dlogits2, h_g_head, (int)D, (int)V, (int)L);

    cudaMemcpy(d_dhidden, h_dhidden, L * D * sizeof(float), cudaMemcpyHostToDevice);

    /* Backward through layers (GPU dispatch) */
    float *dcur = d_dhidden;
    float *dnext = d_dbuf;
    for (size_t li = n_layers; li-- > 0;) {
        cudaMemset(dnext, 0, L * D * sizeof(float));
        /* Note: backward GPU dispatch would go here */
        /* For now: copy to CPU, backward, copy back */
        float *h_dcur = (float *)malloc(L * D * sizeof(float));
        float *h_dnext = (float *)malloc(L * D * sizeof(float));
        float *h_acts_li = (float *)malloc(L * D * sizeof(float));
        cudaMemcpy(h_dcur, dcur, L * D * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_acts_li, d_acts + li * L * D, L * D * sizeof(float), cudaMemcpyDeviceToHost);

        MambaBlockWorkspace *ws = mamba_block_workspace_create(m->layers[li]);
        mamba_backward_ws(m->layers[li], ws, h_dcur, h_acts_li, h_dnext, 0);
        mamba_block_workspace_free(ws);

        cudaMemcpy(dnext, h_dnext, L * D * sizeof(float), cudaMemcpyHostToDevice);
        free(h_dcur); free(h_dnext); free(h_acts_li);

        float *tmp = dcur; dcur = dnext; dnext = tmp;
    }

    /* Accumulate embedding gradients */
    float *h_dcur_final = (float *)malloc(L * D * sizeof(float));
    cudaMemcpy(h_dcur_final, dcur, L * D * sizeof(float), cudaMemcpyDeviceToHost);
    float *h_g_embed = (float *)calloc(V * D, sizeof(float));
    for (size_t t = 0; t < L; t++) {
        float *g = &h_g_embed[(size_t)tok_in[t] * D];
        const float *d = &h_dcur_final[t * D];
        for (size_t j = 0; j < D; j++) g[j] += d[j];
    }
    cudaMemcpy(d_dembed, h_g_embed, V * D * sizeof(float), cudaMemcpyHostToDevice);
    free(h_dcur_final); free(h_g_embed);

    /* Optimizer step on GPU for embedding and head */
    m->step_embed_head++;
    adamw_step_f32(m->embedding, h_g_embed, m->m_embedding, m->v_embedding,
                   m->lr_embed_head, 0.9f, 0.999f, m->opt_blocks.eps, m->weight_decay,
                   V * D, m->step_embed_head);
    adamw_step_f32(m->head, h_g_head, m->m_head, m->v_head,
                   m->lr_embed_head, 0.9f, 0.999f, m->opt_blocks.eps, m->weight_decay,
                   D * V, m->step_embed_head);

    /* Sync updated params to GPU */
    cudaMemcpy(m->gpu.d_embedding, m->embedding, V * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m->gpu.d_head, m->head, D * V * sizeof(float), cudaMemcpyHostToDevice);

    free(h_g_head);
    free(h_hidden_all); free(h_dlogits2);

    /* Cleanup device buffers */
    cudaFree(d_acts); cudaFree(d_logits); cudaFree(d_probs);
    cudaFree(d_dlogits); cudaFree(d_dhidden); cudaFree(d_dbuf); cudaFree(d_dembed);

    return sample_loss / (float)L;
}

/* ============================================================
 * Hybrid batch training: embedding/head on CPU, blocks on GPU
 * ============================================================ */

#ifdef KMAMBA_BUILD_CUDA
#include <cublas_v2.h>

/* Déclaration externe de gpu_optimizer_step depuis cuda/mamba_block.cu */
#ifdef __cplusplus
extern "C" {
#endif
void gpu_optimizer_step(MambaBlock *block, const MBOptimConfig *conf);
#ifdef __cplusplus
}
#endif

/* Global cuBLAS handle for hybrid mode */
static cublasHandle_t g_cublas_handle = NULL;
static int g_cublas_initialized = 0;

static cublasHandle_t get_cublas_handle(void) {
    if (!g_cublas_initialized) {
        /* Initialize CUDA device and context first */
        cudaError_t cuda_err = cudaSetDevice(0);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "[CUDA] Failed to set device: %s\n", cudaGetErrorString(cuda_err));
            return NULL;
        }
        cuda_err = cudaFree(0);  /* Initialize context */
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "[CUDA] Failed to initialize context: %s\n", cudaGetErrorString(cuda_err));
            return NULL;
        }
        cublasStatus_t status = cublasCreate(&g_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[cuBLAS] Failed to create handle: %d\n", (int)status);
            return NULL;
        }
        /* Set cuBLAS workspace to avoid allocation failures */
        void *workspace;
        cudaMalloc(&workspace, 1024 * 1024 * 16);  /* 16MB workspace */
        cublasSetWorkspace(g_cublas_handle, workspace, 1024 * 1024 * 16);
        g_cublas_initialized = 1;
        fprintf(stderr, "[cuBLAS] Handle initialized successfully with 16MB workspace\n");
    }
    return g_cublas_handle;
}

/* Helper: allocate and copy param to GPU */
static float* gpu_param_alloc(const float *host_data, size_t n) {
    float *d_param;
    cudaMalloc((void**)&d_param, n * sizeof(float));
    cudaMemcpy(d_param, host_data, n * sizeof(float), cudaMemcpyHostToDevice);
    return d_param;
}

/* Wrapper: convert MambaBlock pointers to device pointers for cuda_block_forward */
static void hybrid_block_forward(MambaBlock *block, float *d_output, float *d_input, int L, int B) {
    cublasHandle_t handle = get_cublas_handle();
    if (!handle) return;
    size_t D = block->config.dim;
    size_t N = block->config.state_size;
    size_t R = block->config.mimo_rank > 0 ? block->config.mimo_rank : 1;
    size_t NR = N * R;
    size_t TS = D / 2;
    
    /* Allocate param buffers on GPU and copy from host */
    float *d_W_in = gpu_param_alloc(block->W_in.data, R * D);
    float *d_W_out = gpu_param_alloc(block->W_out.data, D * R);
    float *d_A_log = gpu_param_alloc(block->A_log.data, N);
    float *d_W_B = gpu_param_alloc(block->W_B.data, NR * D);
    float *d_W_C = gpu_param_alloc(block->W_C.data, NR * D);
    float *d_delta_proj = gpu_param_alloc(block->delta_proj.data, D);
    float *d_theta = block->theta ? gpu_param_alloc(block->theta, TS) : NULL;
    float *d_lambda_proj = gpu_param_alloc(block->lambda_proj.data, D);
    
    /* Allocate temp buffers */
    float *d_u_raw, *d_u, *d_dt_raw, *d_dt, *d_B_exp, *d_C_exp, *d_h_store, *d_y_scan, *d_y_proj;
    float *d_lambda_raw, *d_lambda;
    cudaMalloc((void**)&d_u_raw, L * R * sizeof(float));
    cudaMalloc((void**)&d_u, L * R * sizeof(float));
    cudaMalloc((void**)&d_dt_raw, L * sizeof(float));
    cudaMalloc((void**)&d_dt, L * sizeof(float));
    cudaMalloc((void**)&d_B_exp, L * NR * sizeof(float));
    cudaMalloc((void**)&d_C_exp, L * NR * sizeof(float));
    cudaMalloc((void**)&d_h_store, L * R * sizeof(float));
    cudaMalloc((void**)&d_y_scan, L * R * sizeof(float));
    cudaMalloc((void**)&d_y_proj, L * D * sizeof(float));
    cudaMalloc((void**)&d_lambda_raw, L * sizeof(float));
    cudaMalloc((void**)&d_lambda, L * sizeof(float));
    
    cuda_block_forward(handle, d_W_in, d_W_out, d_A_log, d_W_B, d_W_C, d_delta_proj, d_theta,
                       d_lambda_proj, d_input, d_output, d_u_raw, d_u, d_dt_raw, d_dt,
                       d_B_exp, d_C_exp, d_lambda_raw, d_lambda, d_h_store, d_y_scan, d_y_proj,
                       L, (int)N, (int)D, (int)R);
    
    /* Free params */
    cudaFree(d_W_in); cudaFree(d_W_out); cudaFree(d_A_log);
    cudaFree(d_W_B); cudaFree(d_W_C); cudaFree(d_delta_proj);
    if (d_theta) cudaFree(d_theta);
    cudaFree(d_lambda_proj);
    
    /* Free temp buffers */
    cudaFree(d_u_raw); cudaFree(d_u); cudaFree(d_dt_raw); cudaFree(d_dt);
    cudaFree(d_B_exp); cudaFree(d_C_exp); cudaFree(d_h_store); cudaFree(d_y_scan);
    cudaFree(d_y_proj); cudaFree(d_lambda_raw); cudaFree(d_lambda);
}

static void hybrid_block_backward(MambaBlock *block, float *d_doutput, float *d_dinput,
                                   float *d_x, int L, int B) {
    cublasHandle_t handle = get_cublas_handle();
    if (!handle) return;
    size_t D = block->config.dim;
    size_t N = block->config.state_size;
    size_t R = block->config.mimo_rank > 0 ? block->config.mimo_rank : 1;
    size_t NR = N * R;
    size_t TS = D / 2;

    MBOptimState *s = (MBOptimState *)block->opt_state;
    if (!s) return;

    /* Allocate param buffers on GPU */
    float *d_W_in = gpu_param_alloc(block->W_in.data, R * D);
    float *d_W_out = gpu_param_alloc(block->W_out.data, D * R);
    float *d_A_log = gpu_param_alloc(block->A_log.data, N);
    float *d_W_B = gpu_param_alloc(block->W_B.data, NR * D);
    float *d_W_C = gpu_param_alloc(block->W_C.data, NR * D);
    float *d_delta_proj = gpu_param_alloc(block->delta_proj.data, D);
    float *d_theta = block->theta ? gpu_param_alloc(block->theta, TS) : NULL;
    float *d_lambda_proj = gpu_param_alloc(block->lambda_proj.data, D);

    /* Allocate temp buffers */
    float *d_dy, *d_y_scan, *d_du, *d_u_raw, *d_dt_raw, *d_ddt_raw, *d_ddt, *d_dA_tmp;
    float *d_B_exp, *d_C_exp, *d_dB_scan, *d_dC_scan, *d_ddt_scan, *d_dlambda, *d_dlambda_raw;
    cudaMalloc((void**)&d_dy, L * D * sizeof(float));
    cudaMalloc((void**)&d_y_scan, L * R * sizeof(float));
    cudaMalloc((void**)&d_du, L * R * sizeof(float));
    cudaMalloc((void**)&d_u_raw, L * R * sizeof(float));
    cudaMalloc((void**)&d_dt_raw, L * sizeof(float));
    cudaMalloc((void**)&d_ddt_raw, L * sizeof(float));
    cudaMalloc((void**)&d_ddt, L * sizeof(float));
    cudaMalloc((void**)&d_dA_tmp, N * sizeof(float));
    cudaMalloc((void**)&d_B_exp, L * NR * sizeof(float));
    cudaMalloc((void**)&d_C_exp, L * NR * sizeof(float));
    cudaMalloc((void**)&d_dB_scan, L * NR * sizeof(float));
    cudaMalloc((void**)&d_dC_scan, L * NR * sizeof(float));
    cudaMalloc((void**)&d_ddt_scan, L * N * sizeof(float));
    cudaMalloc((void**)&d_dlambda, L * sizeof(float));
    cudaMalloc((void**)&d_dlambda_raw, L * sizeof(float));

    /* Copy d_doutput to d_dy */
    cudaMemcpy(d_dy, d_doutput, L * D * sizeof(float), cudaMemcpyDeviceToDevice);

    /* Allocate GPU buffers for gradients */
    float *d_g_W_in = gpu_param_alloc(s->g_W_in, R * D);  /* Will be overwritten */
    float *d_g_W_out = gpu_param_alloc(s->g_W_out, D * R);
    float *d_g_A_log = gpu_param_alloc(s->g_A_log, N);
    float *d_g_W_B = gpu_param_alloc(s->g_W_B, NR * D);
    float *d_g_W_C = gpu_param_alloc(s->g_W_C, NR * D);
    float *d_g_delta_proj = gpu_param_alloc(s->g_delta_proj, D);
    float *d_g_theta = s->g_theta ? gpu_param_alloc(s->g_theta, TS) : NULL;
    float *d_g_lambda_proj = gpu_param_alloc(s->g_lambda_proj, D);

    /* Zero GPU gradients */
    cudaMemset(d_g_W_in, 0, R * D * sizeof(float));
    cudaMemset(d_g_W_out, 0, D * R * sizeof(float));
    cudaMemset(d_g_A_log, 0, N * sizeof(float));
    cudaMemset(d_g_W_B, 0, NR * D * sizeof(float));
    cudaMemset(d_g_W_C, 0, NR * D * sizeof(float));
    cudaMemset(d_g_delta_proj, 0, D * sizeof(float));
    if (d_g_theta) cudaMemset(d_g_theta, 0, TS * sizeof(float));
    cudaMemset(d_g_lambda_proj, 0, D * sizeof(float));

    cuda_block_backward(handle, d_W_in, d_W_out, d_A_log, d_W_B, d_W_C, d_delta_proj, d_theta,
                        d_lambda_proj, d_x, d_dy, d_g_W_in, d_g_W_out, d_g_A_log,
                        d_g_W_B, d_g_W_C, d_g_delta_proj, d_g_theta, d_g_lambda_proj,
                        d_u_raw, d_du, d_dt_raw, d_ddt, d_B_exp, d_C_exp, NULL,
                        d_y_scan, d_dB_scan, d_dC_scan, d_ddt_scan, d_dA_tmp, d_dlambda,
                        d_dlambda_raw, L, (int)N, (int)D, (int)R);

    /* Copy gradients back to CPU and accumulate */
    float *h_g_W_in = (float *)malloc(R * D * sizeof(float));
    float *h_g_W_out = (float *)malloc(D * R * sizeof(float));
    float *h_g_A_log = (float *)malloc(N * sizeof(float));
    float *h_g_W_B = (float *)malloc(NR * D * sizeof(float));
    float *h_g_W_C = (float *)malloc(NR * D * sizeof(float));
    float *h_g_delta_proj = (float *)malloc(D * sizeof(float));
    float *h_g_theta = s->g_theta ? (float *)malloc(TS * sizeof(float)) : NULL;
    float *h_g_lambda_proj = (float *)malloc(D * sizeof(float));

    cudaMemcpy(h_g_W_in, d_g_W_in, R * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_W_out, d_g_W_out, D * R * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_A_log, d_g_A_log, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_W_B, d_g_W_B, NR * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_W_C, d_g_W_C, NR * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_delta_proj, d_g_delta_proj, D * sizeof(float), cudaMemcpyDeviceToHost);
    if (h_g_theta) cudaMemcpy(h_g_theta, d_g_theta, TS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_lambda_proj, d_g_lambda_proj, D * sizeof(float), cudaMemcpyDeviceToHost);

    /* Accumulate */
    for (size_t i = 0; i < R * D; i++) s->g_W_in[i] += h_g_W_in[i];
    for (size_t i = 0; i < D * R; i++) s->g_W_out[i] += h_g_W_out[i];
    for (size_t i = 0; i < N; i++) s->g_A_log[i] += h_g_A_log[i];
    for (size_t i = 0; i < NR * D; i++) s->g_W_B[i] += h_g_W_B[i];
    for (size_t i = 0; i < NR * D; i++) s->g_W_C[i] += h_g_W_C[i];
    for (size_t i = 0; i < D; i++) s->g_delta_proj[i] += h_g_delta_proj[i];
    if (s->g_theta && h_g_theta) for (size_t i = 0; i < TS; i++) s->g_theta[i] += h_g_theta[i];
    for (size_t i = 0; i < D; i++) s->g_lambda_proj[i] += h_g_lambda_proj[i];

    /* Free CPU temp */
    free(h_g_W_in); free(h_g_W_out); free(h_g_A_log); free(h_g_W_B); free(h_g_W_C);
    free(h_g_delta_proj); free(h_g_lambda_proj);
    if (h_g_theta) free(h_g_theta);

    /* Free GPU gradients */
    cudaFree(d_g_W_in); cudaFree(d_g_W_out); cudaFree(d_g_A_log);
    cudaFree(d_g_W_B); cudaFree(d_g_W_C); cudaFree(d_g_delta_proj);
    if (d_g_theta) cudaFree(d_g_theta);
    cudaFree(d_g_lambda_proj);

    /* Copy d_dy back to d_doutput for next layer */
    cudaMemcpy(d_doutput, d_dy, L * D * sizeof(float), cudaMemcpyDeviceToDevice);

    /* Free params */
    cudaFree(d_W_in); cudaFree(d_W_out); cudaFree(d_A_log);
    cudaFree(d_W_B); cudaFree(d_W_C); cudaFree(d_delta_proj);
    if (d_theta) cudaFree(d_theta);
    cudaFree(d_lambda_proj);

    /* Free temp buffers */
    cudaFree(d_dy); cudaFree(d_y_scan); cudaFree(d_du); cudaFree(d_u_raw);
    cudaFree(d_dt_raw); cudaFree(d_ddt_raw); cudaFree(d_ddt); cudaFree(d_dA_tmp);
    cudaFree(d_B_exp); cudaFree(d_C_exp);
    cudaFree(d_dB_scan); cudaFree(d_dC_scan); cudaFree(d_ddt_scan);
    cudaFree(d_dlambda); cudaFree(d_dlambda_raw);
}
#endif

float kmamba_train_batch_hybrid(KMamba *m, const uint32_t *batch_tokens, size_t batch_size) {
#ifdef KMAMBA_BUILD_CUDA
    if (!m || !batch_tokens || !m->for_training || batch_size == 0) return NAN;
    if (!m->gpu.gpu_ready && kmamba_gpu_init(m) != 0) return NAN;

    size_t V = m->cfg.vocab_size;
    size_t L = m->cfg.seq_len;
    size_t D = m->cfg.dim;
    size_t Lp1 = L + 1;
    size_t n_layers = m->cfg.n_layers;
    float invB = 1.0f / (float)batch_size;
    float invL = 1.0f / (float)L;

    /* Initialize cuBLAS handle explicitly before training */
    cublasHandle_t handle = get_cublas_handle();
    if (!handle) {
        fprintf(stderr, "[ERROR] Failed to initialize cuBLAS handle\n");
        return NAN;
    }

    /* Allocate device buffers once */
    float *d_acts = NULL, *d_logits = NULL, *d_hidden = NULL;
    cudaMalloc((void**)&d_acts, (n_layers + 1) * L * D * sizeof(float));
    cudaMalloc((void**)&d_logits, L * V * sizeof(float));
    cudaMalloc((void**)&d_hidden, L * D * sizeof(float));

    /* Gradient accumulators on CPU */
    float *g_head = (float *)xcalloc(D * V, sizeof(float));
    float *g_embed = (float *)xcalloc(V * D, sizeof(float));

    float total_loss = 0.0f;

    for (size_t b = 0; b < batch_size; b++) {
        const uint32_t *seq = &batch_tokens[b * Lp1];
        const uint32_t *tok_in = seq;
        const uint32_t *tok_tgt = seq + 1;

        /* 1. Embedding lookup on CPU */
        float *h_embed = (float *)malloc(L * D * sizeof(float));
        for (size_t t = 0; t < L; t++) {
            memcpy(&h_embed[t * D], &m->embedding[(size_t)tok_in[t] * D], D * sizeof(float));
        }

        /* 2. Upload to GPU */
        cudaMemcpy(d_acts, h_embed, L * D * sizeof(float), cudaMemcpyHostToDevice);
        free(h_embed);

        /* 3. Forward blocks on GPU */
        for (size_t i = 0; i < n_layers; i++) {
            hybrid_block_forward(m->layers[i], d_acts + (i + 1) * L * D,
                              d_acts + i * L * D, L, 1);
        }

        /* 4. Download hidden for head computation on CPU */
        float *h_hidden = (float *)malloc(L * D * sizeof(float));
        cudaMemcpy(h_hidden, d_acts + n_layers * L * D, L * D * sizeof(float),
                   cudaMemcpyDeviceToHost);

        /* 5. Head GEMM on CPU */
        float *h_logits = (float *)malloc(L * V * sizeof(float));
        gemm_f32(h_hidden, m->head, h_logits, (int)L, (int)D, (int)V);

        /* 6. Loss and dlogits on CPU */
        float *h_dlogits = (float *)calloc(L * V, sizeof(float));
        float sample_loss = 0.0f;
        float *probs = (float *)malloc(V * sizeof(float));
        for (size_t t = 0; t < L; t++) {
            softmax_f32(&h_logits[t * V], probs, 1, (int)V);
            float p = probs[(size_t)tok_tgt[t]];
            if (p < 1e-20f) p = 1e-20f;
            sample_loss += -logf(p);
            for (size_t i = 0; i < V; i++) h_dlogits[t * V + i] = probs[i] * invB * invL;
            h_dlogits[t * V + (size_t)tok_tgt[t]] -= invB * invL;
        }
        free(probs);
        total_loss += sample_loss * invL;

        /* 7. d_hidden = dlogits @ head^T on CPU */
        float *h_dhidden = (float *)malloc(L * D * sizeof(float));
        gemm_f32_ABt(h_dlogits, m->head, h_dhidden, (int)L, (int)D, (int)V);

        /* 8. g_head += hidden^T @ dlogits on CPU */
        gemm_f32_AtB(h_hidden, h_dlogits, g_head, (int)D, (int)V, (int)L);

        /* 9. Upload d_hidden to GPU for backward */
        cudaMemcpy(d_hidden, h_dhidden, L * D * sizeof(float), cudaMemcpyHostToDevice);
        free(h_hidden); free(h_logits); free(h_dlogits); free(h_dhidden);

        /* 10. Backward blocks on GPU */
        float *d_dhidden = d_hidden;
        for (size_t li = n_layers; li-- > 0;) {
            float *d_dprev = (li == 0) ? NULL : d_dhidden;
            hybrid_block_backward(m->layers[li], d_dhidden, d_dprev,
                               d_acts + li * L * D, L, 1);
        }

        /* 11. Download d_embedding for this sample */
        float *h_dembed = (float *)malloc(L * D * sizeof(float));
        cudaMemcpy(h_dembed, d_dhidden, L * D * sizeof(float), cudaMemcpyDeviceToHost);

        /* 12. Accumulate embedding gradient */
        for (size_t t = 0; t < L; t++) {
            uint32_t tok = tok_in[t];
            for (size_t i = 0; i < D; i++) {
                g_embed[tok * D + i] += h_dembed[t * D + i];
            }
        }
        free(h_dembed);
    }

    /* Apply gradients with optimizer */
    kmamba_update_lr(m, m->opt_blocks.lr, m->lr_embed_head);

    /* Zero grads, apply, and clip */
    for (size_t i = 0; i < n_layers; i++) {
        gpu_optimizer_step(m->layers[i], &m->opt_blocks);
    }

    /* Sync embeddings and head to GPU */
    cudaMemcpy(m->gpu.d_embedding, m->embedding, V * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m->gpu.d_head, m->head, D * V * sizeof(float), cudaMemcpyHostToDevice);

    /* Cleanup */
    cudaFree(d_acts); cudaFree(d_logits); cudaFree(d_hidden);
    free(g_head); free(g_embed);

    return total_loss * invB;
#else
    (void)m; (void)batch_tokens; (void)batch_size;
    return NAN;
#endif
}

#endif /* KMAMBA_BUILD_CUDA */

/* ============================================================
 * Training state getters (for CSV logging)
 * ============================================================ */

float kmamba_last_grad_norm(const KMamba *m) {
    return m ? m->last_grad_norm : 0.0f;
}

float kmamba_last_grad_over_clip(const KMamba *m) {
    return m ? m->last_grad_over_clip : 0.0f;
}

int kmamba_last_grad_would_clip(const KMamba *m) {
    return m ? m->last_grad_would_clip : 0;
}

size_t kmamba_step_count(const KMamba *m) {
    return m ? m->step_embed_head : 0;
}

/* Update learning rates (for LR scheduler) */
void kmamba_update_lr(KMamba *m, float lr_blocks, float lr_embed) {
    if (!m) return;
    m->opt_blocks.lr = lr_blocks;
    m->lr_embed_head = lr_embed;
}
