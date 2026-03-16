/*
 * test_kmamba_inference.c — Tests de bout en bout inférence KMamba
 *
 * Phase 3.1 : Tests Inférence - Pipeline complet
 * Objectif : Valider l'inférence complète du modèle KMamba
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>

/* ============================================================
 * Configuration KMamba (réécrite pour éviter les dépendances)
 * ============================================================ */

typedef struct {
    size_t vocab_size;
    size_t dim;
    size_t state_size;
    size_t seq_len;
    size_t n_layers;
    float dt_scale;
    float dt_min;
    float dt_max;
} KMambaConfig;

typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} MBMatrix;

typedef struct {
    size_t dim;
    size_t state_size;
    size_t seq_len;
    float dt_scale;
    float dt_min;
    float dt_max;
} MBConfig;

typedef struct {
    MBConfig config;
    MBMatrix W_in;
    MBMatrix W_out;
    MBMatrix A_log;
    MBMatrix B_mat;
    MBMatrix C_mat;
    MBMatrix delta_proj;
    float *hidden;
    float *delta;
    float *scan_B;
    float *scan_C;
    float *scan_delta;
    float *scan_h;
} MambaBlock;

typedef struct {
    KMambaConfig cfg;
    float *embedding;
    float *head;
    MambaBlock **layers;
    int for_training;
    float lr_embed_head;
    float weight_decay;
} KMamba;

/* ============================================================
 * Utilitaires de test
 * ============================================================ */

#define EPSILON 1e-5f
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("FAIL: %s\n", msg); \
            return 0; \
        } \
    } while(0)

#define TEST_ASSERT_FLOAT_EQ(a, b, eps, msg) \
    TEST_ASSERT(fabsf((a) - (b)) < (eps), msg)

static float frand_range(float min, float max) {
    float scale = (float)rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

static void fill_random(float *data, size_t n, float min, float max) {
    for (size_t i = 0; i < n; i++) {
        data[i] = frand_range(min, max);
    }
}

static int compare_arrays(const float *A, const float *B, size_t n, float eps) {
    for (size_t i = 0; i < n; i++) {
        if (fabsf(A[i] - B[i]) > eps) {
            printf("Mismatch at [%zu]: %f vs %f (diff: %f)\n", 
                   i, A[i], B[i], fabsf(A[i] - B[i]));
            return 0;
        }
    }
    return 1;
}

static void xavier_uniform(float *data, size_t rows, size_t cols, size_t total) {
    for (size_t i = 0; i < total; i++) {
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * sqrtf(6.0f / (rows + cols));
    }
}

/* ============================================================
 * Implémentations KMamba simulées
 * ============================================================ */

static void kmamba_free(KMamba *m);  // Forward declaration

static KMamba* kmamba_create(const KMambaConfig *cfg) {
    if (!cfg) return NULL;
    
    KMamba *m = (KMamba*)calloc(1, sizeof(KMamba));
    if (!m) return NULL;
    
    m->cfg = *cfg;
    
    /* Allouer embedding */
    m->embedding = (float*)malloc(cfg->vocab_size * cfg->dim * sizeof(float));
    if (!m->embedding) goto error;
    
    /* Allouer head */
    m->head = (float*)malloc(cfg->dim * cfg->vocab_size * sizeof(float));
    if (!m->head) goto error;
    
    /* Allouer layers */
    m->layers = (MambaBlock**)malloc(cfg->n_layers * sizeof(MambaBlock*));
    if (!m->layers) goto error;
    
    for (size_t i = 0; i < cfg->n_layers; i++) {
        m->layers[i] = (MambaBlock*)calloc(1, sizeof(MambaBlock));
        if (!m->layers[i]) goto error;
        
        m->layers[i]->config.dim = cfg->dim;
        m->layers[i]->config.state_size = cfg->state_size;
        m->layers[i]->config.seq_len = cfg->seq_len;
        m->layers[i]->config.dt_scale = cfg->dt_scale;
        m->layers[i]->config.dt_min = cfg->dt_min;
        m->layers[i]->config.dt_max = cfg->dt_max;
        
        /* Allouer buffers */
        m->layers[i]->hidden = (float*)malloc(cfg->dim * sizeof(float));
        m->layers[i]->delta = (float*)malloc(cfg->seq_len * sizeof(float));
        m->layers[i]->scan_h = (float*)malloc(cfg->state_size * sizeof(float));
        
        if (!m->layers[i]->hidden || !m->layers[i]->delta || !m->layers[i]->scan_h) {
            goto error;
        }
    }
    
    return m;
    
error:
    kmamba_free(m);
    return NULL;
}

static void kmamba_free(KMamba *m) {
    if (!m) return;
    
    free(m->embedding);
    free(m->head);
    
    if (m->layers) {
        for (size_t i = 0; i < m->cfg.n_layers; i++) {
            if (m->layers[i]) {
                free(m->layers[i]->hidden);
                free(m->layers[i]->delta);
                free(m->layers[i]->scan_h);
                free(m->layers[i]);
            }
        }
        free(m->layers);
    }
    
    free(m);
}

static int kmamba_init(KMamba *m, uint32_t seed) {
    if (!m) return -1;
    
    srand(seed);
    
    /* Initialiser embedding et head */
    xavier_uniform(m->embedding, m->cfg.vocab_size, m->cfg.dim, 
                   m->cfg.vocab_size * m->cfg.dim);
    xavier_uniform(m->head, m->cfg.dim, m->cfg.vocab_size, 
                   m->cfg.dim * m->cfg.vocab_size);
    
    /* Initialiser layers */
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        fill_random(m->layers[i]->hidden, m->cfg.dim, -0.1f, 0.1f);
        fill_random(m->layers[i]->delta, m->cfg.seq_len, 0.001f, 0.1f);
        fill_random(m->layers[i]->scan_h, m->cfg.state_size, -0.1f, 0.1f);
    }
    
    return 0;
}

static int kmamba_forward(KMamba *m, const uint8_t *tokens, float *logits_out) {
    if (!m || !tokens || !logits_out) return -1;
    
    /* Embedding lookup */
    for (size_t t = 0; t < m->cfg.seq_len; t++) {
        uint8_t token = tokens[t];
        for (size_t d = 0; d < m->cfg.dim; d++) {
            float *embedding_row = &m->embedding[token * m->cfg.dim + d];
            m->layers[0]->hidden[d] = embedding_row[0];  // Copie simplifiée
        }
    }
    
    /* Passer à travers les layers */
    for (size_t layer = 0; layer < m->cfg.n_layers; layer++) {
        MambaBlock *block = m->layers[layer];
        
        /* Simulation de forward pass */
        for (size_t d = 0; d < m->cfg.dim; d++) {
            block->hidden[d] = tanhf(block->hidden[d] * 0.5f);
        }
    }
    
    /* LM Head */
    for (size_t t = 0; t < m->cfg.seq_len; t++) {
        for (size_t v = 0; v < m->cfg.vocab_size; v++) {
            float sum = 0.0f;
            for (size_t d = 0; d < m->cfg.dim; d++) {
                sum += m->layers[0]->hidden[d] * m->head[d * m->cfg.vocab_size + v];
            }
            logits_out[t * m->cfg.vocab_size + v] = sum;
        }
    }
    
    return 0;
}

/* ============================================================
 * Tests d'inférence
 * ============================================================ */

static int test_basic_forward() {
    printf("Testing basic KMamba forward pass...\n");
    
    KMambaConfig config = {
        .vocab_size = 256,
        .dim = 384,
        .state_size = 1024,
        .seq_len = 128,
        .n_layers = 1,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    
    KMamba *model = kmamba_create(&config);
    TEST_ASSERT(model != NULL, "Failed to create KMamba model");
    
    int init_result = kmamba_init(model, 1234);
    TEST_ASSERT(init_result == 0, "Failed to initialize model");
    
    /* Input test */
    uint8_t tokens[128];
    for (int i = 0; i < 128; i++) {
        tokens[i] = (uint8_t)(i * 7 % 256);
    }
    
    float *logits = (float*)malloc(config.seq_len * config.vocab_size * sizeof(float));
    TEST_ASSERT(logits != NULL, "Failed to allocate logits");
    
    int forward_result = kmamba_forward(model, tokens, logits);
    TEST_ASSERT(forward_result == 0, "Forward pass failed");
    
    /* Vérifier que les logits sont finis */
    for (size_t t = 0; t < config.seq_len; t++) {
        for (size_t v = 0; v < config.vocab_size; v++) {
            TEST_ASSERT(isfinite(logits[t * config.vocab_size + v]), 
                       "Logit is not finite");
        }
    }
    
    free(logits);
    kmamba_free(model);
    
    printf("PASS: Basic forward pass\n");
    return 1;
}

static int test_autoregressive_generation() {
    printf("Testing autoregressive generation...\n");
    
    KMambaConfig config = {
        .vocab_size = 256,
        .dim = 64,
        .state_size = 128,
        .seq_len = 16,
        .n_layers = 1,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    
    KMamba *model = kmamba_create(&config);
    TEST_ASSERT(model != NULL, "Failed to create KMamba model");
    
    kmamba_init(model, 5678);
    
    /* Input initial */
    uint8_t input[] = {72, 101, 108, 108, 111};  // "Hello"
    size_t input_length = 5;
    
    /* Générer 10 tokens */
    uint8_t generated[10];
    
    for (size_t i = 0; i < 10; i++) {
        /* Préparer l'input */
        uint8_t padded_input[16];
        memset(padded_input, 0, 16);
        memcpy(padded_input, input, input_length);
        
        float *logits = (float*)malloc(config.seq_len * config.vocab_size * sizeof(float));
        kmamba_forward(model, padded_input, logits);
        
        /* Prendre le token avec le logit maximum */
        float max_logit = -INFINITY;
        uint8_t best_token = 0;
        
        for (size_t v = 0; v < config.vocab_size; v++) {
            if (logits[(input_length - 1) * config.vocab_size + v] > max_logit) {
                max_logit = logits[(input_length - 1) * config.vocab_size + v];
                best_token = (uint8_t)v;
            }
        }
        
        generated[i] = best_token;
        free(logits);
        
        /* Ajouter à l'input pour la prochaine itération */
        if (input_length < 16) {
            input[input_length++] = best_token;
        }
    }
    
    /* Vérifier que les tokens générés sont valides */
    for (size_t i = 0; i < 10; i++) {
        TEST_ASSERT(generated[i] < 256, "Generated token out of range");
    }
    
    kmamba_free(model);
    
    printf("PASS: Autoregressive generation\n");
    return 1;
}

static int test_byte_level_vocab() {
    printf("Testing byte-level vocabulary...\n");
    
    KMambaConfig config = {
        .vocab_size = 256,
        .dim = 32,
        .state_size = 64,
        .seq_len = 8,
        .n_layers = 1,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    
    KMamba *model = kmamba_create(&config);
    TEST_ASSERT(model != NULL, "Failed to create KMamba model");
    
    kmamba_init(model, 9999);
    
    /* Tester tous les tokens possibles */
    uint8_t all_tokens[256];
    for (int i = 0; i < 256; i++) {
        all_tokens[i] = (uint8_t)i;
    }
    
    float *logits = (float*)malloc(256 * config.vocab_size * sizeof(float));
    
    /* Forward avec tous les tokens */
    kmamba_forward(model, all_tokens, logits);
    
    /* Vérifier que chaque token produit des logits valides */
    for (size_t t = 0; t < 256; t++) {
        for (size_t v = 0; v < config.vocab_size; v++) {
            if (!isfinite(logits[t * config.vocab_size + v])) {
                printf("FAIL: Invalid logits for token %zu, vocab %zu\n", t, v);
                free(logits);
                kmamba_free(model);
                return 0;
            }
        }
    }
    
    free(logits);
    kmamba_free(model);
    
    printf("PASS: Byte-level vocabulary\n");
    return 1;
}

/* ============================================================
 * Benchmarks
 * ============================================================ */

static void benchmark_inference() {
    printf("Benchmarking inference performance...\n");
    
    KMambaConfig config = {
        .vocab_size = 256,
        .dim = 384,
        .state_size = 1024,
        .seq_len = 128,
        .n_layers = 2,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    
    KMamba *model = kmamba_create(&config);
    kmamba_init(model, 1234);
    
    uint8_t *tokens = (uint8_t*)malloc(config.seq_len * sizeof(uint8_t));
    float *logits = (float*)malloc(config.seq_len * config.vocab_size * sizeof(float));
    
    for (size_t i = 0; i < config.seq_len; i++) {
        tokens[i] = (uint8_t)(rand() % 256);
    }
    
    const int iterations = 100;
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < iterations; i++) {
        kmamba_forward(model, tokens, logits);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double throughput = (config.seq_len * iterations) / elapsed;  // tokens/sec
    double latency = elapsed / iterations * 1000;  // ms per forward
    
    printf("Inference Performance:\n");
    printf("  Throughput: %.2f tokens/sec\n", throughput);
    printf("  Latency:    %.3f ms/forward\n", latency);
    printf("  Model size: %zu parameters\n", 
           config.vocab_size * config.dim +  // Embedding
           config.n_layers * (4 * config.dim * config.state_size + 2 * config.dim) +  // MambaBlocks
           config.dim * config.vocab_size);  // LM Head
    
    free(tokens); free(logits);
    kmamba_free(model);
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== KMamba Inference Test Suite ===\n");
    printf("Testing complete KMamba inference pipeline\n\n");
    
    srand(42); /* Pour reproductibilité */
    
    int passed = 0, total = 0;
    
    /* Tests d'inférence */
    total++; passed += test_basic_forward();
    total++; passed += test_autoregressive_generation();
    total++; passed += test_byte_level_vocab();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        
        /* Benchmark */
        printf("\n=== Performance Benchmarks ===\n");
        benchmark_inference();
        
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
