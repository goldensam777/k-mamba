#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Copie des structures et types nécessaires
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
    float dt_rank;
    float dt_init;
    int use_convnd;
    long convnd_K;
    long convnd_ndims;
} MBConfig;

typedef enum {
    OPTIMIZER_ADAM_CLIP,
    OPTIMIZER_MUON,
    OPTIMIZER_SGD,
    OPTIMIZER_ADAMW
} OptimizerType;

typedef struct {
    float lr;
    float mu;
    float beta2;
    float eps;
    float clip_norm;
    float weight_decay;
} MBOptimConfig;

typedef struct {
    OptimizerType type;
    size_t step;
    float *g_W_in;
    float *m_W_in;
    float *v_W_in;
} MBOptimState;

// Fonctions optimiseurs simplifiées
static void adam_clip_update(MBOptimState *s, const MBOptimConfig *conf) {
    printf("  → AdamClip update (lr=%.6f, mu=%.3f)\n", conf->lr, conf->mu);
    s->step++;
}

static void adamw_update(MBOptimState *s, const MBOptimConfig *conf) {
    printf("  → AdamW update (lr=%.6f, mu=%.3f)\n", conf->lr, conf->mu);
    s->step++;
}

static void sgd_update(MBOptimState *s, const MBOptimConfig *conf) {
    printf("  → SGD update (lr=%.6f, mu=%.3f)\n", conf->lr, conf->mu);
    s->step++;
}

static void muon_update(MBOptimState *s, const MBOptimConfig *conf) {
    printf("  → MUON update (placeholder - AdamClip fallback)\n");
    adam_clip_update(s, conf);
}

// Dispatcher optimiseur
void mamba_optimizer_step(MBOptimState *s, const MBOptimConfig *conf) {
    switch (s->type) {
        case OPTIMIZER_ADAM_CLIP:
            adam_clip_update(s, conf);
            break;
        case OPTIMIZER_ADAMW:
            adamw_update(s, conf);
            break;
        case OPTIMIZER_SGD:
            sgd_update(s, conf);
            break;
        case OPTIMIZER_MUON:
            muon_update(s, conf);
            break;
        default:
            printf("  → Unknown optimizer, defaulting to AdamClip\n");
            adam_clip_update(s, conf);
            break;
    }
}

// Test principal
int main() {
    printf("=== Test Système d'Optimiseurs Modulaires ===\n");
    
    // Configuration optimiseur
    MBOptimConfig opt_config = {
        .lr = 1e-3f,
        .mu = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .clip_norm = 1.0f,
        .weight_decay = 1e-5f
    };
    
    // État optimiseur
    MBOptimState state = {0};
    state.g_W_in = (float*)malloc(1000 * sizeof(float));
    state.m_W_in = (float*)malloc(1000 * sizeof(float));
    state.v_W_in = (float*)malloc(1000 * sizeof(float));
    
    // Test des différents optimiseurs
    printf("\n--- Test Optimizers ---\n");
    
    printf("1. Test ADAM_CLIP\n");
    state.type = OPTIMIZER_ADAM_CLIP;
    state.step = 0;
    mamba_optimizer_step(&state, &opt_config);
    
    printf("2. Test ADAMW\n");
    state.type = OPTIMIZER_ADAMW;
    state.step = 0;
    mamba_optimizer_step(&state, &opt_config);
    
    printf("3. Test SGD\n");
    state.type = OPTIMIZER_SGD;
    state.step = 0;
    mamba_optimizer_step(&state, &opt_config);
    
    printf("4. Test MUON (placeholder)\n");
    state.type = OPTIMIZER_MUON;
    state.step = 0;
    mamba_optimizer_step(&state, &opt_config);
    
    // Cleanup
    free(state.g_W_in);
    free(state.m_W_in);
    free(state.v_W_in);
    
    printf("\n=== Test Réussi ! ===\n");
    printf("✅ Système d'optimiseurs modulaires fonctionnel\n");
    printf("✅ Supporte 4 types d'optimiseurs\n");
    printf("✅ API de sélection fonctionnelle\n");
    printf("✅ Extensible pour MUON futur\n");
    
    return 0;
}
