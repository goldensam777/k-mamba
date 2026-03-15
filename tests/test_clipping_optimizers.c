#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../optimatrix/include/optimatrix.h"

/* Test du gradient clipping seul */
int test_gradient_clipping_only() {
    printf("=== Test Gradient Clipping (optimatrix) ===\n");
    
    // Test 1: Pas de clipping nécessaire
    float grad1[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    size_t n1 = 5;
    float max_norm1 = 2.0f;
    
    float norm1_before = gradient_norm(grad1, n1);
    printf("Test 1 - Norme avant clipping: %.6f (max: %.1f) ", norm1_before, max_norm1);
    
    gradient_clip_inplace(grad1, n1, max_norm1);
    float norm1_after = gradient_norm(grad1, n1);
    
    if (fabsf(norm1_before - norm1_after) < 1e-6f) {
        printf("✅ Pas de clipping nécessaire\n");
    } else {
        printf("❌ Erreur: %.6f != %.6f\n", norm1_before, norm1_after);
        return 0;
    }
    
    // Test 2: Clipping nécessaire
    float grad2[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    size_t n2 = 5;
    float max_norm2 = 5.0f;
    
    float norm2_before = gradient_norm(grad2, n2);
    printf("Test 2 - Norme avant clipping: %.6f (max: %.1f) ", norm2_before, max_norm2);
    
    gradient_clip_inplace(grad2, n2, max_norm2);
    float norm2_after = gradient_norm(grad2, n2);
    
    if (norm2_after <= max_norm2 + 1e-6f) {
        printf("✅ Clipping appliqué: %.6f\n", norm2_after);
    } else {
        printf("❌ Erreur: %.6f > %.1f\n", norm2_after, max_norm2);
        return 0;
    }
    
    // Test 3: Version avec copie
    float grad3[] = {1.0f, 2.0f, 3.0f};
    float grad3_clipped[3];
    size_t n3 = 3;
    float max_norm3 = 2.0f;
    
    float norm3_before = gradient_norm(grad3, n3);
    gradient_clip(grad3, grad3_clipped, n3, max_norm3);
    float norm3_after = gradient_norm(grad3_clipped, n3);
    
    printf("Test 3 - Version copie: avant=%.3f, après=%.3f ", norm3_before, norm3_after);
    if (norm3_after <= max_norm3 + 1e-6f) {
        printf("✅\n");
    } else {
        printf("❌\n");
        return 0;
    }
    
    return 1;
}

/* Test des optimiseurs simplifiés */
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

// Simulations des fonctions d'optimiseur
static void adam_clip_update(MBOptimState *s, const MBOptimConfig *conf, size_t param_size) {
    printf("  → AdamClip: clipping=%.3f, lr=%.6f\n", conf->clip_norm, conf->lr);
    
    // Appliquer le clipping sur les gradients
    if (conf->clip_norm > 0.0f) {
        gradient_clip_inplace(s->g_W_in, param_size, conf->clip_norm);
    }
    
    // Simulation de l'update Adam
    for (size_t i = 0; i < param_size; i++) {
        float g = s->g_W_in[i] + conf->weight_decay * 0.0f;  // param = 0
        s->m_W_in[i] = conf->mu * s->m_W_in[i] + (1.0f - conf->mu) * g;
        s->v_W_in[i] = conf->beta2 * s->v_W_in[i] + (1.0f - conf->beta2) * g * g;
        float m_hat = s->m_W_in[i] / (1.0f - powf(conf->mu, (float)s->step));
        float v_hat = s->v_W_in[i] / (1.0f - powf(conf->beta2, (float)s->step));
        // param -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
    s->step++;
}

static void muon_update(MBOptimState *s, const MBOptimConfig *conf, size_t param_size) {
    printf("  → MUON: clipping=%.3f, lr=%.6f\n", conf->clip_norm, conf->lr);
    
    // 1. Momentum
    for (size_t i = 0; i < param_size; i++) {
        float g = s->g_W_in[i] + conf->weight_decay * 0.0f;
        s->m_W_in[i] = conf->mu * s->m_W_in[i] + (1.0f - conf->mu) * g;
    }
    
    // 2. Clipping sur momentum
    if (conf->clip_norm > 0.0f) {
        gradient_clip_inplace(s->m_W_in, param_size, conf->clip_norm);
    }
    
    // 3. Update
    for (size_t i = 0; i < param_size; i++) {
        // param -= lr * m_W_in[i];
    }
    s->step++;
}

int test_optimizers_with_clipping() {
    printf("\n=== Test Optimiseurs avec Clipping ===\n");
    
    MBOptimConfig opt_config = {
        .lr = 1e-3f,
        .mu = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .clip_norm = 0.5f,  // Clipping agressif
        .weight_decay = 1e-5f
    };
    
    size_t param_size = 100;
    
    // Test ADAM_CLIP
    printf("Test ADAM_CLIP avec gradients grands:\n");
    MBOptimState state_adam = {
        .type = OPTIMIZER_ADAM_CLIP,
        .step = 0,
        .g_W_in = malloc(param_size * sizeof(float)),
        .m_W_in = calloc(param_size, sizeof(float)),
        .v_W_in = calloc(param_size, sizeof(float))
    };
    
    // Gradients très grands
    for (size_t i = 0; i < param_size; i++) {
        state_adam.g_W_in[i] = 10.0f;
    }
    
    float norm_before = gradient_norm(state_adam.g_W_in, param_size);
    printf("  Norme gradients avant: %.3f\n", norm_before);
    
    adam_clip_update(&state_adam, &opt_config, param_size);
    
    float norm_after = gradient_norm(state_adam.g_W_in, param_size);
    printf("  Norme gradients après: %.3f\n", norm_after);
    
    if (norm_after <= opt_config.clip_norm + 1e-6f) {
        printf("  ✅ Clipping fonctionnel\n");
    } else {
        printf("  ❌ Clipping échoué\n");
        free(state_adam.g_W_in);
        free(state_adam.m_W_in);
        free(state_adam.v_W_in);
        return 0;
    }
    
    // Test MUON
    printf("\nTest MUON avec gradients grands:\n");
    MBOptimState state_muon = {
        .type = OPTIMIZER_MUON,
        .step = 0,
        .g_W_in = malloc(param_size * sizeof(float)),
        .m_W_in = calloc(param_size, sizeof(float)),
        .v_W_in = NULL  // MUON n'utilise pas v
    };
    
    for (size_t i = 0; i < param_size; i++) {
        state_muon.g_W_in[i] = 10.0f;
    }
    
    muon_update(&state_muon, &opt_config, param_size);
    
    float momentum_norm = gradient_norm(state_muon.m_W_in, param_size);
    printf("  Norme momentum après clipping: %.3f\n", momentum_norm);
    
    if (momentum_norm <= opt_config.clip_norm + 1e-6f) {
        printf("  ✅ Clipping MUON fonctionnel\n");
    } else {
        printf("  ❌ Clipping MUON échoué\n");
        free(state_adam.g_W_in);
        free(state_adam.m_W_in);
        free(state_adam.v_W_in);
        free(state_muon.g_W_in);
        free(state_muon.m_W_in);
        return 0;
    }
    
    // Cleanup
    free(state_adam.g_W_in);
    free(state_adam.m_W_in);
    free(state_adam.v_W_in);
    free(state_muon.g_W_in);
    free(state_muon.m_W_in);
    
    return 1;
}

int main() {
    printf("🧪 Tests Gradient Clipping + Optimiseurs\n");
    printf("========================================\n");
    
    int success = 1;
    
    // Test 1: Gradient clipping
    if (!test_gradient_clipping_only()) {
        success = 0;
    }
    
    // Test 2: Optimiseurs avec clipping
    if (!test_optimizers_with_clipping()) {
        success = 0;
    }
    
    printf("\n=== Résultat Final ===\n");
    if (success) {
        printf("🎉 Tous les tests passés !\n");
        printf("✅ Gradient clipping optimatrix fonctionnel\n");
        printf("✅ Clipping réutilisable par optimiseurs\n");
        printf("✅ ADAM_CLIP avec clipping intégré\n");
        printf("✅ MUON avec clipping sur momentum\n");
    } else {
        printf("❌ Certains tests ont échoué\n");
    }
    
    return success ? 0 : 1;
}
