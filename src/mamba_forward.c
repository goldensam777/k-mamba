#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "optimatrix.h"

#define isnan(x) ((x) != (x))
#define isinf(x) ((x) == INFINITY || (x) == -INFINITY)

// ===== UTILITAIRES MATHÉMATIQUES =====

static inline float fast_exp(float x) {
    // Approximation polynomiale d'ordre 3 de exp(x)
    // Clamp pour éviter l'explosion numérique
    if (x < -5.0f) return 0.0f;
    if (x > 5.0f) return 148.413f;  // exp(5)
    
    return 1.0f + x + 0.5f * x * x + x * x * x / 6.0f;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

static inline float fast_tanh(float x) {
    // Approximation rapide de tanh
    float exp2x = fast_exp(2.0f * x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

static float dot_product(float *a, float *b, long size) {
    float sum = 0.0f;
    for (long i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static inline float compute_gate(float *x, float *Δ, long d) {
    // Gate basé sur le contenu de x et Δ
    float score = dot_product(x, Δ, d);
    return sigmoid(score);  // Sélectivité sigmoïde
}

// ===== SELECTIVE SCAN FORWARD =====

// ===== UTILITAIRES DE CONFIGURATION =====

// Créer une configuration par défaut (stable)
ForwardPassConfig create_default_config() {
    ForwardPassConfig config = {
        .alpha = 0.1f,        // Taux d'apprentissage modéré
        .beta = 0.9f,         // Fort lissage pour stabilité
        .min_val = -10.0f,     // Clamp large
        .max_val = 10.0f,      
        .use_exp = 0,          // Version linéaire (plus stable)
        .exp_scale = 1.0f      // Non utilisé si use_exp=0
    };
    return config;
}

// Créer une configuration agressive (plus rapide mais moins stable)
ForwardPassConfig create_aggressive_config() {
    ForwardPassConfig config = {
        .alpha = 0.3f,        // Taux d'apprentissage élevé
        .beta = 0.7f,         // Moins de lissage
        .min_val = -5.0f,      // Clamp plus serré
        .max_val = 5.0f,       
        .use_exp = 1,          // Version exponentielle
        .exp_scale = 0.5f      // Échelle modérée
    };
    return config;
}

// Créer une configuration personnalisée
ForwardPassConfig create_custom_config(float alpha, float beta, 
                                   float min_val, float max_val, 
                                   int use_exp, float exp_scale) {
    ForwardPassConfig config = {
        .alpha = alpha,
        .beta = beta,
        .min_val = min_val,
        .max_val = max_val,
        .use_exp = use_exp,
        .exp_scale = exp_scale
    };
    return config;
}

// ===== FONCTIONS UNIFIÉES FORWARD + BACKWARD =====

// Allouer et initialiser MambaPassParams
MambaPassParams* mamba_params_create(long L, long D, long M) {
    MambaPassParams *p = malloc(sizeof(MambaPassParams));
    if (!p) return NULL;
    
    // Dimensions
    p->L = L; p->D = D; p->M = M;
    
    // Configuration par défaut
    p->config = create_default_config();
    
    // Allouer les buffers
    p->x = p->input = malloc(L * D * sizeof(float));
    p->A = malloc(M * M * D * sizeof(float));
    p->B = malloc(M * M * D * sizeof(float));
    p->C = malloc(M * M * D * sizeof(float));
    p->Δ = malloc(L * M * D * sizeof(float));
    p->h_prev = malloc(M * D * sizeof(float));
    p->h_curr = malloc(M * D * sizeof(float));
    p->output = malloc(L * D * sizeof(float));
    p->h_states = malloc(L * M * D * sizeof(float));
    
    // Backward buffers
    p->dy = malloc(L * D * sizeof(float));
    p->dx = malloc(L * D * sizeof(float));
    p->dA = malloc(M * M * D * sizeof(float));
    p->dB = malloc(M * M * D * sizeof(float));
    p->dC = malloc(M * M * D * sizeof(float));
    p->ddelta = malloc(L * M * D * sizeof(float));
    
    // Vérifier les allocations
    if (!p->x || !p->A || !p->B || !p->C || !p->Δ || 
        !p->h_prev || !p->h_curr || !p->output || !p->h_states ||
        !p->dy || !p->dx || !p->dA || !p->dB || !p->dC || !p->ddelta) {
        mamba_params_free(p);
        return NULL;
    }
    
    return p;
}

// Libérer tous les buffers
void mamba_params_free(MambaPassParams *p) {
    if (!p) return;
    
    free(p->x); free(p->A); free(p->B); free(p->C); free(p->Δ);
    free(p->h_prev); free(p->h_curr); free(p->output); free(p->h_states);
    free(p->dy); free(p->dx); free(p->dA); free(p->dB); free(p->dC); free(p->ddelta);
    free(p);
}

// Forward pass unifié avec sauvegarde automatique
float* mamba_forward_unified(MambaPassParams *p) {
    if (!p || !p->A || !p->B || !p->C || !p->Δ || !p->h_prev || !p->h_curr || !p->output) {
        printf("❌ Paramètres invalides pour mamba_forward_unified\n");
        return NULL;
    }
    
    printf("🚀 Mamba Forward Unifié (L=%ld, D=%ld, M=%ld)\n", p->L, p->D, p->M);
    printf("   Config: α=%.3f, β=%.3f, [%.1f,%.1f], exp=%d\n", 
           p->config.alpha, p->config.beta, p->config.min_val, p->config.max_val, p->config.use_exp);
    
    // Initialiser l'état caché
    for (long m = 0; m < p->M; m++) {
        for (long d = 0; d < p->D; d++) {
            p->h_curr[m * p->D + d] = p->h_prev[m * p->D + d];
        }
    }
    
    // Initialiser l'output à zéro
    for (long i = 0; i < p->L * p->D; i++) {
        p->output[i] = 0.0f;
    }
    
    // Boucle principale sur la séquence
    for (long t = 0; t < p->L; t++) {
        long x_offset = t * p->D;
        long h_offset = t * p->M * p->D;  // Offset pour h_states[t]
        
        // Pour chaque dimension de l'état
        for (long m = 0; m < p->M; m++) {
            for (long d = 0; d < p->D; d++) {
                // Extraire les poids pour cette dimension
                long weight_idx = m * p->M * p->D + m * p->D + d;
                float A_md = p->A[weight_idx];
                float B_md = p->B[weight_idx];
                float C_md = p->C[weight_idx];
                
                // Récupérer les valeurs actuelles
                float h_prev_val = p->h_curr[m * p->D + d];
                float x_val = p->input[x_offset + d];
                float delta_val = p->Δ[t * p->M * p->D + m * p->D + d];
                
                // Calcul de la mise à jour selon configuration
                float A_x = A_md * x_val;
                float input_term = A_x + B_md;
                
                float h_new;
                if (p->config.use_exp) {
                    // Version avec exponentielle
                    float exp_term = fast_exp(delta_val * p->config.exp_scale);
                    h_new = p->config.beta * h_prev_val + p->config.alpha * exp_term * input_term;
                } else {
                    // Version linéaire simple
                    h_new = p->config.beta * h_prev_val + p->config.alpha * input_term;
                }
                
                // Clamp selon configuration
                if (h_new > p->config.max_val) h_new = p->config.max_val;
                if (h_new < p->config.min_val) h_new = p->config.min_val;
                
                p->h_curr[m * p->D + d] = h_new;
                
                // SAUVEGARDER AUTOMATIQUEMENT l'état pour le backward pass
                p->h_states[h_offset + m * p->D + d] = h_new;
                
                // Calculer la sortie
                p->output[x_offset + d] += h_new * C_md;
            }
        }
    }
    
    return p->output;
}

// Backward pass unifié utilisant les états sauvegardés
void mamba_backward_unified(MambaPassParams *p) {
    if (!p || !p->h_states) {
        printf("❌ Paramètres invalides pour mamba_backward_unified\n");
        return;
    }
    
    printf("🔄 Mamba Backward Unifié (L=%ld, D=%ld, M=%ld)\n", p->L, p->D, p->M);
    
    // Initialiser les gradients à zéro
    for (long i = 0; i < p->L * p->D; i++) {
        p->dx[i] = 0.0f;
        p->dy[i] = 0.0f;  // dy est déjà calculé, mais on s'assure qu'il est propre
    }
    
    for (long i = 0; i < p->M * p->M * p->D; i++) {
        p->dA[i] = 0.0f;
        p->dB[i] = 0.0f;
        p->dC[i] = 0.0f;
    }
    
    for (long i = 0; i < p->L * p->M * p->D; i++) {
        p->ddelta[i] = 0.0f;
    }
    
    // Allouer l'état adjoint
    float *adj_h = malloc(p->M * p->D * sizeof(float));
    if (!adj_h) return;
    
    memset(adj_h, 0, p->M * p->D * sizeof(float));
    
    // Backward pass : parcourir la séquence à l'envers
    for (long t = p->L - 1; t >= 0; t--) {
        long x_offset = t * p->D;
        long h_offset = t * p->M * p->D;
        
        // Pour chaque dimension de l'état
        for (long m = 0; m < p->M; m++) {
            for (long d = 0; d < p->D; d++) {
                // Extraire les poids et valeurs
                long weight_idx = m * p->M * p->D + m * p->D + d;
                float A_md = p->A[weight_idx];
                float B_md = p->B[weight_idx];
                float C_md = p->C[weight_idx];
                
                float x_val = p->input[x_offset + d];
                float delta_val = p->Δ[t * p->M * p->D + m * p->D + d];
                float h_t = p->h_states[h_offset + m * p->D + d];  // État sauvegardé
                
                // Calculer l'adjoint pour cette position
                float ah = adj_h[m * p->D + d] + p->dy[x_offset + d] * C_md;
                
                // Gradients pour C
                p->dC[weight_idx] += p->dy[x_offset + d] * h_t;
                
                // Gradients pour B
                float dB_val = ah * p->config.alpha;
                if (p->config.use_exp) {
                    dB_val *= fast_exp(delta_val * p->config.exp_scale);
                }
                p->dB[weight_idx] += dB_val;
                
                // Gradients pour A
                float dA_val = ah * p->config.alpha * x_val;
                if (p->config.use_exp) {
                    dA_val *= fast_exp(delta_val * p->config.exp_scale);
                }
                p->dA[weight_idx] += dA_val;
                
                // Gradients pour x
                float dx_val = ah * p->config.alpha * A_md;
                if (p->config.use_exp) {
                    dx_val *= fast_exp(delta_val * p->config.exp_scale);
                }
                p->dx[x_offset + d] += dx_val;
                
                // Mettre à jour l'adjoint
                adj_h[m * p->D + d] = ah * p->config.beta;
            }
        }
    }
    
    free(adj_h);
}

// ===== FORWARD PASS PARAMÉTRIQUE =====

float* selective_scan_forward_parametric(SelectiveScanParams *p, float *h_states) {
    if (!p || !p->A || !p->B || !p->C || !p->Δ || !p->h_prev || !p->h_curr || !p->output) {
        printf("❌ Paramètres invalides pour selective_scan_forward_parametric\n");
        return NULL;
    }
    
    printf("🚀 Selective Scan Forward Paramétrique (L=%ld, D=%ld, M=%ld)\n", p->L, p->D, p->M);
    printf("   Config: α=%.3f, β=%.3f, [%.1f,%.1f], exp=%d\n", 
           p->config.alpha, p->config.beta, p->config.min_val, p->config.max_val, p->config.use_exp);
    
    // Initialiser l'état caché
    for (long m = 0; m < p->M; m++) {
        for (long d = 0; d < p->D; d++) {
            p->h_curr[m * p->D + d] = p->h_prev[m * p->D + d];
        }
    }
    
    // Initialiser l'output à zéro
    for (long i = 0; i < p->L * p->D; i++) {
        p->output[i] = 0.0f;
    }
    
    // Boucle principale sur la séquence
    for (long t = 0; t < p->L; t++) {
        long x_offset = t * p->D;
        long h_offset = t * p->M * p->D;  // Offset pour h_states[t]
        
        // Pour chaque dimension de l'état
        for (long m = 0; m < p->M; m++) {
            for (long d = 0; d < p->D; d++) {
                // Extraire les poids pour cette dimension
                long weight_idx = m * p->M * p->D + m * p->D + d;
                float A_md = p->A[weight_idx];
                float B_md = p->B[weight_idx];
                float C_md = p->C[weight_idx];
                
                // Récupérer les valeurs actuelles
                float h_prev_val = p->h_curr[m * p->D + d];
                float x_val = p->input[x_offset + d];
                float delta_val = p->Δ[t * p->M * p->D + m * p->D + d];
                
                // Calcul de la mise à jour selon configuration
                float A_x = A_md * x_val;
                float input_term = A_x + B_md;
                
                float h_new;
                if (p->config.use_exp) {
                    // Version avec exponentielle
                    float exp_term = fast_exp(delta_val * p->config.exp_scale);
                    h_new = p->config.beta * h_prev_val + p->config.alpha * exp_term * input_term;
                } else {
                    // Version linéaire simple
                    h_new = p->config.beta * h_prev_val + p->config.alpha * input_term;
                }
                
                // Clamp selon configuration
                if (h_new > p->config.max_val) h_new = p->config.max_val;
                if (h_new < p->config.min_val) h_new = p->config.min_val;
                
                p->h_curr[m * p->D + d] = h_new;
                
                // Sauvegarder l'état pour le backward pass
                if (h_states) {
                    h_states[h_offset + m * p->D + d] = h_new;
                }
                
                // Calculer la sortie
                p->output[x_offset + d] += h_new * C_md;
            }
        }
    }
    
    return p->output;
}

// ===== FORWARD PASS MAMBA COMPLET =====

float* mamba_forward_complete(MambaModel *model, float *input, long seq_len) {
    if (!model || !input || seq_len <= 0) {
        printf("❌ Paramètres invalides pour mamba_forward_complete\n");
        return NULL;
    }
    
    printf("🚀 Mamba Forward Complet Paramétrique (D=%ld, M=%ld, seq_len=%ld)\n", 
           model->D, model->M, (long)seq_len);
    
    // Allouer les buffers temporaires
    float *output = malloc((long)seq_len * model->D * sizeof(float));
    float *h_buffer = malloc(model->M * model->D * sizeof(float));
    float *h_states = malloc((long)seq_len * model->M * model->D * sizeof(float));
    float *Δ_buffer = malloc((long)seq_len * model->M * model->D * sizeof(float));
    
    if (!output || !h_buffer || !h_states || !Δ_buffer) {
        printf("❌ Erreur d'allocation mémoire\n");
        free(output); free(h_buffer); free(h_states); free(Δ_buffer);
        return NULL;
    }
    
    // Initialiser Δ avec des valeurs plus significatives
    srand(42);
    for (long i = 0; i < seq_len * model->M * model->D; i++) {
        Δ_buffer[i] = 0.5f + 0.5f * (float)rand() / RAND_MAX;
    }
    
    // Créer la configuration par défaut (stable)
    ForwardPassConfig config = create_default_config();
    
    // Préparer les paramètres pour le selective scan
    SelectiveScanParams scan_params = {
        .x = input,
        .input = input,
        .A = model->A,
        .B = model->B,
        .C = model->C,
        .Δ = Δ_buffer,
        .h_prev = model->h,
        .h_curr = h_buffer,
        .output = output,
        .L = seq_len,
        .D = model->D,
        .M = model->M,
        .config = config  // UTILISER LA CONFIGURATION PARAMÉTRIQUE
    };
    
    // Exécuter le forward pass paramétrique
    float *result = selective_scan_forward_parametric(&scan_params, h_states);
    
    // Mettre à jour l'état final dans le modèle
    if (result && seq_len > 0) {
        for (long m = 0; m < model->M; m++) {
            for (long d = 0; d < model->D; d++) {
                model->h[m * model->D + d] = h_buffer[m * model->D + d];
            }
        }
    }
    
    // Nettoyer
    free(h_buffer);
    free(Δ_buffer);
    free(h_states);
    
    return result;
}
