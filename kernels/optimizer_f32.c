/*
 * optimizer_f32.c — Optimizer utilities in pure C (zero dependency)
 *
 * Newton-Schulz orthogonalization, gradient clipping, AdamW
 */

#include "kmamba_kernels.h"
#include <stdlib.h>
#include <string.h>

/* L2 norm */
float gradient_norm_f32(const float *grad, size_t n) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum_sq += grad[i] * grad[i];
    }
    return sqrtf(sum_sq);
}

/* Clip gradient in-place */
void gradient_clip_inplace_f32(float *grad, size_t n, float max_norm) {
    float norm = gradient_norm_f32(grad, n);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (size_t i = 0; i < n; i++) {
            grad[i] *= scale;
        }
    }
}

/* Newton-Schulz: G <- 1.5*G - 0.5*G*G^T*G (orthogonalizes G) */
static void newton_schulz_step_f32(float *G, size_t rows, size_t cols) {
    /* Allocate temp for G @ G^T @ G */
    float *temp = (float *)malloc(rows * cols * sizeof(float));
    if (!temp) return;
    
    /* Compute G @ G^T */
    float *GGt = (float *)malloc(rows * rows * sizeof(float));
    if (!GGt) {
        free(temp);
        return;
    }
    
    /* GGt[i,j] = sum_k G[i,k] * G[j,k] */
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < rows; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols; k++) {
                sum += G[i * cols + k] * G[j * cols + k];
            }
            GGt[i * rows + j] = sum;
        }
    }
    
    /* temp = GGt @ G */
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < rows; k++) {
                sum += GGt[i * rows + k] * G[k * cols + j];
            }
            temp[i * cols + j] = sum;
        }
    }
    
    /* G <- 1.5*G - 0.5*temp */
    for (size_t i = 0; i < rows * cols; i++) {
        G[i] = 1.5f * G[i] - 0.5f * temp[i];
    }
    
    free(temp);
    free(GGt);
}

void newton_schulz5_inplace_f32(float *G, size_t rows, size_t cols, int steps) {
    for (int s = 0; s < steps; s++) {
        newton_schulz_step_f32(G, rows, cols);
    }
}

/* AdamW step */
void adamw_step_f32(float *param, const float *grad,
                    float *m, float *v,
                    float lr, float beta1, float beta2,
                    float eps, float wd,
                    size_t n, int t) {
    /* Bias correction */
    float bias_correction1 = 1.0f - powf(beta1, t);
    float bias_correction2 = 1.0f - powf(beta2, t);
    
    for (size_t i = 0; i < n; i++) {
        /* Update moments */
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
        
        /* Bias-corrected moments */
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;
        
        /* AdamW: decoupled weight decay */
        param[i] = param[i] * (1.0f - lr * wd) - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

/* ============================================================
 * MUON Optimizer
 * ============================================================ */

/* MUON update for vector (no orthogonalization, just momentum + clip) */
void muon_update_vec_f32(float *param, const float *grad, float *m,
                         size_t n, const MBOptimConfig *conf, int t) {
    (void)t;
    
    /* Update momentum: m = mu * m + grad */
    for (size_t i = 0; i < n; i++) {
        m[i] = conf->mu * m[i] + grad[i];
    }
    
    /* Compute norm for clipping */
    float norm = gradient_norm_f32(m, n);
    float scale = (norm > conf->clip_norm) ? (conf->clip_norm / norm) : 1.0f;
    
    /* Update param with clipped momentum */
    for (size_t i = 0; i < n; i++) {
        param[i] -= conf->lr * m[i] * scale;
    }
}

/* MUON update for matrix (Newton-Schulz orthogonalization) */
void muon_update_mat_f32(float *param, const float *grad, float *m,
                         size_t rows, size_t cols, const MBOptimConfig *conf, int t) {
    (void)t;
    size_t n = rows * cols;
    
    /* Update momentum: m = mu * m + grad */
    for (size_t i = 0; i < n; i++) {
        m[i] = conf->mu * m[i] + grad[i];
    }
    
    /* Copy momentum to temp buffer for Newton-Schulz */
    float *G = (float *)malloc(n * sizeof(float));
    if (!G) return;
    
    for (size_t i = 0; i < n; i++) {
        G[i] = m[i];
    }
    
    /* Newton-Schulz orthogonalization (5 steps) */
    newton_schulz5_inplace_f32(G, rows, cols, 5);
    
    /* Compute norm for clipping */
    float norm = gradient_norm_f32(G, n);
    float scale = (norm > conf->clip_norm) ? (conf->clip_norm / norm) : 1.0f;
    
    /* Update param with orthogonalized and clipped update */
    for (size_t i = 0; i < n; i++) {
        param[i] -= conf->lr * G[i] * scale;
    }
    
    free(G);
}
