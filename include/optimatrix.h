#ifndef OPTIMATRIX_H
#define OPTIMATRIX_H

void gemv(float *A, float *x, float *y, long m, long n);
void gemv_avx2(float *A, float *x, float *y, long m, long n);
void gemm(float *A, float *B, float *C, long m, long k, long n);
void gemm_avx2(float *A, float *B, float *C, long m, long k, long n);
void hadamard(float *x, float *y, float *z, long n);
void hadamard_avx2(float *x, float *y, float *z, long n);
void relu_f32(float *x, float *y, long n);
void sigmoid_f32(float *x, float *y, long n);
void silu_f32(float *x, float *y, long n);
void softplus_f32(float *x, float *y, long n);

typedef struct {
    float *x;
    float *A;
    float *B;
    float *C;
    float *delta;
    float *h;
    float *y;
    long   L;
    long   D;
    long   M;
} ScanParams;

void scan1d(ScanParams *p);

typedef struct {
    float *x;
    float *A;
    float *B;
    float *C;
    float *delta;
    float *h0;
    float *h;
    float *dy;
    float *dx;
    float *dA;
    float *dB;
    float *dC;
    float *ddelta;
    long   L;
    long   D;
    long   M;
} ScanBackwardParams;

void scan1d_backward(ScanBackwardParams *p);

typedef struct {
    float *x;
    float *A;
    float *A_diag;
    float *B;
    float *C;
    float *delta;
    float *h0;
    float *h;
    float *dy;
    float *dx;
    float *dA;
    float *dB;
    float *dC;
    float *ddelta;
    long   L;
    long   D;
} ScanBackwardSharedParams;

void scan1d_backward_m1_shared_bc(ScanBackwardSharedParams *p);
void scan1d_backward_m1_shared_bc_asm(ScanBackwardSharedParams *p);
void scan1d_backward_m1_shared_bc_simple_asm(ScanBackwardSharedParams *p);

typedef struct {
    float *x;
    float *A;
    float *B;
    float *C;
    float *delta;
    float *h0;
    float *h;
    float *dy;
    float *dx;
    float *dA;
    float *dB;
    float *dC;
    float *ddelta;
    long   L;
    long   D;
    long   M;
} ScanBackwardMParams;

void scan1d_backward_m_generic(ScanBackwardMParams *p);

typedef struct {
    float *input;     // Input tensor [N1, N2, ..., ND, D]
    float *A;         // State matrix [M, M, D]
    float *B;         // Input matrix [M, M, D]
    float *C;         // Output matrix [M, M, D]
    float *delta;     // Delta values [N1, N2, ..., ND, M, M, D]
    float *h0;        // Initial state [M, D]
    float *output;    // Output tensor [N1, N2, ..., ND, D]
    long  *dims;      // Dimensions [N1, N2, ..., ND]
    long   ndims;     // Number of dimensions
    long   D;         // Feature dimension
    long   M;         // State dimension
} ConvNDParams;

void convnd_forward(ConvNDParams *p);
void convnd_backward(ConvNDParams *p);

// ===== STRUCTURE D'ENTRAÎNEMENT INTÉGRÉ =====

typedef struct {
    // Paramètres du modèle
    long D;              // Dimension des features
    long M;              // Dimension de l'état caché
    
    // Poids du modèle (alloués dynamiquement)
    float *A;            // Matrice d'état [M×M×D]
    float *B;            // Matrice d'entrée [M×M×D]
    float *C;            // Matrice de sortie [M×M×D]
    
    // États et buffers
    float *h;            // État caché courant [M×D]
    float *adj_h;         // État adjoint (backward) [M×D]
    
    // Hyperparamètres
    float learning_rate;  // Taux d'apprentissage
    int   epochs;        // Nombre d'époques
    int   batch_size;    // Taille du batch
    
    // Métriques d'entraînement
    float loss;          // Loss actuelle
    float accuracy;      // Précision (classification)
    float precision;     // Précision (régression)
    float recall;        // Rappel (classification)
    float f1_score;      // Score F1 (classification)
    
    // Métriques de performance
    double forward_time_ms;    // Temps forward moyen
    double backward_time_ms;   // Temps backward moyen
    double total_time_ms;     // Temps total par epoch
    long total_samples;        // Nombre total d'échantillons traités
    
    // État du modèle
    int is_trained;           // 0 = non entraîné, 1 = entraîné
    float best_loss;           // Meilleure loss atteinte
    int best_epoch;            // Meilleure epoch
} MambaModel;

// ===== API D'ENTRAÎNEMENT SIMPLIFIÉE =====

// Création et destruction
MambaModel* mamba_create(long D, long M);
void        mamba_destroy(MambaModel *model);

// Entraînement
void mamba_train(MambaModel *model, 
                float *X_train, float *y_train, long batch_size, long seq_len,
                int epochs, float learning_rate);

// Prédiction
float* mamba_forward(MambaModel *model, float *input, long seq_len);

// Évaluation
void mamba_evaluate(MambaModel *model, float *X_test, float *y_test, 
                   long test_size, long seq_len);

// Métriques
void mamba_update_metrics(MambaModel *model, float *predictions, float *targets, long size);

// Utilitaires
void   mamba_save(MambaModel *model, const char *filename);
MambaModel* mamba_load(const char *filename);
void   mamba_print_info(MambaModel *model);
void   mamba_print_metrics(MambaModel *model);

// ===== FORWARD PASS COMPLET =====

// Paramètres configurables pour le forward pass
typedef struct {
    float alpha;         // Taux d'apprentissage pour mise à jour (ex: 0.1f)
    float beta;          // Facteur de lissage (ex: 0.9f)
    float min_val;       // Limite inférieure de clamp (ex: -10.0f)
    float max_val;       // Limite supérieure de clamp (ex: 10.0f)
    int use_exp;         // Utiliser exponentielle? 0=non, 1=oui
    float exp_scale;     // Échelle pour exponentielle (ex: 1.0f)
} ForwardPassConfig;

// Structure unifiée pour Forward + Backward
typedef struct {
    // Paramètres partagés
    ForwardPassConfig config;
    long L, D, M;
    
    // Forward data
    float *x;            // Input sequence [L×D]
    float *input;        // Input buffer (alias pour x) [L×D]
    float *A;            // State transition matrix [M×M×D]
    float *B;            // Input matrix [M×M×D]  
    float *C;            // Output matrix [M×M×D]
    float *Δ;            // Delta values [L×M×D]
    float *h_prev;        // Previous hidden state [M×D]
    float *h_curr;        // Current hidden state [M×D]
    float *output;         // Output [L×D]
    float *h_states;      // États sauvegardés [L×M×D]
    
    // Backward data  
    float *dy;           // Gradient de sortie [L×D]
    float *dx;           // Gradient d'entrée [L×D]
    float *dA;           // Gradient de A [M×M×D]
    float *dB;           // Gradient de B [M×M×D]
    float *dC;           // Gradient de C [M×M×D]
    float *ddelta;       // Gradient de Δ [L×M×D]
} MambaPassParams;

typedef struct {
    float *x;            // Input sequence [L×D]
    float *input;        // Input buffer (alias pour x) [L×D]
    float *A;            // State transition matrix [M×M×D]
    float *B;            // Input matrix [M×M×D]  
    float *C;            // Output matrix [M×M×D]
    float *Δ;            // Delta values [L×M×D]
    float *h_prev;        // Previous hidden state [M×D]
    float *h_curr;        // Current hidden state [M×D]
    float *output;         // Output [L×D]
    long L, D, M;        // Dimensions
    ForwardPassConfig config;  // Configuration paramétrique
} SelectiveScanParams;

// Fonctions unifiées Forward + Backward
float* mamba_forward_unified(MambaPassParams *p);
void   mamba_backward_unified(MambaPassParams *p);

// Fonctions d'allocation/libération pour MambaPassParams
MambaPassParams* mamba_params_create(long L, long D, long M);
void            mamba_params_free(MambaPassParams *p);

// Forward pass avec sélectivité complète (legacy)
float* selective_scan_forward(SelectiveScanParams *p);
float* selective_scan_forward_parametric(SelectiveScanParams *p, float *h_states);

// Utilitaires de configuration pour forward pass
ForwardPassConfig create_default_config();
ForwardPassConfig create_aggressive_config();
ForwardPassConfig create_custom_config(float alpha, float beta, 
                                   float min_val, float max_val, 
                                   int use_exp, float exp_scale);

// Approximation rapide de exp()
static inline float fast_exp(float x);

// Mécanisme de gate pour sélectivité
static inline float compute_gate(float *x, float *Δ, long d);

// Forward pass Mamba complet
float* mamba_forward_complete(MambaModel *model, float *input, long seq_len);

typedef struct {
    float *x;
    float *A1;
    float *A2;
    float *B;
    float *C;
    float *delta1;
    float *delta2;
    float *h;
    float *y;
    long   d1;
    long   d2;
    long   D;
    long   M;
} Scan2DParams;

void scan2d(Scan2DParams *p);

#endif
