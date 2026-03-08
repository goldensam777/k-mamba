/*
 * test_phase4.c — Validation Hadamard + Activations (float32)
 */

#include <stdio.h>
#include <math.h>
#include <string.h>

extern void hadamard      (float *x, float *y, float *z, long n);
extern void hadamard_avx2 (float *x, float *y, float *z, long n);
extern void relu_f32      (float *x, float *y, long n);
extern void sigmoid_f32   (float *x, float *y, long n);
extern void silu_f32      (float *x, float *y, long n);
extern void softplus_f32  (float *x, float *y, long n);

static int erreurs = 0;

static void verifier(const char *nom, float obtenu, float attendu) {
    if (fabsf(obtenu - attendu) < 1e-5f) {
        printf("  OK  %-20s = %10.7f\n", nom, obtenu);
    } else {
        printf("  FAIL %-20s : obtenu=%.8f  attendu=%.8f\n",
               nom, obtenu, attendu);
        erreurs++;
    }
}

/* ---- Références C ---- */
static float ref_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float ref_silu(float x)    { return x * ref_sigmoid(x); }
static float ref_softplus(float x){ return logf(1.0f + expf(x)); }
static float ref_relu(float x)    { return x > 0.0f ? x : 0.0f; }

/* ============================================================
 * Test Hadamard
 * ============================================================ */
static void test_hadamard(void) {
    printf("\n[Hadamard] z = x ⊙ y\n");

    float x[7] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float y[7] = {2.0f, 0.5f, 3.0f, 0.25f, 4.0f, 1.0f, 0.5f};
    float z_scal[7] = {0};
    float z_avx2[7] = {0};

    hadamard      (x, y, z_scal, 7);
    hadamard_avx2 (x, y, z_avx2, 7);

    char nom[32];
    for (int i = 0; i < 7; i++) {
        float attendu = x[i] * y[i];
        snprintf(nom, sizeof(nom), "scal z[%d]", i);
        verifier(nom, z_scal[i], attendu);
        snprintf(nom, sizeof(nom), "avx2 z[%d]", i);
        verifier(nom, z_avx2[i], attendu);
    }
}

/* ============================================================
 * Test Activations — valeurs connues + référence C
 * ============================================================ */
static void test_activations(void) {
    printf("\n[Activations] valeurs remarquables\n");

    float x[6]   = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
    float out[6] = {0};
    char  nom[32];

    /* ReLU */
    relu_f32(x, out, 6);
    for (int i = 0; i < 6; i++) {
        snprintf(nom, sizeof(nom), "relu(%.1f)", x[i]);
        verifier(nom, out[i], ref_relu(x[i]));
    }

    /* Sigmoid */
    printf("\n");
    sigmoid_f32(x, out, 6);
    for (int i = 0; i < 6; i++) {
        snprintf(nom, sizeof(nom), "sigmoid(%.1f)", x[i]);
        verifier(nom, out[i], ref_sigmoid(x[i]));
    }

    /* SiLU */
    printf("\n");
    silu_f32(x, out, 6);
    for (int i = 0; i < 6; i++) {
        snprintf(nom, sizeof(nom), "silu(%.1f)", x[i]);
        verifier(nom, out[i], ref_silu(x[i]));
    }

    /* Softplus */
    printf("\n");
    softplus_f32(x, out, 6);
    for (int i = 0; i < 6; i++) {
        snprintf(nom, sizeof(nom), "softplus(%.1f)", x[i]);
        verifier(nom, out[i], ref_softplus(x[i]));
    }
}

int main(void) {
    printf("=== optimatrix — Phase 4 : Hadamard + Activations ===");
    test_hadamard();
    test_activations();
    printf("\n%s\n", erreurs == 0
        ? "Toutes les Volontés ont convergé."
        : "Des Volontés ont divergé.");
    return erreurs;
}
