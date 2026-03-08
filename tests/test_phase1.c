/*
 * test_phase1.c — Validation de GEMV et GEMM (float32)
 *
 * On vérifie que les résultats ASM correspondent
 * aux valeurs attendues calculées à la main.
 */

#include <stdio.h>
#include <math.h>

/* Déclarations des fonctions ASM */
extern void gemv(float *A, float *x, float *y, long m, long n);
extern void gemm(float *A, float *B, float *C, long m, long k, long n);

static int erreurs = 0;

static void verifier(const char *nom, float obtenu, float attendu) {
    if (fabsf(obtenu - attendu) < 1e-5f) {
        printf("  OK  %s = %.4f\n", nom, obtenu);
    } else {
        printf("  FAIL %s : obtenu %.4f, attendu %.4f\n", nom, obtenu, attendu);
        erreurs++;
    }
}

/* -------------------------------------------------------
 * Test GEMV : y = A · x
 *
 * A = | 1  2 |    x = | 1 |
 *     | 3  4 |        | 2 |
 *     | 5  6 |
 *
 * y attendu :
 *   y[0] = 1*1 + 2*2 = 5
 *   y[1] = 3*1 + 4*2 = 11
 *   y[2] = 5*1 + 6*2 = 17
 * ------------------------------------------------------- */
static void test_gemv(void) {
    printf("\n[GEMV] y = A · x\n");

    float A[6] = {1, 2, 3, 4, 5, 6};
    float x[2] = {1, 2};
    float y[3] = {0, 0, 0};

    gemv(A, x, y, 3, 2);

    verifier("y[0]", y[0],  5.0f);
    verifier("y[1]", y[1], 11.0f);
    verifier("y[2]", y[2], 17.0f);
}

/* -------------------------------------------------------
 * Test GEMM : C = A · B
 *
 * A = | 1  2 |    B = | 5  6 |
 *     | 3  4 |        | 7  8 |
 *
 * C attendu :
 *   C[0][0] = 1*5 + 2*7 = 19
 *   C[0][1] = 1*6 + 2*8 = 22
 *   C[1][0] = 3*5 + 4*7 = 43
 *   C[1][1] = 3*6 + 4*8 = 50
 * ------------------------------------------------------- */
static void test_gemm(void) {
    printf("\n[GEMM] C = A · B\n");

    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float C[4] = {0, 0, 0, 0};

    gemm(A, B, C, 2, 2, 2);

    verifier("C[0][0]", C[0], 19.0f);
    verifier("C[0][1]", C[1], 22.0f);
    verifier("C[1][0]", C[2], 43.0f);
    verifier("C[1][1]", C[3], 50.0f);
}

int main(void) {
    printf("=== optimatrix — Phase 1 : Scalaire ===");
    test_gemv();
    test_gemm();
    printf("\n%s\n", erreurs == 0 ? "Toutes les Volontés ont convergé." : "Des Volontés ont divergé.");
    return erreurs;
}
