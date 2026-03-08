/*
 * test_phase2.c — Validation GEMV/GEMM AVX2 (float32)
 *
 * Même calculs que Phase 1, mais via les versions vectorisées.
 * On compare aussi les résultats scalaire vs AVX2 pour vérifier
 * la cohérence.
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

extern void gemv      (float *A, float *x, float *y, long m, long n);
extern void gemv_avx2 (float *A, float *x, float *y, long m, long n);
extern void gemm      (float *A, float *B, float *C, long m, long k, long n);
extern void gemm_avx2 (float *A, float *B, float *C, long m, long k, long n);

static int erreurs = 0;

static void verifier(const char *nom, float obtenu, float attendu) {
    if (fabsf(obtenu - attendu) < 1e-4f) {
        printf("  OK  %s = %.4f\n", nom, obtenu);
    } else {
        printf("  FAIL %s : obtenu %.6f, attendu %.6f\n", nom, obtenu, attendu);
        erreurs++;
    }
}

/* -------------------------------------------------------
 * Test GEMV AVX2 — même cas que phase 1
 * ------------------------------------------------------- */
static void test_gemv_avx2(void) {
    printf("\n[GEMV AVX2] y = A · x\n");

    float A[6] = {1, 2, 3, 4, 5, 6};
    float x[2] = {1, 2};
    float y[3] = {0, 0, 0};

    gemv_avx2(A, x, y, 3, 2);

    verifier("y[0]", y[0],  5.0f);
    verifier("y[1]", y[1], 11.0f);
    verifier("y[2]", y[2], 17.0f);
}

/* -------------------------------------------------------
 * Test GEMM AVX2 — même cas que phase 1
 * ------------------------------------------------------- */
static void test_gemm_avx2(void) {
    printf("\n[GEMM AVX2] C = A · B\n");

    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float C[4] = {0, 0, 0, 0};

    gemm_avx2(A, B, C, 2, 2, 2);

    verifier("C[0][0]", C[0], 19.0f);
    verifier("C[0][1]", C[1], 22.0f);
    verifier("C[1][0]", C[2], 43.0f);
    verifier("C[1][1]", C[3], 50.0f);
}

/* -------------------------------------------------------
 * Benchmark : scalaire vs AVX2 sur grande matrice
 * GEMM (64×64) × (64×64) — 200 itérations
 * ------------------------------------------------------- */
#define N 64

static void benchmark(void) {
    static float A[N*N], B[N*N], C_scal[N*N], C_avx2[N*N];

    /* initialiser A et B */
    for (int i = 0; i < N*N; i++) {
        A[i] = (float)(i % 7 + 1);
        B[i] = (float)(i % 5 + 1);
    }

    int iter = 200;
    clock_t t0, t1;

    /* scalaire */
    memset(C_scal, 0, sizeof(C_scal));
    t0 = clock();
    for (int i = 0; i < iter; i++) {
        memset(C_scal, 0, sizeof(C_scal));
        gemm(A, B, C_scal, N, N, N);
    }
    t1 = clock();
    double ms_scal = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

    /* AVX2 */
    memset(C_avx2, 0, sizeof(C_avx2));
    t0 = clock();
    for (int i = 0; i < iter; i++) {
        memset(C_avx2, 0, sizeof(C_avx2));
        gemm_avx2(A, B, C_avx2, N, N, N);
    }
    t1 = clock();
    double ms_avx2 = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

    /* vérifier que les résultats sont identiques */
    int coherent = 1;
    for (int i = 0; i < N*N; i++) {
        if (fabsf(C_scal[i] - C_avx2[i]) > 1e-3f) {
            coherent = 0;
            break;
        }
    }

    printf("\n[Benchmark] GEMM %dx%d × %d itérations\n", N, N, iter);
    printf("  Scalaire : %.2f ms\n", ms_scal);
    printf("  AVX2     : %.2f ms\n", ms_avx2);
    printf("  Speedup  : %.2fx\n", ms_scal / ms_avx2);
    printf("  Résultats cohérents : %s\n", coherent ? "oui" : "NON");
    if (!coherent) erreurs++;
}

int main(void) {
    printf("=== optimatrix — Phase 2 : AVX2 ===");
    test_gemv_avx2();
    test_gemm_avx2();
    benchmark();
    printf("\n%s\n", erreurs == 0
        ? "Toutes les Volontés ont convergé."
        : "Des Volontés ont divergé.");
    return erreurs;
}
