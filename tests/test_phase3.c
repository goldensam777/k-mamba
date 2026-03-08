/*
 * test_phase3.c — Validation du Selective Scan 1D et 2D (float32)
 *
 * Stratégie : comparer chaque résultat ASM avec une référence C
 * calculée avec le même algorithme. Si les résultats coïncident
 * à 1e-5 près, la Volonté a convergé.
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ---- Structures miroir des struc NASM ---- */

typedef struct {
    float  *x;
    float  *A;
    float  *B;
    float  *C;
    float  *delta;
    float  *h;
    float  *y;
    long    L;
    long    D;
    long    M;
} ScanParams;

typedef struct {
    float  *x;
    float  *A;
    float  *B;
    float  *C;
    float  *delta;
    float  *h0;
    float  *h;
    float  *dy;
    float  *dx;
    float  *dA;
    float  *dB;
    float  *dC;
    float  *ddelta;
    long    L;
    long    D;
    long    M;
} ScanBackwardParams;

typedef struct {
    float  *x;
    float  *A;
    float  *A_diag;
    float  *B;
    float  *C;
    float  *delta;
    float  *h0;
    float  *h;
    float  *dy;
    float  *dx;
    float  *dA;
    float  *dB;
    float  *dC;
    float  *ddelta;
    long    L;
    long    D;
} ScanBackwardSharedParams;

typedef struct {
    float  *x;
    float  *A1;
    float  *A2;
    float  *B;
    float  *C;
    float  *delta1;
    float  *delta2;
    float  *h;
    float  *y;
    long    d1;
    long    d2;
    long    D;
    long    M;
} Scan2DParams;

/* ---- Déclarations ASM ---- */
extern void scan1d(ScanParams *p);
extern void scan1d_backward(ScanBackwardParams *p);
extern void scan1d_backward_m1_shared_bc(ScanBackwardSharedParams *p);
extern void scan2d(Scan2DParams *p);

static int erreurs = 0;

static void verifier(const char *nom, float obtenu, float attendu) {
    if (fabsf(obtenu - attendu) < 1e-5f) {
        printf("  OK  %s = %.6f\n", nom, obtenu);
    } else {
        printf("  FAIL %s : obtenu=%.8f attendu=%.8f (diff=%.2e)\n",
               nom, obtenu, attendu, fabsf(obtenu - attendu));
        erreurs++;
    }
}

static void verifier_tol(const char *nom, float obtenu, float attendu, float tol) {
    if (fabsf(obtenu - attendu) < tol) {
        printf("  OK  %s = %.6f\n", nom, obtenu);
    } else {
        printf("  FAIL %s : obtenu=%.8f attendu=%.8f (diff=%.2e)\n",
               nom, obtenu, attendu, fabsf(obtenu - attendu));
        erreurs++;
    }
}

/* ============================================================
 * Référence C — Scan 1D
 * ============================================================ */
static void scan1d_ref(ScanParams *p) {
    for (long t = 0; t < p->L; t++) {
        for (long d = 0; d < p->D; d++) {
            long td = t * p->D + d;
            float dt = p->delta[td];
            float xt = p->x[td];
            p->y[td] = 0.0f;
            for (long m = 0; m < p->M; m++) {
                long dm  = d * p->M + m;
                long tdm = td * p->M + m;
                float dA = expf(dt * p->A[dm]);
                float dB = dt * p->B[tdm];
                p->h[dm] = dA * p->h[dm] + dB * xt;
                p->y[td] += p->C[tdm] * p->h[dm];
            }
        }
    }
}

static void scan1d_forward_history_ref(const float *x, const float *A,
                                      const float *B, const float *C,
                                      const float *delta, float *h_hist,
                                      float *y, long L, long D, long M) {
    float *h = (float *)calloc((size_t)(D * M), sizeof(float));
    if (!h) return;

    for (long t = 0; t < L; t++) {
        for (long d = 0; d < D; d++) {
            long td = t * D + d;
            float dt = delta[td];
            float xt = x[td];
            y[td] = 0.0f;

            for (long m = 0; m < M; m++) {
                long dm = d * M + m;
                long tdm = td * M + m;
                float dA = expf(dt * A[dm]);
                float dB = dt * B[tdm];

                h[dm] = dA * h[dm] + dB * xt;
                h_hist[tdm] = h[dm];
                y[td] += C[tdm] * h[dm];
            }
        }
    }

    free(h);
}

static float scan1d_loss_ref(const float *x, const float *A, const float *B,
                             const float *C, const float *delta,
                             const float *dy, long L, long D, long M) {
    size_t y_size = (size_t)(L * D);
    size_t h_size = y_size * (size_t)M;
    float *h_hist = (float *)calloc(h_size, sizeof(float));
    float *y = (float *)calloc(y_size, sizeof(float));
    float loss = 0.0f;

    if (!h_hist || !y) {
        free(h_hist);
        free(y);
        return 0.0f;
    }

    scan1d_forward_history_ref(x, A, B, C, delta, h_hist, y, L, D, M);
    for (size_t i = 0; i < y_size; i++) loss += y[i] * dy[i];

    free(h_hist);
    free(y);
    return loss;
}

static void scan1d_forward_history_shared_ref(const float *x, const float *A,
                                             const float *B, const float *C,
                                             const float *delta, float *h_hist,
                                             float *a_diag_hist, float *y,
                                             long L, long D) {
    float *h = (float *)calloc((size_t)D, sizeof(float));
    if (!h) return;

    for (long t = 0; t < L; t++) {
        float dt = delta[t];
        for (long d = 0; d < D; d++) {
            long td = t * D + d;
            float dA = expf(dt * A[d]);
            float dB = dt * B[d];

            h[d] = dA * h[d] + dB * x[td];
            h_hist[td] = h[d];
            if (a_diag_hist) a_diag_hist[td] = dA;
            y[td] = C[d] * h[d];
        }
    }

    free(h);
}

static float scan1d_loss_shared_ref(const float *x, const float *A, const float *B,
                                    const float *C, const float *delta,
                                    const float *dy, long L, long D) {
    size_t size = (size_t)(L * D);
    float *h_hist = (float *)calloc(size, sizeof(float));
    float *y = (float *)calloc(size, sizeof(float));
    float loss = 0.0f;

    if (!h_hist || !y) {
        free(h_hist);
        free(y);
        return 0.0f;
    }

    scan1d_forward_history_shared_ref(x, A, B, C, delta, h_hist, NULL, y, L, D);
    for (size_t i = 0; i < size; i++) loss += y[i] * dy[i];

    free(h_hist);
    free(y);
    return loss;
}

/* ============================================================
 * Référence C — Scan 2D (wavefront)
 * ============================================================ */
static void scan2d_ref(Scan2DParams *p) {
    long DM = p->D * p->M;

    for (long k = 0; k < p->d1 + p->d2 - 1; k++) {
        long i_min = (k - p->d2 + 1 > 0) ? k - p->d2 + 1 : 0;
        long i_max = (k < p->d1 - 1)      ? k              : p->d1 - 1;

        for (long i = i_min; i <= i_max; i++) {
            long j       = k - i;
            long pos     = i * p->d2 + j;

            for (long d = 0; d < p->D; d++) {
                long pd  = pos * p->D + d;
                p->y[pd] = 0.0f;

                for (long m = 0; m < p->M; m++) {
                    long dm   = d * p->M + m;
                    long pDM  = pos * DM + dm;

                    /* h_prev1 : h(i-1, j) */
                    float hp1 = 0.0f;
                    if (i > 0) hp1 = p->h[(pos - p->d2) * DM + dm];

                    /* h_prev2 : h(i, j-1) */
                    float hp2 = 0.0f;
                    if (j > 0) hp2 = p->h[(pos - 1) * DM + dm];

                    float dt1 = p->delta1[pd];
                    float dt2 = p->delta2[pd];
                    float dA1 = expf(dt1 * p->A1[dm]);
                    float dA2 = expf(dt2 * p->A2[dm]);
                    float dB  = (dt1 + dt2) * 0.5f * p->B[pDM];
                    float x_v = p->x[pd];

                    p->h[pDM] = dA1 * hp1 + dA2 * hp2 + dB * x_v;
                    p->y[pd] += p->C[pDM] * p->h[pDM];
                }
            }
        }
    }
}

/* ============================================================
 * Test Scan 1D
 * L=4, D=2, M=2
 * ============================================================ */
static void test_scan1d(void) {
    printf("\n[Scan 1D] L=4, D=2, M=2\n");

    long L=4, D=2, M=2;

    float x[8]  = {1.0f, 0.5f, 2.0f, 1.5f, 0.8f, 1.2f, 1.1f, 0.9f};
    float A[4]  = {-0.1f, -0.2f, -0.3f, -0.15f};
    float B[16] = {0.5f,0.3f, 0.6f,0.4f, 0.7f,0.2f, 0.4f,0.5f,
                   0.3f,0.6f, 0.5f,0.3f, 0.8f,0.1f, 0.2f,0.7f};
    float C[16] = {1.0f,0.5f, 0.8f,0.6f, 0.9f,0.4f, 0.7f,0.3f,
                   0.6f,0.8f, 0.5f,0.9f, 0.4f,0.7f, 0.3f,0.6f};
    float delta[8] = {0.1f,0.2f, 0.15f,0.25f, 0.1f,0.2f, 0.15f,0.25f};

    /* Référence C */
    float h_ref[4] = {0,0,0,0};
    float y_ref[8] = {0,0,0,0,0,0,0,0};
    ScanParams ref = { x,A,B,C,delta, h_ref, y_ref, L,D,M };
    scan1d_ref(&ref);

    /* Version ASM */
    float h_asm[4] = {0,0,0,0};
    float y_asm[8] = {0,0,0,0,0,0,0,0};
    ScanParams asm_ = { x,A,B,C,delta, h_asm, y_asm, L,D,M };
    scan1d(&asm_);

    /* Comparer */
    char nom[32];
    for (int t = 0; t < L; t++) {
        for (int d = 0; d < D; d++) {
            snprintf(nom, sizeof(nom), "y[%d][%d]", t, d);
            verifier(nom, y_asm[t*D+d], y_ref[t*D+d]);
        }
    }
}

static void test_scan1d_backward(void) {
    printf("\n[Scan 1D Backward] L=3, D=2, M=2\n");

    long L = 3, D = 2, M = 2;
    size_t x_size = (size_t)(L * D);
    size_t dm_size = (size_t)(D * M);
    size_t tdm_size = x_size * (size_t)M;
    const float eps = 1e-3f;
    const float tol = 6e-3f;

    float x[6] = {1.0f, 0.5f, 1.5f, 1.2f, 0.8f, 1.1f};
    float A[4] = {-0.2f, -0.1f, -0.3f, -0.25f};
    float B[12] = {
        0.4f, 0.2f, 0.5f, 0.3f,
        0.6f, 0.1f, 0.7f, 0.2f,
        0.3f, 0.8f, 0.4f, 0.5f
    };
    float C[12] = {
        1.0f, 0.7f, 0.8f, 0.6f,
        0.9f, 0.5f, 0.4f, 0.3f,
        0.6f, 0.2f, 0.7f, 0.9f
    };
    float delta[6] = {0.10f, 0.15f, 0.20f, 0.25f, 0.12f, 0.18f};
    float dy[6] = {0.3f, -0.4f, 0.6f, 0.2f, -0.5f, 0.7f};

    float h_hist[12] = {0};
    float y[6] = {0};
    float dx[6] = {0};
    float dA[4] = {0};
    float dB[12] = {0};
    float dC[12] = {0};
    float ddelta[6] = {0};

    scan1d_forward_history_ref(x, A, B, C, delta, h_hist, y, L, D, M);

    ScanBackwardParams bp = {
        .x = x, .A = A, .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h_hist, .dy = dy,
        .dx = dx, .dA = dA, .dB = dB, .dC = dC, .ddelta = ddelta,
        .L = L, .D = D, .M = M
    };
    scan1d_backward(&bp);

    for (size_t i = 0; i < x_size; i++) {
        float saved = x[i];
        float loss_pos, loss_neg;
        char nom[32];

        x[i] = saved + eps;
        loss_pos = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        x[i] = saved - eps;
        loss_neg = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        x[i] = saved;

        snprintf(nom, sizeof(nom), "dx[%zu]", i);
        verifier_tol(nom, dx[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }

    for (size_t i = 0; i < dm_size; i++) {
        float saved = A[i];
        float loss_pos, loss_neg;
        char nom[32];

        A[i] = saved + eps;
        loss_pos = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        A[i] = saved - eps;
        loss_neg = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        A[i] = saved;

        snprintf(nom, sizeof(nom), "dA[%zu]", i);
        verifier_tol(nom, dA[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }

    for (size_t i = 0; i < tdm_size; i++) {
        float saved = B[i];
        float loss_pos, loss_neg;
        char nom[32];

        B[i] = saved + eps;
        loss_pos = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        B[i] = saved - eps;
        loss_neg = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        B[i] = saved;

        snprintf(nom, sizeof(nom), "dB[%zu]", i);
        verifier_tol(nom, dB[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }

    for (size_t i = 0; i < tdm_size; i++) {
        float saved = C[i];
        float loss_pos, loss_neg;
        char nom[32];

        C[i] = saved + eps;
        loss_pos = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        C[i] = saved - eps;
        loss_neg = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        C[i] = saved;

        snprintf(nom, sizeof(nom), "dC[%zu]", i);
        verifier_tol(nom, dC[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }

    for (size_t i = 0; i < x_size; i++) {
        float saved = delta[i];
        float loss_pos, loss_neg;
        char nom[32];

        delta[i] = saved + eps;
        loss_pos = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        delta[i] = saved - eps;
        loss_neg = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        delta[i] = saved;

        snprintf(nom, sizeof(nom), "ddelta[%zu]", i);
        verifier_tol(nom, ddelta[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }
}

static void test_scan1d_backward_m1(void) {
    printf("\n[Scan 1D Backward] L=4, D=3, M=1\n");

    long L = 4, D = 3, M = 1;
    size_t x_size = (size_t)(L * D);
    size_t dm_size = (size_t)(D * M);
    size_t tdm_size = x_size * (size_t)M;
    const float eps = 1e-3f;
    const float tol = 6e-3f;

    float x[12] = {
        1.0f, 0.5f, 0.8f,
        1.3f, 0.7f, 1.1f,
        0.9f, 1.2f, 0.6f,
        1.4f, 0.4f, 1.0f
    };
    float A[3] = {-0.15f, -0.25f, -0.35f};
    float B[12] = {
        0.4f, 0.3f, 0.2f,
        0.5f, 0.1f, 0.6f,
        0.7f, 0.4f, 0.3f,
        0.2f, 0.8f, 0.5f
    };
    float C[12] = {
        1.0f, 0.7f, 0.6f,
        0.8f, 0.5f, 0.9f,
        0.4f, 1.1f, 0.3f,
        0.6f, 0.2f, 1.0f
    };
    float delta[12] = {
        0.10f, 0.12f, 0.14f,
        0.16f, 0.18f, 0.20f,
        0.11f, 0.13f, 0.15f,
        0.17f, 0.19f, 0.21f
    };
    float dy[12] = {
        0.3f, -0.2f, 0.4f,
        0.6f, 0.1f, -0.5f,
        -0.4f, 0.7f, 0.2f,
        0.5f, -0.3f, 0.8f
    };

    float h_hist[12] = {0};
    float A_diag_hist[12] = {0};
    float y[12] = {0};
    float dx[12] = {0};
    float dA[3] = {0};
    float dB[12] = {0};
    float dC[12] = {0};
    float ddelta[12] = {0};

    scan1d_forward_history_ref(x, A, B, C, delta, h_hist, y, L, D, M);

    ScanBackwardParams bp = {
        .x = x, .A = A, .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h_hist, .dy = dy,
        .dx = dx, .dA = dA, .dB = dB, .dC = dC, .ddelta = ddelta,
        .L = L, .D = D, .M = M
    };
    scan1d_backward(&bp);

    for (size_t i = 0; i < x_size; i++) {
        float saved = x[i];
        float loss_pos, loss_neg;
        char nom[32];

        x[i] = saved + eps;
        loss_pos = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        x[i] = saved - eps;
        loss_neg = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        x[i] = saved;

        snprintf(nom, sizeof(nom), "dx_m1[%zu]", i);
        verifier_tol(nom, dx[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }

    for (size_t i = 0; i < dm_size; i++) {
        float saved = A[i];
        float loss_pos, loss_neg;
        char nom[32];

        A[i] = saved + eps;
        loss_pos = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        A[i] = saved - eps;
        loss_neg = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        A[i] = saved;

        snprintf(nom, sizeof(nom), "dA_m1[%zu]", i);
        verifier_tol(nom, dA[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }

    for (size_t i = 0; i < tdm_size; i++) {
        float saved = B[i];
        float loss_pos, loss_neg;
        char nom[32];

        B[i] = saved + eps;
        loss_pos = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        B[i] = saved - eps;
        loss_neg = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        B[i] = saved;

        snprintf(nom, sizeof(nom), "dB_m1[%zu]", i);
        verifier_tol(nom, dB[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }

    for (size_t i = 0; i < tdm_size; i++) {
        float saved = C[i];
        float loss_pos, loss_neg;
        char nom[32];

        C[i] = saved + eps;
        loss_pos = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        C[i] = saved - eps;
        loss_neg = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        C[i] = saved;

        snprintf(nom, sizeof(nom), "dC_m1[%zu]", i);
        verifier_tol(nom, dC[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }

    for (size_t i = 0; i < x_size; i++) {
        float saved = delta[i];
        float loss_pos, loss_neg;
        char nom[32];

        delta[i] = saved + eps;
        loss_pos = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        delta[i] = saved - eps;
        loss_neg = scan1d_loss_ref(x, A, B, C, delta, dy, L, D, M);
        delta[i] = saved;

        snprintf(nom, sizeof(nom), "ddelta_m1[%zu]", i);
        verifier_tol(nom, ddelta[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }
}

static void test_scan1d_backward_m1_shared_bc(void) {
    printf("\n[Scan 1D Backward] L=4, D=3, M=1 shared B/C\n");

    long L = 4, D = 3;
    size_t x_size = (size_t)(L * D);
    const float eps = 1e-3f;
    const float tol = 6e-3f;

    float x[12] = {
        1.0f, 0.5f, 0.8f,
        1.3f, 0.7f, 1.1f,
        0.9f, 1.2f, 0.6f,
        1.4f, 0.4f, 1.0f
    };
    float A[3] = {-0.15f, -0.25f, -0.35f};
    float B[3] = {0.4f, 0.3f, 0.2f};
    float C[3] = {1.0f, 0.7f, 0.6f};
    float delta[4] = {0.10f, 0.16f, 0.11f, 0.17f};
    float dy[12] = {
        0.3f, -0.2f, 0.4f,
        0.6f, 0.1f, -0.5f,
        -0.4f, 0.7f, 0.2f,
        0.5f, -0.3f, 0.8f
    };

    float h_hist[12] = {0};
    float A_diag_hist[12] = {0};
    float y[12] = {0};
    float dx[12] = {0};
    float dA[3] = {0};
    float dB[3] = {0};
    float dC[3] = {0};
    float ddelta[4] = {0};

    scan1d_forward_history_shared_ref(x, A, B, C, delta, h_hist, A_diag_hist, y, L, D);

    ScanBackwardSharedParams bp = {
        .x = x, .A = A, .A_diag = A_diag_hist, .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h_hist, .dy = dy,
        .dx = dx, .dA = dA, .dB = dB, .dC = dC, .ddelta = ddelta,
        .L = L, .D = D
    };
    scan1d_backward_m1_shared_bc(&bp);

    for (size_t i = 0; i < x_size; i++) {
        float saved = x[i];
        float loss_pos, loss_neg;
        char nom[40];

        x[i] = saved + eps;
        loss_pos = scan1d_loss_shared_ref(x, A, B, C, delta, dy, L, D);
        x[i] = saved - eps;
        loss_neg = scan1d_loss_shared_ref(x, A, B, C, delta, dy, L, D);
        x[i] = saved;

        snprintf(nom, sizeof(nom), "dx_shared[%zu]", i);
        verifier_tol(nom, dx[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }

    for (size_t i = 0; i < (size_t)D; i++) {
        float saved = A[i];
        float loss_pos, loss_neg;
        char nom[40];

        A[i] = saved + eps;
        loss_pos = scan1d_loss_shared_ref(x, A, B, C, delta, dy, L, D);
        A[i] = saved - eps;
        loss_neg = scan1d_loss_shared_ref(x, A, B, C, delta, dy, L, D);
        A[i] = saved;

        snprintf(nom, sizeof(nom), "dA_shared[%zu]", i);
        verifier_tol(nom, dA[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }

    for (size_t i = 0; i < (size_t)D; i++) {
        float saved = B[i];
        float loss_pos, loss_neg;
        char nom[40];

        B[i] = saved + eps;
        loss_pos = scan1d_loss_shared_ref(x, A, B, C, delta, dy, L, D);
        B[i] = saved - eps;
        loss_neg = scan1d_loss_shared_ref(x, A, B, C, delta, dy, L, D);
        B[i] = saved;

        snprintf(nom, sizeof(nom), "dB_shared[%zu]", i);
        verifier_tol(nom, dB[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }

    for (size_t i = 0; i < (size_t)D; i++) {
        float saved = C[i];
        float loss_pos, loss_neg;
        char nom[40];

        C[i] = saved + eps;
        loss_pos = scan1d_loss_shared_ref(x, A, B, C, delta, dy, L, D);
        C[i] = saved - eps;
        loss_neg = scan1d_loss_shared_ref(x, A, B, C, delta, dy, L, D);
        C[i] = saved;

        snprintf(nom, sizeof(nom), "dC_shared[%zu]", i);
        verifier_tol(nom, dC[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }

    for (size_t i = 0; i < (size_t)L; i++) {
        float saved = delta[i];
        float loss_pos, loss_neg;
        char nom[40];

        delta[i] = saved + eps;
        loss_pos = scan1d_loss_shared_ref(x, A, B, C, delta, dy, L, D);
        delta[i] = saved - eps;
        loss_neg = scan1d_loss_shared_ref(x, A, B, C, delta, dy, L, D);
        delta[i] = saved;

        snprintf(nom, sizeof(nom), "ddelta_shared[%zu]", i);
        verifier_tol(nom, ddelta[i], (loss_pos - loss_neg) / (2.0f * eps), tol);
    }
}

/* ============================================================
 * Test Scan 2D
 * d1=3, d2=3, D=1, M=1
 * ============================================================ */
static void test_scan2d(void) {
    printf("\n[Scan 2D] d1=3, d2=3, D=1, M=1 (wavefront)\n");

    long d1=3, d2=3, D=1, M=1;

    float x[9]      = {1.0f, 0.5f, 0.8f, 1.2f, 0.9f, 1.1f, 0.7f, 1.3f, 0.6f};
    float A1[1]     = {-0.1f};
    float A2[1]     = {-0.2f};
    float B[9]      = {0.5f, 0.6f, 0.4f, 0.7f, 0.3f, 0.8f, 0.5f, 0.6f, 0.4f};
    float C[9]      = {1.0f, 0.8f, 0.9f, 0.7f, 1.1f, 0.6f, 0.8f, 0.7f, 0.9f};
    float delta1[9] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float delta2[9] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f};

    /* Référence C */
    float h_ref[9] = {0};
    float y_ref[9] = {0};
    Scan2DParams ref = { x,A1,A2,B,C,delta1,delta2, h_ref,y_ref, d1,d2,D,M };
    scan2d_ref(&ref);

    /* Version ASM */
    float h_asm[9] = {0};
    float y_asm[9] = {0};
    Scan2DParams asm_ = { x,A1,A2,B,C,delta1,delta2, h_asm,y_asm, d1,d2,D,M };
    scan2d(&asm_);

    /* Comparer */
    char nom[32];
    for (int i = 0; i < d1; i++) {
        for (int j = 0; j < d2; j++) {
            snprintf(nom, sizeof(nom), "y[%d][%d]", i, j);
            verifier(nom, y_asm[i*d2+j], y_ref[i*d2+j]);
        }
    }
}

int main(void) {
    printf("=== optimatrix — Phase 3 : Selective Scan ===");
    test_scan1d();
    test_scan1d_backward_m1_shared_bc();
    test_scan1d_backward_m1();
    test_scan1d_backward();
    test_scan2d();
    printf("\n%s\n", erreurs == 0
        ? "Toutes les Volontés ont convergé."
        : "Des Volontés ont divergé.");
    return erreurs;
}
