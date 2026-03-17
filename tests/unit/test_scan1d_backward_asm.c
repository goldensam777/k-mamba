/*
 * test_scan1d_backward_asm.c — Test ASM backward vs C backward
 *
 * Compare scan1d_backward_m1_shared_bc_asm() and
 * scan1d_backward_m1_shared_bc_simple_asm() against
 * the C implementation scan1d_backward_m1_shared_bc().
 *
 * Uses precomputed A_diag (both ASM versions require it for
 * the production-quality path).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "scan.h"

#define EPS 1e-4f

/* ── Declarations ─────────────────────────────────────────────── */
extern void scan1d_backward_m1_shared_bc(ScanBackwardSharedParams *p);
extern void scan1d_backward_m1_shared_bc_asm(ScanBackwardSharedParams *p);
extern void scan1d_backward_m1_shared_bc_simple_asm(ScanBackwardSharedParams *p);

/* ── Forward reference (M=1, shared B/C) ──────────────────────── */
static void forward_m1_shared(
    const float *x, const float *A, const float *B, const float *C,
    const float *delta, float *h, float *y,
    long L, long D)
{
    float *state = (float *)calloc(D, sizeof(float));
    memset(y, 0, L * D * sizeof(float));

    for (long t = 0; t < L; t++) {
        float dt = delta[t];
        for (long d = 0; d < D; d++) {
            long td = t * D + d;
            float dA = expf(dt * A[d]);
            float dB = dt * B[d];
            state[d] = dA * state[d] + dB * x[td];
            h[td] = state[d];
            y[td] = C[d] * state[d];
        }
    }
    free(state);
}

/* ── Utilities ────────────────────────────────────────────────── */
static void fill_rand(float *p, long n, float lo, float hi) {
    for (long i = 0; i < n; i++)
        p[i] = lo + (hi - lo) * ((float)rand() / RAND_MAX);
}

static int compare(const char *tag, const float *ref, const float *test,
                   long n, float eps) {
    float worst = 0;
    int bad_idx = -1;
    for (long i = 0; i < n; i++) {
        float d = fabsf(ref[i] - test[i]);
        if (d > worst) { worst = d; bad_idx = (int)i; }
    }
    if (worst > eps) {
        printf("  FAIL %s: worst diff=%.6f at [%d] (ref=%.6f asm=%.6f)\n",
               tag, worst, bad_idx, ref[bad_idx], test[bad_idx]);
        return 0;
    }
    return 1;
}

/* ── Single test case ─────────────────────────────────────────── */
typedef void (*backward_fn)(ScanBackwardSharedParams *);

static int run_test(const char *name, backward_fn asm_fn, long L, long D) {
    printf("  %s (L=%ld D=%ld)... ", name, L, D);

    long LD = L * D;

    /* Allocate inputs */
    float *x     = (float *)malloc(LD * sizeof(float));
    float *A     = (float *)malloc(D * sizeof(float));
    float *B     = (float *)malloc(D * sizeof(float));
    float *C     = (float *)malloc(D * sizeof(float));
    float *delta = (float *)malloc(L * sizeof(float));
    float *dy    = (float *)malloc(LD * sizeof(float));
    float *h     = (float *)malloc(LD * sizeof(float));
    float *y     = (float *)malloc(LD * sizeof(float));

    /* Precomputed A_diag = exp(delta[t] * A[d]) */
    float *A_diag = (float *)malloc(LD * sizeof(float));

    /* C backward outputs (reference) */
    float *dx_c   = (float *)calloc(LD, sizeof(float));
    float *dA_c   = (float *)calloc(D, sizeof(float));
    float *dB_c   = (float *)calloc(D, sizeof(float));
    float *dC_c   = (float *)calloc(D, sizeof(float));
    float *ddt_c  = (float *)calloc(L, sizeof(float));

    /* ASM backward outputs */
    float *dx_a   = (float *)calloc(LD, sizeof(float));
    float *dA_a   = (float *)calloc(D, sizeof(float));
    float *dB_a   = (float *)calloc(D, sizeof(float));
    float *dC_a   = (float *)calloc(D, sizeof(float));
    float *ddt_a  = (float *)calloc(L, sizeof(float));

    /* Fill with random data */
    fill_rand(x,     LD, -1.0f, 1.0f);
    fill_rand(A,     D,  -0.5f, -0.01f);
    fill_rand(B,     D,  -0.3f, 0.3f);
    fill_rand(C,     D,  -0.2f, 0.2f);
    fill_rand(delta, L,   0.01f, 0.1f);
    fill_rand(dy,    LD, -1.0f, 1.0f);

    /* Forward pass to get h */
    forward_m1_shared(x, A, B, C, delta, h, y, L, D);

    /* Precompute A_diag */
    for (long t = 0; t < L; t++)
        for (long d = 0; d < D; d++)
            A_diag[t * D + d] = expf(delta[t] * A[d]);

    /* Run C backward (reference) */
    ScanBackwardSharedParams p_c = {
        .x = x, .A = A, .A_diag = A_diag,
        .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h, .dy = dy,
        .dx = dx_c, .dA = dA_c, .dB = dB_c, .dC = dC_c, .ddelta = ddt_c,
        .L = L, .D = D
    };
    scan1d_backward_m1_shared_bc(&p_c);

    /* Run ASM backward */
    ScanBackwardSharedParams p_a = {
        .x = x, .A = A, .A_diag = A_diag,
        .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h, .dy = dy,
        .dx = dx_a, .dA = dA_a, .dB = dB_a, .dC = dC_a, .ddelta = ddt_a,
        .L = L, .D = D
    };
    asm_fn(&p_a);

    /* Compare all outputs */
    int ok = 1;
    ok &= compare("dx",     dx_c,  dx_a,  LD, EPS);
    ok &= compare("dA",     dA_c,  dA_a,  D,  EPS);
    ok &= compare("dB",     dB_c,  dB_a,  D,  EPS);
    ok &= compare("dC",     dC_c,  dC_a,  D,  EPS);
    ok &= compare("ddelta", ddt_c, ddt_a, L,  EPS);

    if (ok) printf("OK\n");

    /* Cleanup */
    free(x); free(A); free(B); free(C); free(delta); free(dy);
    free(h); free(y); free(A_diag);
    free(dx_c); free(dA_c); free(dB_c); free(dC_c); free(ddt_c);
    free(dx_a); free(dA_a); free(dB_a); free(dC_a); free(ddt_a);

    return ok;
}

/* ── Main ─────────────────────────────────────────────────────── */
int main(void) {
    printf("=== Scan1D Backward ASM Test Suite ===\n\n");
    srand(42);

    int passed = 0, total = 0;

    /* Simple ASM (scalar) */
    printf("Simple ASM (scalar):\n");
    total++; passed += run_test("small",  scan1d_backward_m1_shared_bc_simple_asm, 8,  4);
    total++; passed += run_test("medium", scan1d_backward_m1_shared_bc_simple_asm, 32, 16);
    total++; passed += run_test("large",  scan1d_backward_m1_shared_bc_simple_asm, 64, 32);

    /* AVX2 ASM (vectorized) */
    printf("\nAVX2 ASM (vectorized):\n");
    total++; passed += run_test("small",  scan1d_backward_m1_shared_bc_asm, 8,  4);
    total++; passed += run_test("medium", scan1d_backward_m1_shared_bc_asm, 32, 16);
    total++; passed += run_test("D=17",   scan1d_backward_m1_shared_bc_asm, 16, 17);  /* scalar tail */
    total++; passed += run_test("large",  scan1d_backward_m1_shared_bc_asm, 64, 32);

    printf("\n=== Results: %d/%d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
