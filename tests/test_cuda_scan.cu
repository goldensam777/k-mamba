/*
 * test_cuda_scan.cu — Tests des kernels scan1d CUDA vs reference CPU
 *
 * Valide :
 *   1. Forward : om_scan1d_forward() vs reference C pure
 *   2. Backward : om_scan1d_backward() vs reference C pure
 *
 * Deux tailles testees :
 *   - Petite (L=16, D=4, M=2) : Blelloch path (L <= 1024)
 *   - Grande (L=2048, D=4, M=2) : sequential fallback (L > 1024)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "scan.h"

#define EPSILON_FWD  1e-4f
#define EPSILON_BWD  1e-3f   /* backward accumule plus d'erreur float32 */

/* ── Reference CPU (C pur, pas de SIMD) ────────────────────────── */

static void scan1d_forward_ref(
    const float *x, const float *A,
    const float *B, const float *C,
    const float *dt,
    float *y, float *h,
    int L, int D, int M)
{
    memset(y, 0, L * D * sizeof(float));

    for (int d = 0; d < D; d++) {
        for (int m = 0; m < M; m++) {
            int dm = d * M + m;
            float h_prev = 0.0f;

            for (int t = 0; t < L; t++) {
                int t_d  = t * D + d;
                int t_dm = t * D * M + dm;

                float dt_val = dt[t_d];
                float a = expf(dt_val * A[dm]);
                float b = dt_val * B[t_dm] * x[t_d];

                float h_cur = a * h_prev + b;
                h[t_dm]  = h_cur;
                h_prev   = h_cur;

                y[t_d] += C[t_dm] * h_cur;
            }
        }
    }
}

static void scan1d_backward_ref(
    const float *dy,
    const float *x, const float *A,
    const float *B, const float *C,
    const float *dt, const float *h,
    float *dx, float *dA,
    float *dB, float *dC, float *ddt,
    int L, int D, int M)
{
    memset(dx,  0, L * D     * sizeof(float));
    memset(dA,  0, D * M     * sizeof(float));
    memset(dB,  0, L * D * M * sizeof(float));
    memset(dC,  0, L * D * M * sizeof(float));
    memset(ddt, 0, L * D     * sizeof(float));

    for (int d = 0; d < D; d++) {
        for (int m = 0; m < M; m++) {
            int dm = d * M + m;
            float g_h = 0.0f;

            for (int t = L - 1; t >= 0; t--) {
                int t_d  = t * D + d;
                int t_dm = t * D * M + dm;

                float dt_val = dt[t_d];
                float a_t    = expf(dt_val * A[dm]);
                float h_prev = (t > 0) ? h[(t - 1) * D * M + dm] : 0.0f;

                g_h += C[t_dm] * dy[t_d];

                dC[t_dm] = dy[t_d] * h[t_dm];
                dB[t_dm] = g_h * dt_val * x[t_d];
                dx[t_d] += g_h * dt_val * B[t_dm];
                dA[dm]  += g_h * h_prev * dt_val * a_t;
                ddt[t_d]+= g_h * (h_prev * A[dm] * a_t + B[t_dm] * x[t_d]);

                g_h = a_t * g_h;
            }
        }
    }
}

/* ── Utilitaires ───────────────────────────────────────────────── */

static void fill_random(float *data, int n, float lo, float hi) {
    for (int i = 0; i < n; i++)
        data[i] = lo + (hi - lo) * ((float)rand() / RAND_MAX);
}

static int compare(const char *name, const float *ref, const float *gpu,
                   int n, float eps) {
    float max_diff = 0.0f;
    int   max_idx  = -1;
    int   mismatches = 0;

    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - gpu[i]);
        if (diff > eps) {
            mismatches++;
            if (diff > max_diff) {
                max_diff = diff;
                max_idx  = i;
            }
        }
    }

    if (mismatches > 0) {
        printf("  FAIL %s: %d/%d mismatches, worst [%d] ref=%.6f gpu=%.6f diff=%.6f\n",
               name, mismatches, n, max_idx, ref[max_idx], gpu[max_idx], max_diff);
        return 0;
    }
    return 1;
}

/* ── Forward test ──────────────────────────────────────────────── */

static int test_forward(int L, int D, int M, const char *label) {
    printf("  Forward %s (L=%d D=%d M=%d)... ", label, L, D, M);

    int sz_LD  = L * D;
    int sz_DM  = D * M;
    int sz_LDM = L * D * M;

    /* Host buffers */
    float *h_x  = (float*)malloc(sz_LD  * sizeof(float));
    float *h_A  = (float*)malloc(sz_DM  * sizeof(float));
    float *h_B  = (float*)malloc(sz_LDM * sizeof(float));
    float *h_C  = (float*)malloc(sz_LDM * sizeof(float));
    float *h_dt = (float*)malloc(sz_LD  * sizeof(float));

    float *h_y_ref = (float*)calloc(sz_LD,  sizeof(float));
    float *h_h_ref = (float*)calloc(sz_LDM, sizeof(float));
    float *h_y_gpu = (float*)malloc(sz_LD  * sizeof(float));
    float *h_h_gpu = (float*)malloc(sz_LDM * sizeof(float));

    fill_random(h_x,  sz_LD,  -1.0f, 1.0f);
    fill_random(h_A,  sz_DM,  -0.5f, -0.01f);  /* A negatif pour stabilite */
    fill_random(h_B,  sz_LDM, -0.3f, 0.3f);
    fill_random(h_C,  sz_LDM, -0.2f, 0.2f);
    fill_random(h_dt, sz_LD,   0.01f, 0.1f);

    /* Reference CPU */
    scan1d_forward_ref(h_x, h_A, h_B, h_C, h_dt,
                       h_y_ref, h_h_ref, L, D, M);

    /* GPU */
    float *d_x, *d_A, *d_B, *d_C, *d_dt, *d_y, *d_h;
    OM_CHECK(cudaMalloc(&d_x,  sz_LD  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_A,  sz_DM  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_B,  sz_LDM * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_C,  sz_LDM * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dt, sz_LD  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_y,  sz_LD  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_h,  sz_LDM * sizeof(float)));

    OM_CHECK(cudaMemcpy(d_x,  h_x,  sz_LD  * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_A,  h_A,  sz_DM  * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_B,  h_B,  sz_LDM * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_C,  h_C,  sz_LDM * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_dt, h_dt, sz_LD  * sizeof(float), cudaMemcpyHostToDevice));

    om_scan1d_forward(d_x, d_A, d_B, d_C, d_dt, d_y, d_h, L, D, M);

    OM_CHECK(cudaMemcpy(h_y_gpu, d_y, sz_LD  * sizeof(float), cudaMemcpyDeviceToHost));
    OM_CHECK(cudaMemcpy(h_h_gpu, d_h, sz_LDM * sizeof(float), cudaMemcpyDeviceToHost));

    int ok = 1;
    ok &= compare("y", h_y_ref, h_y_gpu, sz_LD,  EPSILON_FWD);
    ok &= compare("h", h_h_ref, h_h_gpu, sz_LDM, EPSILON_FWD);

    if (ok) printf("OK\n");

    cudaFree(d_x); cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C); cudaFree(d_dt); cudaFree(d_y); cudaFree(d_h);
    free(h_x); free(h_A); free(h_B); free(h_C); free(h_dt);
    free(h_y_ref); free(h_h_ref); free(h_y_gpu); free(h_h_gpu);

    return ok;
}

/* ── Backward test ─────────────────────────────────────────────── */

static int test_backward(int L, int D, int M, const char *label) {
    printf("  Backward %s (L=%d D=%d M=%d)... ", label, L, D, M);

    int sz_LD  = L * D;
    int sz_DM  = D * M;
    int sz_LDM = L * D * M;

    /* Inputs forward */
    float *h_x  = (float*)malloc(sz_LD  * sizeof(float));
    float *h_A  = (float*)malloc(sz_DM  * sizeof(float));
    float *h_B  = (float*)malloc(sz_LDM * sizeof(float));
    float *h_C  = (float*)malloc(sz_LDM * sizeof(float));
    float *h_dt = (float*)malloc(sz_LD  * sizeof(float));
    float *h_dy = (float*)malloc(sz_LD  * sizeof(float));

    /* Forward outputs (needed by backward) */
    float *h_y  = (float*)calloc(sz_LD,  sizeof(float));
    float *h_h  = (float*)calloc(sz_LDM, sizeof(float));

    /* Backward reference outputs */
    float *h_dx_ref  = (float*)calloc(sz_LD,  sizeof(float));
    float *h_dA_ref  = (float*)calloc(sz_DM,  sizeof(float));
    float *h_dB_ref  = (float*)calloc(sz_LDM, sizeof(float));
    float *h_dC_ref  = (float*)calloc(sz_LDM, sizeof(float));
    float *h_ddt_ref = (float*)calloc(sz_LD,  sizeof(float));

    /* Backward GPU outputs */
    float *h_dx_gpu  = (float*)malloc(sz_LD  * sizeof(float));
    float *h_dA_gpu  = (float*)malloc(sz_DM  * sizeof(float));
    float *h_dB_gpu  = (float*)malloc(sz_LDM * sizeof(float));
    float *h_dC_gpu  = (float*)malloc(sz_LDM * sizeof(float));
    float *h_ddt_gpu = (float*)malloc(sz_LD  * sizeof(float));

    fill_random(h_x,  sz_LD,  -1.0f, 1.0f);
    fill_random(h_A,  sz_DM,  -0.5f, -0.01f);
    fill_random(h_B,  sz_LDM, -0.3f, 0.3f);
    fill_random(h_C,  sz_LDM, -0.2f, 0.2f);
    fill_random(h_dt, sz_LD,   0.01f, 0.1f);
    fill_random(h_dy, sz_LD,  -1.0f, 1.0f);

    /* Forward CPU pour obtenir h (necessaire au backward) */
    scan1d_forward_ref(h_x, h_A, h_B, h_C, h_dt, h_y, h_h, L, D, M);

    /* Backward CPU reference */
    scan1d_backward_ref(h_dy, h_x, h_A, h_B, h_C, h_dt, h_h,
                        h_dx_ref, h_dA_ref, h_dB_ref, h_dC_ref, h_ddt_ref,
                        L, D, M);

    /* GPU : forward pour obtenir h, puis backward */
    float *d_x, *d_A, *d_B, *d_C, *d_dt, *d_y, *d_h, *d_dy;
    float *d_dx, *d_dA, *d_dB, *d_dC, *d_ddt;

    OM_CHECK(cudaMalloc(&d_x,   sz_LD  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_A,   sz_DM  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_B,   sz_LDM * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_C,   sz_LDM * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dt,  sz_LD  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_y,   sz_LD  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_h,   sz_LDM * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dy,  sz_LD  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dx,  sz_LD  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dA,  sz_DM  * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dB,  sz_LDM * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_dC,  sz_LDM * sizeof(float)));
    OM_CHECK(cudaMalloc(&d_ddt, sz_LD  * sizeof(float)));

    OM_CHECK(cudaMemcpy(d_x,  h_x,  sz_LD  * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_A,  h_A,  sz_DM  * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_B,  h_B,  sz_LDM * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_C,  h_C,  sz_LDM * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_dt, h_dt, sz_LD  * sizeof(float), cudaMemcpyHostToDevice));
    OM_CHECK(cudaMemcpy(d_dy, h_dy, sz_LD  * sizeof(float), cudaMemcpyHostToDevice));

    /* Forward GPU pour obtenir d_h */
    om_scan1d_forward(d_x, d_A, d_B, d_C, d_dt, d_y, d_h, L, D, M);

    /* Backward GPU */
    om_scan1d_backward(d_dy, d_x, d_A, d_B, d_C, d_dt, d_h,
                       d_dx, d_dA, d_dB, d_dC, d_ddt, L, D, M);

    OM_CHECK(cudaMemcpy(h_dx_gpu,  d_dx,  sz_LD  * sizeof(float), cudaMemcpyDeviceToHost));
    OM_CHECK(cudaMemcpy(h_dA_gpu,  d_dA,  sz_DM  * sizeof(float), cudaMemcpyDeviceToHost));
    OM_CHECK(cudaMemcpy(h_dB_gpu,  d_dB,  sz_LDM * sizeof(float), cudaMemcpyDeviceToHost));
    OM_CHECK(cudaMemcpy(h_dC_gpu,  d_dC,  sz_LDM * sizeof(float), cudaMemcpyDeviceToHost));
    OM_CHECK(cudaMemcpy(h_ddt_gpu, d_ddt, sz_LD  * sizeof(float), cudaMemcpyDeviceToHost));

    int ok = 1;
    ok &= compare("dx",  h_dx_ref,  h_dx_gpu,  sz_LD,  EPSILON_BWD);
    ok &= compare("dA",  h_dA_ref,  h_dA_gpu,  sz_DM,  EPSILON_BWD);
    ok &= compare("dB",  h_dB_ref,  h_dB_gpu,  sz_LDM, EPSILON_BWD);
    ok &= compare("dC",  h_dC_ref,  h_dC_gpu,  sz_LDM, EPSILON_BWD);
    ok &= compare("ddt", h_ddt_ref, h_ddt_gpu, sz_LD,  EPSILON_BWD);

    if (ok) printf("OK\n");

    cudaFree(d_x); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_dt); cudaFree(d_y); cudaFree(d_h); cudaFree(d_dy);
    cudaFree(d_dx); cudaFree(d_dA); cudaFree(d_dB); cudaFree(d_dC);
    cudaFree(d_ddt);

    free(h_x); free(h_A); free(h_B); free(h_C); free(h_dt); free(h_dy);
    free(h_y); free(h_h);
    free(h_dx_ref); free(h_dA_ref); free(h_dB_ref); free(h_dC_ref); free(h_ddt_ref);
    free(h_dx_gpu); free(h_dA_gpu); free(h_dB_gpu); free(h_dC_gpu); free(h_ddt_gpu);

    return ok;
}

/* ── Main ──────────────────────────────────────────────────────── */

int main() {
    printf("=== CUDA Scan1D Test Suite ===\n\n");

    srand(42);

    int passed = 0, total = 0;

    /* Forward : Blelloch path (L <= 1024) */
    total++; passed += test_forward(16,  4, 2, "small/Blelloch");
    total++; passed += test_forward(128, 8, 4, "medium/Blelloch");
    total++; passed += test_forward(512, 4, 2, "large/Blelloch");

    /* Forward : sequential fallback (L > 1024) */
    total++; passed += test_forward(2048, 4, 2, "seq-fallback");

    /* Backward (uses sequential kernel for all L) */
    total++; passed += test_backward(16,  4, 2, "small");
    total++; passed += test_backward(128, 8, 4, "medium");
    total++; passed += test_backward(512, 4, 2, "large");

    printf("\n=== Results: %d/%d passed ===\n", passed, total);

    return (passed == total) ? 0 : 1;
}
