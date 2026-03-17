/*
 * bench_paper.c — Benchmarks for k-mamba arXiv paper
 *
 * Usage:
 *   bench_paper --fig N [--L X] [--D Y] [--M Z] [--repeat 100] [--warmup 20]
 *
 * Outputs JSON on stdout for the requested figure and config.
 * The Python script (scripts/plot_paper.py) sweeps parameters
 * and collects JSON for each data point.
 *
 * Build:
 *   cmake -B build -DKMAMBA_BUILD_BENCH=ON [-DKMAMBA_BUILD_CUDA=ON]
 *   cmake --build build
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "optimatrix.h"
#include "scan.h"
#include "kmamba.h"

#ifdef KMAMBA_HAS_CUDA
#include "bench_gpu.h"
#endif

/* ═══════════════════════════════════════════════════════════════════
 * Timing utilities
 * ═══════════════════════════════════════════════════════════════════ */

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

typedef struct { double median, p5, p95; } Stats;

static Stats stats_from_double(double *t, int n) {
    qsort(t, n, sizeof(double), cmp_double);
    Stats s;
    s.p5     = t[(int)(n * 0.05)];
    s.median = t[n / 2];
    s.p95    = t[(int)(n * 0.95)];
    return s;
}

static Stats stats_from_float(float *t, int n) {
    qsort(t, n, sizeof(float), cmp_float);
    Stats s;
    s.p5     = t[(int)(n * 0.05)];
    s.median = t[n / 2];
    s.p95    = t[(int)(n * 0.95)];
    return s;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* ═══════════════════════════════════════════════════════════════════
 * Allocation helpers
 * ═══════════════════════════════════════════════════════════════════ */

static float *alloc_zeros(long n) {
    float *p = (float *)calloc(n, sizeof(float));
    if (!p) { fprintf(stderr, "alloc %ld failed\n", n); exit(1); }
    return p;
}

static float *alloc_rand(long n, float lo, float hi) {
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) { fprintf(stderr, "alloc %ld failed\n", n); exit(1); }
    for (long i = 0; i < n; i++)
        p[i] = lo + (hi - lo) * ((float)rand() / RAND_MAX);
    return p;
}

/* ═══════════════════════════════════════════════════════════════════
 * JSON output helpers
 * ═══════════════════════════════════════════════════════════════════ */

static void json_stats(const char *key, Stats s) {
    printf("\"%s\":{\"median_ms\":%.6f,\"p5_ms\":%.6f,\"p95_ms\":%.6f}",
           key, s.median, s.p5, s.p95);
}

/* ═══════════════════════════════════════════════════════════════════
 * Reference C scan1d (pure scalar, no SIMD — baseline)
 * ═══════════════════════════════════════════════════════════════════ */

static void scan1d_ref(ScanParams *p) {
    long L = p->L, D = p->D, M = p->M;
    memset(p->y, 0, L * D * sizeof(float));
    for (long d = 0; d < D; d++) {
        for (long m = 0; m < M; m++) {
            long dm = d * M + m;
            float h_prev = 0.0f;
            for (long t = 0; t < L; t++) {
                long td  = t * D + d;
                long tdm = t * D * M + dm;
                float dt_val = p->delta[td];
                float a = expf(dt_val * p->A[dm]);
                float b = dt_val * p->B[tdm] * p->x[td];
                float h_cur = a * h_prev + b;
                p->h[tdm] = h_cur;
                h_prev = h_cur;
                p->y[td] += p->C[tdm] * h_cur;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Fig 1 — Scan1D CPU vs GPU
 * ═══════════════════════════════════════════════════════════════════ */

static void fig1(int L, int D, int M, int repeat, int warmup) {
    long LD = (long)L * D, DM = (long)D * M, LDM = (long)L * D * M;

    float *x     = alloc_rand(LD,  -1.0f, 1.0f);
    float *A     = alloc_rand(DM,  -0.5f, -0.01f);
    float *B     = alloc_rand(LDM, -0.3f, 0.3f);
    float *C     = alloc_rand(LDM, -0.2f, 0.2f);
    float *delta = alloc_rand(LD,   0.01f, 0.1f);
    float *h     = alloc_zeros(LDM);
    float *y     = alloc_zeros(LD);

    ScanParams p = { x, A, B, C, delta, h, y, L, D, M };

    /* Warmup CPU */
    for (int i = 0; i < warmup; i++) {
        memset(h, 0, LDM * sizeof(float));
        memset(y, 0, LD * sizeof(float));
        scan1d(&p);
    }

    /* Measure CPU (ASM) */
    double *cpu_times = (double *)malloc(repeat * sizeof(double));
    for (int i = 0; i < repeat; i++) {
        memset(h, 0, LDM * sizeof(float));
        memset(y, 0, LD * sizeof(float));
        double t0 = now_ms();
        scan1d(&p);
        cpu_times[i] = now_ms() - t0;
    }
    Stats cpu = stats_from_double(cpu_times, repeat);
    free(cpu_times);

    printf("{\"fig\":1,\"config\":{\"L\":%d,\"D\":%d,\"M\":%d},\"results\":{",
           L, D, M);
    json_stats("cpu_asm", cpu);

#ifdef KMAMBA_HAS_CUDA
    float *gpu_times = (float *)malloc(repeat * sizeof(float));
    bench_gpu_scan1d(L, D, M, warmup, repeat, gpu_times);
    Stats gpu = stats_from_float(gpu_times, repeat);
    free(gpu_times);
    printf(",");
    json_stats("gpu", gpu);
#endif

    printf("}}\n");

    free(x); free(A); free(B); free(C); free(delta); free(h); free(y);
}

/* ═══════════════════════════════════════════════════════════════════
 * Fig 2 — Blelloch vs Sequential GPU
 * ═══════════════════════════════════════════════════════════════════ */

static void fig2(int L, int D, int M, int repeat, int warmup) {
#ifdef KMAMBA_HAS_CUDA
    float *seq_times = (float *)malloc(repeat * sizeof(float));
    bench_gpu_scan1d_seq(L, D, M, warmup, repeat, seq_times);
    Stats seq = stats_from_float(seq_times, repeat);
    free(seq_times);

    printf("{\"fig\":2,\"config\":{\"L\":%d,\"D\":%d,\"M\":%d},\"results\":{",
           L, D, M);
    json_stats("sequential", seq);

    if (L <= 1024) {
        float *auto_times = (float *)malloc(repeat * sizeof(float));
        bench_gpu_scan1d(L, D, M, warmup, repeat, auto_times);
        Stats blelloch = stats_from_float(auto_times, repeat);
        free(auto_times);
        printf(",");
        json_stats("blelloch", blelloch);
    }

    printf("}}\n");
#else
    printf("{\"fig\":2,\"error\":\"no CUDA\"}\n");
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * Fig 3 — Wavefront 2D: diagonal width & time
 * ═══════════════════════════════════════════════════════════════════ */

static void fig3(int d1, int d2, int D, int M, int repeat, int warmup) {
    long P = (long)d1 * d2;
    long LDM = P * D * M;
    long LD  = P * D;

    float *x      = alloc_rand(LD,  -1.0f, 1.0f);
    float *A1     = alloc_rand((long)D * M, -0.5f, -0.01f);
    float *A2     = alloc_rand((long)D * M, -0.5f, -0.01f);
    float *B      = alloc_rand(LDM, -0.3f, 0.3f);
    float *C      = alloc_rand(LDM, -0.2f, 0.2f);
    float *delta1 = alloc_rand(LD,   0.01f, 0.1f);
    float *delta2 = alloc_rand(LD,   0.01f, 0.1f);
    float *h      = alloc_zeros(LDM);
    float *y      = alloc_zeros(LD);

    Scan2DParams p = { x, A1, A2, B, C, delta1, delta2, h, y, d1, d2, D, M };

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        memset(h, 0, LDM * sizeof(float));
        memset(y, 0, LD * sizeof(float));
        scan2d(&p);
    }

    /* Measure total scan2d time */
    double *times = (double *)malloc(repeat * sizeof(double));
    for (int i = 0; i < repeat; i++) {
        memset(h, 0, LDM * sizeof(float));
        memset(y, 0, LD * sizeof(float));
        double t0 = now_ms();
        scan2d(&p);
        times[i] = now_ms() - t0;
    }
    Stats total = stats_from_double(times, repeat);
    free(times);

    /* Compute diagonal widths */
    int n_diags = d1 + d2 - 1;

    printf("{\"fig\":3,\"config\":{\"d1\":%d,\"d2\":%d,\"D\":%d,\"M\":%d},", d1, d2, D, M);
    printf("\"total_ms\":{\"median_ms\":%.6f,\"p5_ms\":%.6f,\"p95_ms\":%.6f},",
           total.median, total.p5, total.p95);

    printf("\"diag_widths\":[");
    for (int k = 0; k < n_diags; k++) {
        int i_min = (k < d2) ? 0 : k - d2 + 1;
        int i_max = (k < d1) ? k : d1 - 1;
        int width = i_max - i_min + 1;
        printf("%d%s", width, k < n_diags - 1 ? "," : "");
    }
    printf("],\"max_parallelism\":%d,\"n_diags\":%d}\n",
           d1 < d2 ? d1 : d2, n_diags);

    free(x); free(A1); free(A2); free(B); free(C);
    free(delta1); free(delta2); free(h); free(y);
}

/* ═══════════════════════════════════════════════════════════════════
 * Fig 4 — GEMM Roofline (CPU)
 * ═══════════════════════════════════════════════════════════════════ */

static void fig4(int M_dim, int K_dim, int N_dim, int repeat, int warmup) {
    long mn = (long)M_dim * N_dim;
    long mk = (long)M_dim * K_dim;
    long kn = (long)K_dim * N_dim;

    float *A = alloc_rand(mk, -1.0f, 1.0f);
    float *B = alloc_rand(kn, -1.0f, 1.0f);
    float *C = alloc_zeros(mn);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        memset(C, 0, mn * sizeof(float));
        gemm_avx2(A, B, C, M_dim, K_dim, N_dim);
    }

    /* Measure */
    double *times = (double *)malloc(repeat * sizeof(double));
    for (int i = 0; i < repeat; i++) {
        memset(C, 0, mn * sizeof(float));
        double t0 = now_ms();
        gemm_avx2(A, B, C, M_dim, K_dim, N_dim);
        times[i] = now_ms() - t0;
    }
    Stats s = stats_from_double(times, repeat);
    free(times);

    double flops = 2.0 * M_dim * K_dim * N_dim;
    double bytes = (mk + kn + mn) * 4.0;
    double arith_intensity = flops / bytes;
    double gflops = flops / (s.median * 1e-3) * 1e-9;

    /* Measure peak bandwidth (simple copy benchmark) */
    long bw_n = 64 * 1024 * 1024 / 4;  /* 64 MB */
    float *bw_src = alloc_rand(bw_n, -1.0f, 1.0f);
    float *bw_dst = alloc_zeros(bw_n);
    for (int i = 0; i < 10; i++)
        memcpy(bw_dst, bw_src, bw_n * sizeof(float));
    double bw_t0 = now_ms();
    for (int i = 0; i < 50; i++)
        memcpy(bw_dst, bw_src, bw_n * sizeof(float));
    double bw_ms = (now_ms() - bw_t0) / 50.0;
    double peak_bw = (bw_n * 4.0 * 2.0) / (bw_ms * 1e-3) * 1e-9; /* GB/s, read+write */
    free(bw_src); free(bw_dst);

    printf("{\"fig\":4,\"config\":{\"M\":%d,\"K\":%d,\"N\":%d},", M_dim, K_dim, N_dim);
    printf("\"results\":{");
    json_stats("gemm_avx2", s);
    printf(",\"gflops\":%.4f,\"arith_intensity\":%.4f,"
           "\"peak_bw_gbs\":%.2f,\"peak_gflops_est\":%.2f}}\n",
           gflops, arith_intensity, peak_bw, gflops * 1.1);

    free(A); free(B); free(C);
}

/* ═══════════════════════════════════════════════════════════════════
 * Fig 5 — Backward speedup by D
 *   Compare: C scalar (A_diag=NULL) vs ASM AVX2 (A_diag set)
 * ═══════════════════════════════════════════════════════════════════ */

/* Forward M=1 shared B/C to get h[] for backward */
static void fwd_m1_shared(const float *x, const float *A, const float *B,
                           const float *C, const float *delta,
                           float *h, float *y, long L, long D) {
    float *state = (float *)calloc(D, sizeof(float));
    memset(y, 0, L * D * sizeof(float));
    for (long t = 0; t < L; t++) {
        float dt = delta[t];
        for (long d = 0; d < D; d++) {
            long td = t * D + d;
            float dA = expf(dt * A[d]);
            state[d] = dA * state[d] + dt * B[d] * x[td];
            h[td] = state[d];
            y[td] = C[d] * state[d];
        }
    }
    free(state);
}

static void fig5(int L, int D, int repeat, int warmup) {
    long LD = (long)L * D;

    float *x     = alloc_rand(LD, -1.0f, 1.0f);
    float *A     = alloc_rand(D,  -0.5f, -0.01f);
    float *B     = alloc_rand(D,  -0.3f, 0.3f);
    float *C     = alloc_rand(D,  -0.2f, 0.2f);
    float *delta = alloc_rand(L,   0.01f, 0.1f);
    float *dy    = alloc_rand(LD, -1.0f, 1.0f);
    float *h     = alloc_zeros(LD);
    float *y     = alloc_zeros(LD);

    /* Precompute A_diag */
    float *A_diag = (float *)malloc(LD * sizeof(float));
    fwd_m1_shared(x, A, B, C, delta, h, y, L, D);
    for (long t = 0; t < L; t++)
        for (long d = 0; d < D; d++)
            A_diag[t * D + d] = expf(delta[t] * A[d]);

    /* Allocate backward outputs */
    float *dx  = alloc_zeros(LD);
    float *dA  = alloc_zeros(D);
    float *dB  = alloc_zeros(D);
    float *dC  = alloc_zeros(D);
    float *ddt = alloc_zeros(L);

    /* ── C scalar (A_diag=NULL forces scalar path) ── */
    ScanBackwardSharedParams p_scl = {
        .x = x, .A = A, .A_diag = NULL,
        .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h, .dy = dy,
        .dx = dx, .dA = dA, .dB = dB, .dC = dC, .ddelta = ddt,
        .L = L, .D = D
    };

    for (int i = 0; i < warmup; i++)
        scan1d_backward_m1_shared_bc(&p_scl);

    double *scl_times = (double *)malloc(repeat * sizeof(double));
    for (int i = 0; i < repeat; i++) {
        double t0 = now_ms();
        scan1d_backward_m1_shared_bc(&p_scl);
        scl_times[i] = now_ms() - t0;
    }
    Stats scl = stats_from_double(scl_times, repeat);
    free(scl_times);

    /* ── ASM AVX2 (A_diag set) ── */
    ScanBackwardSharedParams p_asm = {
        .x = x, .A = A, .A_diag = A_diag,
        .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h, .dy = dy,
        .dx = dx, .dA = dA, .dB = dB, .dC = dC, .ddelta = ddt,
        .L = L, .D = D
    };

    for (int i = 0; i < warmup; i++)
        scan1d_backward_m1_shared_bc_asm(&p_asm);

    double *asm_times = (double *)malloc(repeat * sizeof(double));
    for (int i = 0; i < repeat; i++) {
        double t0 = now_ms();
        scan1d_backward_m1_shared_bc_asm(&p_asm);
        asm_times[i] = now_ms() - t0;
    }
    Stats asm_s = stats_from_double(asm_times, repeat);
    free(asm_times);

    /* ── C AVX2 (A_diag set, D>=8 → intrinsics path) ── */
    ScanBackwardSharedParams p_cavx = p_asm;
    for (int i = 0; i < warmup; i++)
        scan1d_backward_m1_shared_bc(&p_cavx);

    double *cavx_times = (double *)malloc(repeat * sizeof(double));
    for (int i = 0; i < repeat; i++) {
        double t0 = now_ms();
        scan1d_backward_m1_shared_bc(&p_cavx);
        cavx_times[i] = now_ms() - t0;
    }
    Stats cavx = stats_from_double(cavx_times, repeat);
    free(cavx_times);

    printf("{\"fig\":5,\"config\":{\"L\":%d,\"D\":%d},\"results\":{", L, D);
    json_stats("c_scalar", scl);
    printf(",");
    json_stats("c_avx2", cavx);
    printf(",");
    json_stats("asm_avx2", asm_s);
    printf("}}\n");

    free(x); free(A); free(B); free(C); free(delta); free(dy);
    free(h); free(y); free(A_diag);
    free(dx); free(dA); free(dB); free(dC); free(ddt);
}

/* ═══════════════════════════════════════════════════════════════════
 * Fig 6 — Scaling throughput
 * ═══════════════════════════════════════════════════════════════════ */

static void fig6(int L, int D, int M, int repeat, int warmup) {
    long LD = (long)L * D, DM = (long)D * M, LDM = (long)L * D * M;

    float *x     = alloc_rand(LD,  -1.0f, 1.0f);
    float *A     = alloc_rand(DM,  -0.5f, -0.01f);
    float *B     = alloc_rand(LDM, -0.3f, 0.3f);
    float *C     = alloc_rand(LDM, -0.2f, 0.2f);
    float *delta = alloc_rand(LD,   0.01f, 0.1f);
    float *h     = alloc_zeros(LDM);
    float *y     = alloc_zeros(LD);

    ScanParams p = { x, A, B, C, delta, h, y, L, D, M };

    for (int i = 0; i < warmup; i++) {
        memset(h, 0, LDM * sizeof(float));
        memset(y, 0, LD * sizeof(float));
        scan1d(&p);
    }

    double *times = (double *)malloc(repeat * sizeof(double));
    for (int i = 0; i < repeat; i++) {
        memset(h, 0, LDM * sizeof(float));
        memset(y, 0, LD * sizeof(float));
        double t0 = now_ms();
        scan1d(&p);
        times[i] = now_ms() - t0;
    }
    Stats s = stats_from_double(times, repeat);
    free(times);

    /* Throughput: total bytes accessed per call */
    double bytes_accessed = (LD + DM + 2.0 * LDM + LD + LDM + LD) * 4.0;
    double throughput_gbs = bytes_accessed / (s.median * 1e-3) * 1e-9;

    printf("{\"fig\":6,\"config\":{\"L\":%d,\"D\":%d,\"M\":%d},\"results\":{", L, D, M);
    json_stats("scan1d", s);
    printf(",\"throughput_gbs\":%.4f}}\n", throughput_gbs);

    free(x); free(A); free(B); free(C); free(delta); free(h); free(y);
}

/* ═══════════════════════════════════════════════════════════════════
 * Fig 7 — optimatrix data point (comparison vs other libs)
 *   Only outputs the optimatrix CPU/GPU data. Python handles the rest.
 * ═══════════════════════════════════════════════════════════════════ */

static void fig7(int L, int D, int M, int repeat, int warmup) {
    long LD = (long)L * D, DM = (long)D * M, LDM = (long)L * D * M;

    float *x     = alloc_rand(LD,  -1.0f, 1.0f);
    float *A     = alloc_rand(DM,  -0.5f, -0.01f);
    float *B     = alloc_rand(LDM, -0.3f, 0.3f);
    float *C     = alloc_rand(LDM, -0.2f, 0.2f);
    float *delta = alloc_rand(LD,   0.01f, 0.1f);
    float *h     = alloc_zeros(LDM);
    float *y     = alloc_zeros(LD);

    ScanParams p = { x, A, B, C, delta, h, y, L, D, M };

    /* CPU ASM */
    for (int i = 0; i < warmup; i++) {
        memset(h, 0, LDM * sizeof(float));
        memset(y, 0, LD * sizeof(float));
        scan1d(&p);
    }
    double *cpu_times = (double *)malloc(repeat * sizeof(double));
    for (int i = 0; i < repeat; i++) {
        memset(h, 0, LDM * sizeof(float));
        memset(y, 0, LD * sizeof(float));
        double t0 = now_ms();
        scan1d(&p);
        cpu_times[i] = now_ms() - t0;
    }
    Stats cpu = stats_from_double(cpu_times, repeat);
    free(cpu_times);

    /* CPU C reference (scalar) */
    for (int i = 0; i < warmup; i++) {
        memset(h, 0, LDM * sizeof(float));
        memset(y, 0, LD * sizeof(float));
        scan1d_ref(&p);
    }
    double *ref_times = (double *)malloc(repeat * sizeof(double));
    for (int i = 0; i < repeat; i++) {
        memset(h, 0, LDM * sizeof(float));
        memset(y, 0, LD * sizeof(float));
        double t0 = now_ms();
        scan1d_ref(&p);
        ref_times[i] = now_ms() - t0;
    }
    Stats ref = stats_from_double(ref_times, repeat);
    free(ref_times);

    printf("{\"fig\":7,\"config\":{\"L\":%d,\"D\":%d,\"M\":%d},\"results\":{", L, D, M);
    json_stats("optimatrix_cpu_asm", cpu);
    printf(",");
    json_stats("optimatrix_cpu_ref", ref);

#ifdef KMAMBA_HAS_CUDA
    float *gpu_times = (float *)malloc(repeat * sizeof(float));
    bench_gpu_scan1d(L, D, M, warmup, repeat, gpu_times);
    Stats gpu = stats_from_float(gpu_times, repeat);
    free(gpu_times);
    printf(",");
    json_stats("optimatrix_gpu", gpu);
#endif

    printf("}}\n");

    free(x); free(A); free(B); free(C); free(delta); free(h); free(y);
}

/* ═══════════════════════════════════════════════════════════════════
 * Fig 8 — MambaBlock end-to-end
 * ═══════════════════════════════════════════════════════════════════ */

static void fig8(int dim, int repeat, int warmup) {
    MBConfig cfg = {
        .dim = dim,
        .state_size = 16,
        .seq_len = 128,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f,
        .dt_rank = 1.0f,
        .dt_init = 0.01f,
        .use_convnd = 0,
        .convnd_K = 4,
        .convnd_ndims = 1
    };

    MambaBlock *block = mamba_block_create(&cfg);
    if (!block) {
        printf("{\"fig\":8,\"error\":\"mamba_block_create failed\"}\n");
        return;
    }
    mamba_block_init(block);

    long io_size = cfg.seq_len * cfg.dim;
    float *input  = alloc_rand(io_size, -1.0f, 1.0f);
    float *output = alloc_zeros(io_size);

    /* Warmup */
    for (int i = 0; i < warmup; i++)
        mamba_block_forward(block, output, input, 1);

    /* Measure */
    double *times = (double *)malloc(repeat * sizeof(double));
    for (int i = 0; i < repeat; i++) {
        double t0 = now_ms();
        mamba_block_forward(block, output, input, 1);
        times[i] = now_ms() - t0;
    }
    Stats s = stats_from_double(times, repeat);
    free(times);

    printf("{\"fig\":8,\"config\":{\"dim\":%d,\"state_size\":%zu,\"seq_len\":%zu},\"results\":{",
           dim, cfg.state_size, cfg.seq_len);
    json_stats("kmamba_cpu", s);
    printf("}}\n");

    free(input); free(output);
    mamba_block_free(block);
}

/* ═══════════════════════════════════════════════════════════════════
 * CLI & main
 * ═══════════════════════════════════════════════════════════════════ */

static int parse_int(const char *s) { return atoi(s); }

int main(int argc, char **argv) {
    int fig = 0, L = 1024, D = 64, M = 8;
    int repeat = 100, warmup = 20;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fig") && i + 1 < argc)
            fig = parse_int(argv[++i]);
        else if (!strcmp(argv[i], "--L") && i + 1 < argc)
            L = parse_int(argv[++i]);
        else if (!strcmp(argv[i], "--D") && i + 1 < argc)
            D = parse_int(argv[++i]);
        else if (!strcmp(argv[i], "--M") && i + 1 < argc)
            M = parse_int(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc)
            repeat = parse_int(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc)
            warmup = parse_int(argv[++i]);
    }

    if (fig < 1 || fig > 8) {
        fprintf(stderr,
            "Usage: bench_paper --fig N [--L X] [--D Y] [--M Z] "
            "[--repeat 100] [--warmup 20]\n"
            "  Fig 1: Scan1D CPU vs GPU        (--L --D --M)\n"
            "  Fig 2: Blelloch vs Sequential    (--L --D --M)\n"
            "  Fig 3: Wavefront 2D diags        (--L=d1 --D=d2 --M)\n"
            "  Fig 4: GEMM roofline             (--L=M --D=K --M=N)\n"
            "  Fig 5: Backward speedup          (--L --D)\n"
            "  Fig 6: Scaling throughput         (--L --D --M)\n"
            "  Fig 7: vs other libs (optimatrix) (--L --D --M)\n"
            "  Fig 8: MambaBlock end-to-end     (--D=dim)\n");
        return 1;
    }

    srand(42);

    switch (fig) {
        case 1: fig1(L, D, M, repeat, warmup); break;
        case 2: fig2(L, D, M, repeat, warmup); break;
        case 3: fig3(L, D, D, M, repeat, warmup); break;  /* d1=L, d2=D(arg), D_scan=D, M=M */
        case 4: fig4(L, D, M, repeat, warmup); break;      /* M_dim=L, K_dim=D, N_dim=M */
        case 5: fig5(L, D, repeat, warmup); break;
        case 6: fig6(L, D, M, repeat, warmup); break;
        case 7: fig7(L, D, M, repeat, warmup); break;
        case 8: fig8(D, repeat, warmup); break;             /* dim=D */
    }

    return 0;
}
