/*
 * bench_gpu.cu — GPU benchmark implementations with cudaEvent timing
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scan.h"
#include "bench_gpu.h"

/* ── Sequential scan kernel (copy from scan1d.cu for forced-seq timing) ── */

__global__ static void bench_seq_kernel(
    const float *x,   const float *A,
    const float *B,   const float *C,
    const float *dt,
    float *y,         float *h,
    int L, int D, int M)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= D * M) return;

    int m = tid % M;
    int d = tid / M;
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
        atomicAdd(&y[t_d], C[t_dm] * h_cur);
    }
}

/* ── Helpers ──────────────────────────────────────────────────────── */

static void fill_rand_host(float *p, int n, float lo, float hi)
{
    for (int i = 0; i < n; i++)
        p[i] = lo + (hi - lo) * ((float)rand() / RAND_MAX);
}

/* Allocate device buffer, fill with random host data, copy to device */
static float* gpu_alloc_rand(int n, float lo, float hi)
{
    float *h = (float*)malloc(n * sizeof(float));
    fill_rand_host(h, n, lo, hi);
    float *d;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    return d;
}

static float* gpu_alloc_zero(int n)
{
    float *d;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));
    return d;
}

/* ── bench_gpu_scan1d: auto path (Blelloch for L<=1024) ──────────── */

extern "C"
void bench_gpu_scan1d(int L, int D, int M,
                      int warmup, int repeat, float *out_ms)
{
    srand(42);
    int LD = L * D, DM = D * M, LDM = L * D * M;

    float *d_x  = gpu_alloc_rand(LD,  -1.0f, 1.0f);
    float *d_A  = gpu_alloc_rand(DM,  -0.5f, -0.01f);
    float *d_B  = gpu_alloc_rand(LDM, -0.3f, 0.3f);
    float *d_C  = gpu_alloc_rand(LDM, -0.2f, 0.2f);
    float *d_dt = gpu_alloc_rand(LD,   0.01f, 0.1f);
    float *d_y  = gpu_alloc_zero(LD);
    float *d_h  = gpu_alloc_zero(LDM);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        cudaMemset(d_y, 0, LD * sizeof(float));
        om_scan1d_forward(d_x, d_A, d_B, d_C, d_dt, d_y, d_h, L, D, M);
    }
    cudaDeviceSynchronize();

    /* Measure */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < repeat; i++) {
        cudaMemset(d_y, 0, LD * sizeof(float));
        cudaEventRecord(start);
        om_scan1d_forward(d_x, d_A, d_B, d_C, d_dt, d_y, d_h, L, D, M);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&out_ms[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_dt); cudaFree(d_y); cudaFree(d_h);
}

/* ── bench_gpu_scan1d_seq: forced sequential kernel ──────────────── */

extern "C"
void bench_gpu_scan1d_seq(int L, int D, int M,
                          int warmup, int repeat, float *out_ms)
{
    srand(42);
    int LD = L * D, DM = D * M, LDM = L * D * M;

    float *d_x  = gpu_alloc_rand(LD,  -1.0f, 1.0f);
    float *d_A  = gpu_alloc_rand(DM,  -0.5f, -0.01f);
    float *d_B  = gpu_alloc_rand(LDM, -0.3f, 0.3f);
    float *d_C  = gpu_alloc_rand(LDM, -0.2f, 0.2f);
    float *d_dt = gpu_alloc_rand(LD,   0.01f, 0.1f);
    float *d_y  = gpu_alloc_zero(LD);
    float *d_h  = gpu_alloc_zero(LDM);

    int blocks = (DM + 255) / 256;

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        cudaMemset(d_y, 0, LD * sizeof(float));
        bench_seq_kernel<<<blocks, 256>>>(
            d_x, d_A, d_B, d_C, d_dt, d_y, d_h, L, D, M);
        cudaDeviceSynchronize();
    }

    /* Measure */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < repeat; i++) {
        cudaMemset(d_y, 0, LD * sizeof(float));
        cudaEventRecord(start);
        bench_seq_kernel<<<blocks, 256>>>(
            d_x, d_A, d_B, d_C, d_dt, d_y, d_h, L, D, M);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&out_ms[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_dt); cudaFree(d_y); cudaFree(d_h);
}
