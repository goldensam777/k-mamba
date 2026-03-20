#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mamba_scan_cuda.h"
#include "scan_nd.h"

#define EPSILON_ND 1e-4f

static void fill_random(float *data, int n, float lo, float hi) {
    for (int i = 0; i < n; i++) {
        data[i] = lo + (hi - lo) * ((float)rand() / RAND_MAX);
    }
}

static int compare_array(const char *name, const float *ref, const float *gpu, int n, float eps) {
    float max_diff = 0.0f;
    int worst = -1;

    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - gpu[i]);
        if (diff > eps && diff > max_diff) {
            max_diff = diff;
            worst = i;
        }
    }

    if (worst >= 0) {
        printf("  FAIL %s: worst[%d] ref=%.6f gpu=%.6f diff=%.6f\n",
               name, worst, ref[worst], gpu[worst], max_diff);
        return 0;
    }
    return 1;
}

static int run_case(const long *dims, long ndims, int D, int M, const char *label) {
    long total_points = 1;
    int sz_x;
    int sz_A;
    int sz_delta;
    int sz_h;
    float *x = NULL;
    float *A = NULL;
    float *B = NULL;
    float *C = NULL;
    float *delta = NULL;
    float *h_ref = NULL;
    float *y_ref = NULL;
    float *h_gpu = NULL;
    float *y_gpu = NULL;
    float *d_x = NULL;
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    float *d_delta = NULL;
    float *d_h = NULL;
    float *d_y = NULL;
    int ok = 1;
    ScanNDParams p_ref;
    ScanNDParams p_gpu;

    for (long i = 0; i < ndims; i++) total_points *= dims[i];

    sz_x = (int)(total_points * D);
    sz_A = (int)(ndims * D * M);
    sz_delta = (int)(ndims * total_points * D);
    sz_h = (int)(total_points * D * M);

    x = (float *)malloc((size_t)sz_x * sizeof(float));
    A = (float *)malloc((size_t)sz_A * sizeof(float));
    B = (float *)malloc((size_t)sz_h * sizeof(float));
    C = (float *)malloc((size_t)sz_h * sizeof(float));
    delta = (float *)malloc((size_t)sz_delta * sizeof(float));
    h_ref = (float *)calloc((size_t)sz_h, sizeof(float));
    y_ref = (float *)calloc((size_t)sz_x, sizeof(float));
    h_gpu = (float *)malloc((size_t)sz_h * sizeof(float));
    y_gpu = (float *)malloc((size_t)sz_x * sizeof(float));

    fill_random(x, sz_x, -0.5f, 0.5f);
    fill_random(A, sz_A, -0.4f, -0.05f);
    fill_random(B, sz_h, -0.2f, 0.2f);
    fill_random(C, sz_h, -0.2f, 0.2f);
    fill_random(delta, sz_delta, 0.01f, 0.1f);

    p_ref.dims = dims;
    p_ref.ndims = ndims;
    p_ref.D = D;
    p_ref.M = M;
    p_ref.x = x;
    p_ref.A = A;
    p_ref.B = B;
    p_ref.C = C;
    p_ref.delta = delta;
    p_ref.h = h_ref;
    p_ref.y = y_ref;

    if (scannd_ref(&p_ref) != 0) {
        printf("  FAIL %s: scannd_ref failed\n", label);
        ok = 0;
        goto cleanup;
    }

    cudaMalloc(&d_x, (size_t)sz_x * sizeof(float));
    cudaMalloc(&d_A, (size_t)sz_A * sizeof(float));
    cudaMalloc(&d_B, (size_t)sz_h * sizeof(float));
    cudaMalloc(&d_C, (size_t)sz_h * sizeof(float));
    cudaMalloc(&d_delta, (size_t)sz_delta * sizeof(float));
    cudaMalloc(&d_h, (size_t)sz_h * sizeof(float));
    cudaMalloc(&d_y, (size_t)sz_x * sizeof(float));

    cudaMemcpy(d_x, x, (size_t)sz_x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, (size_t)sz_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (size_t)sz_h * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, (size_t)sz_h * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta, delta, (size_t)sz_delta * sizeof(float), cudaMemcpyHostToDevice);

    p_gpu.dims = dims;
    p_gpu.ndims = ndims;
    p_gpu.D = D;
    p_gpu.M = M;
    p_gpu.x = d_x;
    p_gpu.A = d_A;
    p_gpu.B = d_B;
    p_gpu.C = d_C;
    p_gpu.delta = d_delta;
    p_gpu.h = d_h;
    p_gpu.y = d_y;

    if (mamba_scannd_cuda_forward(&p_gpu) != 0) {
        printf("  FAIL %s: mamba_scannd_cuda_forward failed\n", label);
        ok = 0;
        goto cleanup;
    }

    cudaMemcpy(h_gpu, d_h, (size_t)sz_h * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_gpu, d_y, (size_t)sz_x * sizeof(float), cudaMemcpyDeviceToHost);

    ok &= compare_array("h", h_ref, h_gpu, sz_h, EPSILON_ND);
    ok &= compare_array("y", y_ref, y_gpu, sz_x, EPSILON_ND);

    if (ok) printf("  OK %s\n", label);

cleanup:
    cudaFree(d_x);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_delta);
    cudaFree(d_h);
    cudaFree(d_y);
    free(x);
    free(A);
    free(B);
    free(C);
    free(delta);
    free(h_ref);
    free(y_ref);
    free(h_gpu);
    free(y_gpu);
    return ok;
}

int main(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    int passed = 0;
    int total = 0;

    if (err != cudaSuccess || device_count <= 0) {
        printf("=== CUDA scan_nd test skipped: no CUDA device available ===\n");
        return 0;
    }

    srand(1234);
    printf("=== Tests scan_nd CUDA ===\n");

    total++; passed += run_case((const long[]){8}, 1, 3, 2, "1D");
    total++; passed += run_case((const long[]){3, 4}, 2, 2, 3, "2D");
    total++; passed += run_case((const long[]){2, 2, 3}, 3, 2, 2, "3D");

    printf("\n=== Result: %d/%d tests pass ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
