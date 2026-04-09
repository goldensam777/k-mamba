/*
 * convnd.cu — Convolution ND CUDA avec wavefront CPU
 *
 * Architecture: La topologie wavefront est générée côté host (CPU)
 * via km_wavefront_plan_create() et réutilisée pour orchestrer
 * les kernels CUDA niveau par niveau.
 *
 * Chaque niveau wavefront contient des positions spatialement
 * indépendantes qui peuvent être traitées en parallèle par le GPU.
 *
 * Forward: y[out] = sum_{k in kernel} x[in + offset(k)] * kernel[k] + bias
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#include "convnd.h"
#include "wavefront_plan.h"
#include "km_topology.h"

/* ── CUDA error checking ────────────────────────────────────── */
#define CONVND_CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        return -1; \
    } \
} while(0)

/* ── Host helpers ──────────────────────────────────────────── */

static long convnd_power_long(long base, long exp) {
    long out = 1;
    for (long i = 0; i < exp; i++) out *= base;
    return out;
}

static int convnd_validate_params(const ConvNDParams *p) {
    if (!p || !p->input || !p->kernel || !p->output ||
        !p->dims || p->ndims <= 0 || p->D <= 0 || p->K <= 0) {
        return 0;
    }
    /* Check dims are valid */
    for (long i = 0; i < p->ndims; i++) {
        if (p->dims[i] <= 0) return 0;
    }
    return 1;
}

static void convnd_unravel_index(long linear, const long *dims, long ndims, long *coords) {
    for (long axis = ndims; axis-- > 0;) {
        coords[axis] = linear % dims[axis];
        linear /= dims[axis];
    }
}

static void convnd_make_row_major_strides(const long *dims, long ndims, long *strides) {
    long stride = 1;
    for (long axis = ndims; axis-- > 0;) {
        strides[axis] = stride;
        stride *= dims[axis];
    }
}

/* ── Device helpers ────────────────────────────────────────── */

__device__ static void d_unravel_index(long linear, const long *dims, long ndims,
                                        long *strides, long *coords) {
    for (long axis = ndims; axis-- > 0;) {
        coords[axis] = linear / strides[axis];
        linear %= strides[axis];
    }
}

/* ── Forward kernel ─────────────────────────────────────────── */
/*
 * Chaque thread traite un (point_ordinal, d) pair:
 *   - point_ordinal: position dans le niveau wavefront courant
 *   - d: canal/feature
 *
 * La convolution ND est calculée en parcourant le noyau K^ndims.
 */
__global__ void convnd_level_kernel(
    const float *input,         /* [total_spatial * D] */
    const float *kernel,        /* [kernel_volume * D] */
    const float *bias,          /* [D] or NULL */
    float *output,              /* [total_spatial * D] */
    const long *ordered_offsets,/* offsets spatiaux groupés par niveau */
    long level_start,           /* début du niveau dans ordered_offsets */
    long level_size,            /* nombre de points dans ce niveau */
    const long *dims,           /* [ndims] */
    long ndims,
    const long *strides,        /* [ndims] row-major strides */
    long D,
    long K,
    long kernel_volume)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total_threads = level_size * D;

    if (tid >= total_threads) return;

    int d = (int)(tid % D);
    long point_ordinal = tid / D;
    long slot = level_start + point_ordinal;
    long out_linear = ordered_offsets[slot];

    /* Unravel output coordinates */
    long out_coords[KMAMBA_MAX_NDIMS];
    d_unravel_index(out_linear, dims, ndims, strides, out_coords);

    float sum = (bias) ? bias[d] : 0.0f;

    /* Iterate over kernel elements */
    for (long k_linear = 0; k_linear < kernel_volume; k_linear++) {
        long tmp = k_linear;
        long src_linear = 0;
        int valid = 1;

        /* Compute source position for this kernel element */
        for (long axis = ndims; axis-- > 0;) {
            long k_axis = tmp % K;
            tmp /= K;

            /* Causal/valid convolution: src = out - (K - 1 - k) */
            long src_coord = out_coords[axis] - (K - 1 - k_axis);
            if (src_coord < 0 || src_coord >= dims[axis]) {
                valid = 0;
                break;
            }
            src_linear += src_coord * strides[axis];
        }

        if (!valid) continue;

        sum += input[src_linear * D + d] * kernel[k_linear * D + d];
    }

    output[out_linear * D + d] = sum;
}

/* ── Backward kernel (gradients w.r.t. input) ───────────────── */
__global__ void convnd_bwd_input_kernel(
    const float *dy,            /* [total_spatial * D] gradient w.r.t. output */
    const float *kernel,        /* [kernel_volume * D] */
    float *dinput,              /* [total_spatial * D] gradient w.r.t. input (accumulate) */
    const long *ordered_offsets,
    long level_start,
    long level_size,
    const long *dims,
    long ndims,
    const long *strides,
    long D,
    long K,
    long kernel_volume)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total_threads = level_size * D;

    if (tid >= total_threads) return;

    int d = (int)(tid % D);
    long point_ordinal = tid / D;
    long slot = level_start + point_ordinal;
    long out_linear = ordered_offsets[slot];

    long out_coords[KMAMBA_MAX_NDIMS];
    d_unravel_index(out_linear, dims, ndims, strides, out_coords);

    float grad_out = dy[out_linear * D + d];

    /* Propagate gradient to input positions */
    for (long k_linear = 0; k_linear < kernel_volume; k_linear++) {
        long tmp = k_linear;
        long src_linear = 0;
        int valid = 1;

        for (long axis = ndims; axis-- > 0;) {
            long k_axis = tmp % K;
            tmp /= K;

            long src_coord = out_coords[axis] - (K - 1 - k_axis);
            if (src_coord < 0 || src_coord >= dims[axis]) {
                valid = 0;
                break;
            }
            src_linear += src_coord * strides[axis];
        }

        if (!valid) continue;

        /* Atomic add because multiple output positions may write to same input */
        float grad = grad_out * kernel[k_linear * D + d];
        atomicAdd(&dinput[src_linear * D + d], grad);
    }
}

/* ── Backward kernel (gradients w.r.t. kernel) ─────────────── */
__global__ void convnd_bwd_kernel_kernel(
    const float *input,
    const float *dy,
    float *dkernel,             /* [kernel_volume * D] accumulate */
    const long *ordered_offsets,
    long level_start,
    long level_size,
    const long *dims,
    long ndims,
    const long *strides,
    long D,
    long K,
    long kernel_volume)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total_threads = level_size * D;

    if (tid >= total_threads) return;

    int d = (int)(tid % D);
    long point_ordinal = tid / D;
    long slot = level_start + point_ordinal;
    long out_linear = ordered_offsets[slot];

    long out_coords[KMAMBA_MAX_NDIMS];
    d_unravel_index(out_linear, dims, ndims, strides, out_coords);

    float grad_out = dy[out_linear * D + d];

    for (long k_linear = 0; k_linear < kernel_volume; k_linear++) {
        long tmp = k_linear;
        long src_linear = 0;
        int valid = 1;

        for (long axis = ndims; axis-- > 0;) {
            long k_axis = tmp % K;
            tmp /= K;

            long src_coord = out_coords[axis] - (K - 1 - k_axis);
            if (src_coord < 0 || src_coord >= dims[axis]) {
                valid = 0;
                break;
            }
            src_linear += src_coord * strides[axis];
        }

        if (!valid) continue;

        float grad = grad_out * input[src_linear * D + d];
        atomicAdd(&dkernel[k_linear * D + d], grad);
    }
}

/* ── API: Forward pass ─────────────────────────────────────── */
int om_convnd_forward(ConvNDParams *p) {
    KMWavefrontPlan *plan = NULL;
    long *d_ordered_offsets = NULL;
    long *d_dims = NULL;
    long *d_strides = NULL;
    long *h_strides = NULL;
    int rc = -1;

    if (!convnd_validate_params(p)) return -1;

    /* Create wavefront plan on host */
    plan = km_wavefront_plan_create(p->dims, p->ndims);
    if (!plan) return -1;

    /* Validate plan matches dims */
    if (!km_wavefront_plan_matches_dims(plan, p->dims, p->ndims)) {
        km_wavefront_plan_free(plan);
        return -1;
    }

    /* Compute row-major strides on host */
    h_strides = (long *)malloc((size_t)p->ndims * sizeof(long));
    if (!h_strides) goto cleanup;
    convnd_make_row_major_strides(p->dims, p->ndims, h_strides);

    /* Transfer plan data to device */
    CONVND_CUDA_CHECK(cudaMalloc(&d_ordered_offsets,
                                 (size_t)plan->total_points * sizeof(long)));
    CONVND_CUDA_CHECK(cudaMalloc(&d_dims, (size_t)p->ndims * sizeof(long)));
    CONVND_CUDA_CHECK(cudaMalloc(&d_strides, (size_t)p->ndims * sizeof(long)));

    CONVND_CUDA_CHECK(cudaMemcpy(d_ordered_offsets, plan->level_offsets,
                                 (size_t)plan->total_points * sizeof(long),
                                 cudaMemcpyHostToDevice));
    CONVND_CUDA_CHECK(cudaMemcpy(d_dims, p->dims,
                                 (size_t)p->ndims * sizeof(long),
                                 cudaMemcpyHostToDevice));
    CONVND_CUDA_CHECK(cudaMemcpy(d_strides, h_strides,
                                 (size_t)p->ndims * sizeof(long),
                                 cudaMemcpyHostToDevice));

    long kernel_volume = convnd_power_long(p->K, p->ndims);

    /* Process each level */
    for (long level = 0; level <= plan->max_level; level++) {
        long level_size = km_wavefront_plan_level_size(plan, level);
        if (level_size <= 0) continue;

        long level_start = plan->level_starts[level];
        long total_threads = level_size * p->D;
        int blocks = (int)((total_threads + 255) / 256);

        convnd_level_kernel<<<blocks, 256>>>(
            p->input, p->kernel, p->bias, p->output,
            d_ordered_offsets, level_start, level_size,
            d_dims, p->ndims, d_strides,
            p->D, p->K, kernel_volume);

        CONVND_CUDA_CHECK(cudaGetLastError());
    }

    CONVND_CUDA_CHECK(cudaDeviceSynchronize());
    rc = 0;

cleanup:
    cudaFree(d_ordered_offsets);
    cudaFree(d_dims);
    cudaFree(d_strides);
    free(h_strides);
    km_wavefront_plan_free(plan);
    return rc;
}

/* ── API: Backward pass ────────────────────────────────────── */
int om_convnd_backward(ConvNDParams *p) {
    KMWavefrontPlan *plan = NULL;
    long *d_ordered_offsets = NULL;
    long *d_dims = NULL;
    long *d_strides = NULL;
    long *h_strides = NULL;
    long total_spatial;
    long kernel_volume;
    int rc = -1;

    if (!p || !p->input || !p->kernel || !p->dy ||
        !p->dinput || !p->dkernel ||
        !p->dims || p->ndims <= 0 || p->D <= 0 || p->K <= 0) {
        return -1;
    }

    /* Compute total spatial points */
    total_spatial = 1;
    for (long i = 0; i < p->ndims; i++) total_spatial *= p->dims[i];
    kernel_volume = convnd_power_long(p->K, p->ndims);

    /* Create plan */
    plan = km_wavefront_plan_create(p->dims, p->ndims);
    if (!plan) return -1;

    /* Init gradient buffers */
    CONVND_CUDA_CHECK(cudaMemset(p->dinput, 0,
                                 (size_t)(total_spatial * p->D) * sizeof(float)));
    CONVND_CUDA_CHECK(cudaMemset(p->dkernel, 0,
                                 (size_t)(kernel_volume * p->D) * sizeof(float)));
    if (p->dbias) {
        CONVND_CUDA_CHECK(cudaMemset(p->dbias, 0,
                                     (size_t)p->D * sizeof(float)));
    }

    /* Strides */
    h_strides = (long *)malloc((size_t)p->ndims * sizeof(long));
    if (!h_strides) goto cleanup;
    convnd_make_row_major_strides(p->dims, p->ndims, h_strides);

    /* Transfer to device */
    CONVND_CUDA_CHECK(cudaMalloc(&d_ordered_offsets,
                                 (size_t)plan->total_points * sizeof(long)));
    CONVND_CUDA_CHECK(cudaMalloc(&d_dims, (size_t)p->ndims * sizeof(long)));
    CONVND_CUDA_CHECK(cudaMalloc(&d_strides, (size_t)p->ndims * sizeof(long)));

    CONVND_CUDA_CHECK(cudaMemcpy(d_ordered_offsets, plan->level_offsets,
                                 (size_t)plan->total_points * sizeof(long),
                                 cudaMemcpyHostToDevice));
    CONVND_CUDA_CHECK(cudaMemcpy(d_dims, p->dims,
                                 (size_t)p->ndims * sizeof(long),
                                 cudaMemcpyHostToDevice));
    CONVND_CUDA_CHECK(cudaMemcpy(d_strides, h_strides,
                                 (size_t)p->ndims * sizeof(long),
                                 cudaMemcpyHostToDevice));

    /* Backward through levels (reverse order) */
    for (long level = plan->max_level; level >= 0; level--) {
        long level_size = km_wavefront_plan_level_size(plan, level);
        if (level_size <= 0) continue;

        long level_start = plan->level_starts[level];
        long total_threads = level_size * p->D;
        int blocks = (int)((total_threads + 255) / 256);

        /* Gradient w.r.t. input */
        convnd_bwd_input_kernel<<<blocks, 256>>>(
            p->dy, p->kernel, p->dinput,
            d_ordered_offsets, level_start, level_size,
            d_dims, p->ndims, d_strides,
            p->D, p->K, kernel_volume);

        /* Gradient w.r.t. kernel */
        convnd_bwd_kernel_kernel<<<blocks, 256>>>(
            p->input, p->dy, p->dkernel,
            d_ordered_offsets, level_start, level_size,
            d_dims, p->ndims, d_strides,
            p->D, p->K, kernel_volume);

        /* Gradient w.r.t. bias (simple reduction) */
        if (p->dbias) {
            /* TODO: Optimize with shared memory reduction */
            for (long point = 0; point < level_size; point++) {
                long slot = level_start + point;
                long out_linear = plan->level_offsets[slot];
                for (int d = 0; d < p->D; d++) {
                    p->dbias[d] += p->dy[out_linear * p->D + d];
                }
            }
        }

        CONVND_CUDA_CHECK(cudaGetLastError());
    }

    CONVND_CUDA_CHECK(cudaDeviceSynchronize());
    rc = 0;

cleanup:
    cudaFree(d_ordered_offsets);
    cudaFree(d_dims);
    cudaFree(d_strides);
    free(h_strides);
    km_wavefront_plan_free(plan);
    return rc;
}
