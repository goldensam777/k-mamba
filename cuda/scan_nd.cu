#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "scan_nd.h"

#define SCANND_CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        return -1; \
    } \
} while (0)

static int scannd_dims_valid_host(const long *dims, long ndims) {
    if (!dims || ndims <= 0) return 0;
    for (long axis = 0; axis < ndims; axis++) {
        if (dims[axis] <= 0) return 0;
    }
    return 1;
}

static int scannd_safe_add_long(long a, long b, long *out) {
    if (!out) return 0;
    if ((b > 0 && a > LONG_MAX - b) || (b < 0 && a < LONG_MIN - b)) return 0;
    *out = a + b;
    return 1;
}

static int scannd_safe_mul_long(long a, long b, long *out) {
    if (!out || a < 0 || b < 0) return 0;
    if (a == 0 || b == 0) {
        *out = 0;
        return 1;
    }
    if (a > LONG_MAX / b) return 0;
    *out = a * b;
    return 1;
}

static long scannd_total_points_host(const long *dims, long ndims) {
    long total = 1;

    if (!scannd_dims_valid_host(dims, ndims)) return -1;
    for (long axis = 0; axis < ndims; axis++) {
        if (!scannd_safe_mul_long(total, dims[axis], &total)) return -1;
    }
    return total;
}

static long scannd_max_level_host(const long *dims, long ndims) {
    long max_level = 0;

    if (!scannd_dims_valid_host(dims, ndims)) return -1;
    for (long axis = 0; axis < ndims; axis++) {
        if (!scannd_safe_add_long(max_level, dims[axis] - 1, &max_level)) return -1;
    }
    return max_level;
}

static int scannd_build_strides_host(const long *dims, long ndims, long *strides) {
    long stride = 1;

    if (!dims || !strides || ndims <= 0) return 0;

    for (long axis = ndims - 1; axis >= 0; axis--) {
        strides[axis] = stride;
        if (!scannd_safe_mul_long(stride, dims[axis], &stride)) return 0;
    }
    return 1;
}

static int scannd_build_suffix_caps(const long *dims, long ndims, long *suffix_caps) {
    long suffix = 0;

    if (!dims || !suffix_caps || ndims <= 0) return 0;

    suffix_caps[ndims] = 0;
    for (long axis = ndims - 1; axis >= 0; axis--) {
        long cap = dims[axis] - 1;
        if (cap < 0 || !scannd_safe_add_long(suffix, cap, &suffix)) return 0;
        suffix_caps[axis] = suffix;
    }
    return 1;
}

static long scannd_max_long(long a, long b) {
    return (a > b) ? a : b;
}

static long scannd_min_long(long a, long b) {
    return (a < b) ? a : b;
}

static long scannd_level_size_recursive(const long *dims,
                                        const long *suffix_caps,
                                        long ndims,
                                        long axis,
                                        long remaining) {
    long lo;
    long hi;
    long count = 0;

    if (axis == ndims - 1) {
        return (remaining >= 0 && remaining < dims[axis]) ? 1 : 0;
    }

    lo = scannd_max_long(0, remaining - suffix_caps[axis + 1]);
    hi = scannd_min_long(dims[axis] - 1, remaining);

    for (long value = lo; value <= hi; value++) {
        long sub = scannd_level_size_recursive(dims, suffix_caps, ndims, axis + 1, remaining - value);
        if (sub < 0 || !scannd_safe_add_long(count, sub, &count)) return -1;
    }

    return count;
}

typedef struct {
    const long *dims;
    const long *strides;
    const long *level_offsets;
    const long *suffix_caps;
    long *ordered_offsets;
    long *prev_offsets;
    long *idx;
    long ndims;
    long level;
    int failed;
} ScanNDEmitCtx;

static void scannd_emit_level_recursive(ScanNDEmitCtx *ctx, long axis, long remaining) {
    long lo;
    long hi;

    if (!ctx || ctx->failed) return;

    if (axis == ctx->ndims - 1) {
        if (remaining >= 0 && remaining < ctx->dims[axis]) {
            long ordinal = 0;
            long offset = 0;
            long slot;

            ctx->idx[axis] = remaining;
            for (long d = 0; d < ctx->ndims; d++) {
                offset += ctx->idx[d] * ctx->strides[d];
            }

            slot = ctx->level_offsets[ctx->level];
            while (slot + ordinal < ctx->level_offsets[ctx->level + 1] &&
                   ctx->ordered_offsets[slot + ordinal] != -1) {
                ordinal++;
            }

            if (slot + ordinal >= ctx->level_offsets[ctx->level + 1]) {
                ctx->failed = 1;
                return;
            }

            slot += ordinal;
            ctx->ordered_offsets[slot] = offset;
            for (long d = 0; d < ctx->ndims; d++) {
                ctx->prev_offsets[slot * ctx->ndims + d] =
                    (ctx->idx[d] > 0) ? (offset - ctx->strides[d]) : -1;
            }
        }
        return;
    }

    lo = scannd_max_long(0, remaining - ctx->suffix_caps[axis + 1]);
    hi = scannd_min_long(ctx->dims[axis] - 1, remaining);

    for (long value = lo; value <= hi; value++) {
        ctx->idx[axis] = value;
        scannd_emit_level_recursive(ctx, axis + 1, remaining - value);
        if (ctx->failed) return;
    }
}

static int scannd_build_schedule(const long *dims, long ndims,
                                 long **level_offsets_out,
                                 long **ordered_offsets_out,
                                 long **prev_offsets_out,
                                 long *total_points_out,
                                 long *max_level_out) {
    long total_points;
    long max_level;
    long *strides = NULL;
    long *suffix_caps = NULL;
    long *level_offsets = NULL;
    long *ordered_offsets = NULL;
    long *prev_offsets = NULL;
    long *idx = NULL;
    int ok = 0;

    if (!level_offsets_out || !ordered_offsets_out || !prev_offsets_out ||
        !total_points_out || !max_level_out) {
        return 0;
    }

    total_points = scannd_total_points_host(dims, ndims);
    max_level = scannd_max_level_host(dims, ndims);
    if (total_points <= 0 || max_level < 0) return 0;

    strides = (long *)malloc((size_t)ndims * sizeof(long));
    suffix_caps = (long *)malloc((size_t)(ndims + 1) * sizeof(long));
    level_offsets = (long *)malloc((size_t)(max_level + 2) * sizeof(long));
    ordered_offsets = (long *)malloc((size_t)total_points * sizeof(long));
    prev_offsets = (long *)malloc((size_t)(total_points * ndims) * sizeof(long));
    idx = (long *)malloc((size_t)ndims * sizeof(long));

    if (!strides || !suffix_caps || !level_offsets || !ordered_offsets || !prev_offsets || !idx) goto cleanup;
    if (!scannd_build_strides_host(dims, ndims, strides)) goto cleanup;
    if (!scannd_build_suffix_caps(dims, ndims, suffix_caps)) goto cleanup;

    for (long i = 0; i < total_points; i++) ordered_offsets[i] = -1;

    level_offsets[0] = 0;
    for (long level = 0; level <= max_level; level++) {
        long level_size = scannd_level_size_recursive(dims, suffix_caps, ndims, 0, level);
        if (level_size < 0 || !scannd_safe_add_long(level_offsets[level], level_size, &level_offsets[level + 1])) {
            goto cleanup;
        }
    }

    for (long level = 0; level <= max_level; level++) {
        ScanNDEmitCtx ctx;
        ctx.dims = dims;
        ctx.strides = strides;
        ctx.level_offsets = level_offsets;
        ctx.suffix_caps = suffix_caps;
        ctx.ordered_offsets = ordered_offsets;
        ctx.prev_offsets = prev_offsets;
        ctx.idx = idx;
        ctx.ndims = ndims;
        ctx.level = level;
        ctx.failed = 0;

        scannd_emit_level_recursive(&ctx, 0, level);
        if (ctx.failed) goto cleanup;
    }

    *level_offsets_out = level_offsets;
    *ordered_offsets_out = ordered_offsets;
    *prev_offsets_out = prev_offsets;
    *total_points_out = total_points;
    *max_level_out = max_level;
    level_offsets = NULL;
    ordered_offsets = NULL;
    prev_offsets = NULL;
    ok = 1;

cleanup:
    free(strides);
    free(suffix_caps);
    free(idx);
    free(level_offsets);
    free(ordered_offsets);
    free(prev_offsets);
    return ok;
}

__global__ void scannd_level_kernel(const float *x,
                                    const float *A,
                                    const float *B,
                                    const float *delta,
                                    const long *ordered_offsets,
                                    const long *prev_offsets,
                                    long level_start,
                                    long level_size,
                                    long total_points,
                                    int ndims,
                                    int D,
                                    int M,
                                    float *h) {
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total_threads = level_size * (long)D * M;

    if (tid >= total_threads) return;

    int m = (int)(tid % M);
    long tmp = tid / M;
    int d = (int)(tmp % D);
    long point_ordinal = tmp / D;
    long slot = level_start + point_ordinal;
    long offset = ordered_offsets[slot];
    long dm = (long)d * M + m;
    long pdm = offset * ((long)D * M) + dm;
    float x_val = x[offset * D + d];
    float dt_bar = 0.0f;
    float h_new;

    for (int axis = 0; axis < ndims; axis++) {
        dt_bar += delta[((long)axis * total_points + offset) * D + d];
    }
    dt_bar /= (float)ndims;

    h_new = dt_bar * B[pdm] * x_val;
    for (int axis = 0; axis < ndims; axis++) {
        long prev_offset = prev_offsets[slot * ndims + axis];
        if (prev_offset >= 0) {
            float dt_axis = delta[((long)axis * total_points + offset) * D + d];
            float a_val = A[((long)axis * D + d) * M + m];
            long prev_pdm = prev_offset * ((long)D * M) + dm;
            h_new += expf(dt_axis * a_val) * h[prev_pdm];
        }
    }

    h[pdm] = h_new;
}

__global__ void scannd_output_kernel(const float *h,
                                     const float *C,
                                     long total_points,
                                     int D,
                                     int M,
                                     float *y) {
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total_threads = total_points * (long)D;

    if (tid >= total_threads) return;

    long offset = tid / D;
    int d = (int)(tid % D);
    long base = offset * ((long)D * M) + (long)d * M;
    float acc = 0.0f;

    for (int m = 0; m < M; m++) {
        acc += C[base + m] * h[base + m];
    }
    y[offset * D + d] = acc;
}

int om_scannd_forward(ScanNDParams *p) {
    long *h_level_offsets = NULL;
    long *h_ordered_offsets = NULL;
    long *h_prev_offsets = NULL;
    long total_points;
    long max_level;
    long *d_ordered_offsets = NULL;
    long *d_prev_offsets = NULL;
    int rc = -1;

    if (!p || !scannd_dims_valid_host(p->dims, p->ndims) || p->D <= 0 || p->M <= 0 ||
        !p->x || !p->A || !p->B || !p->C || !p->delta || !p->h || !p->y) {
        return -1;
    }

    if (!scannd_build_schedule(p->dims, p->ndims,
                               &h_level_offsets, &h_ordered_offsets, &h_prev_offsets,
                               &total_points, &max_level)) {
        return -1;
    }

    SCANND_CUDA_CHECK(cudaMalloc(&d_ordered_offsets, (size_t)total_points * sizeof(long)));
    SCANND_CUDA_CHECK(cudaMalloc(&d_prev_offsets, (size_t)(total_points * p->ndims) * sizeof(long)));
    SCANND_CUDA_CHECK(cudaMemcpy(d_ordered_offsets, h_ordered_offsets,
                                 (size_t)total_points * sizeof(long),
                                 cudaMemcpyHostToDevice));
    SCANND_CUDA_CHECK(cudaMemcpy(d_prev_offsets, h_prev_offsets,
                                 (size_t)(total_points * p->ndims) * sizeof(long),
                                 cudaMemcpyHostToDevice));

    for (long level = 0; level <= max_level; level++) {
        long level_start = h_level_offsets[level];
        long level_size = h_level_offsets[level + 1] - level_start;
        long total_threads = level_size * p->D * p->M;
        int blocks = (int)((total_threads + 255) / 256);

        if (level_size <= 0) continue;

        scannd_level_kernel<<<blocks, 256>>>(
            p->x, p->A, p->B, p->delta,
            d_ordered_offsets, d_prev_offsets,
            level_start, level_size, total_points,
            (int)p->ndims, (int)p->D, (int)p->M, p->h);
        SCANND_CUDA_CHECK(cudaGetLastError());
    }

    {
        long total_threads = total_points * p->D;
        int blocks = (int)((total_threads + 255) / 256);
        scannd_output_kernel<<<blocks, 256>>>(
            p->h, p->C, total_points, (int)p->D, (int)p->M, p->y);
        SCANND_CUDA_CHECK(cudaGetLastError());
    }

    SCANND_CUDA_CHECK(cudaDeviceSynchronize());
    rc = 0;

    cudaFree(d_ordered_offsets);
    cudaFree(d_prev_offsets);
    free(h_level_offsets);
    free(h_ordered_offsets);
    free(h_prev_offsets);
    return rc;
}
