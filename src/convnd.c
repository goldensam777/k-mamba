/* ============================================================
 * convnd.c — Convolution ND native wavefront parallèle
 *
 * Architecture :
 *   Convolution ND dense avec ordonnancement wavefront.
 *   Noyau complet K^N (pas de séparabilité).
 *   Parallélisme intra-niveau via OpenMP (optionnel).
 *
 * Forward  : wavefront niveau par niveau, parallèle intra-niveau
 * Backward : wavefront inverse, accumulation gradients
 *
 * Unification :
 *   - Plus de distinction séparable/full
 *   - Une seule API convnd() avec wavefront natif
 *   - Parallélisme OpenMP optionnel (#ifdef _OPENMP)
 * ============================================================ */

#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#include "wavefront_plan.h"
#include "km_topology.h"
#include "convnd.h"
#include "kmamba_cuda_utils.h"

/* ============================================================
 * Helpers
 * ============================================================ */

static long product(const long *arr, long n) {
    long p = 1;
    for (long i = 0; i < n; i++) p *= arr[i];
    return p;
}

static long convnd_power_long(long base, long exp) {
    long out = 1;
    for (long i = 0; i < exp; i++) out *= base;
    return out;
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

long convnd_kernel_volume(long ndims, long K) {
    if (ndims <= 0 || K <= 0) return 0;
    return convnd_power_long(K, ndims);
}

/* ============================================================
 * Forward Wavefront - Parallélisé intra-niveau
 * ============================================================ */

void convnd_forward_wavefront(ConvNDParams *p, const KMWavefrontPlan *plan) {
    long kernel_volume;
    long *spatial_strides;

    if (!p || !p->input || !p->kernel || !p->output ||
        !p->dims || p->ndims <= 0 || p->D <= 0 || p->K <= 0) {
        return;
    }
    if (!km_wavefront_plan_matches_dims(plan, p->dims, p->ndims)) return;

    kernel_volume = convnd_kernel_volume(p->ndims, p->K);
    spatial_strides = (long *)malloc((size_t)p->ndims * sizeof(long));
    if (!spatial_strides) return;

    convnd_make_row_major_strides(p->dims, p->ndims, spatial_strides);

    for (long level = 0; level <= plan->max_level; level++) {
        const long *level_offsets = km_wavefront_plan_level_offsets(plan, level);
        long level_size = km_wavefront_plan_level_size(plan, level);
        if (!level_offsets || level_size < 0) break;

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (long point = 0; point < level_size; point++) {
            long out_linear = level_offsets[point];
            long out_coords[p->ndims];

            convnd_unravel_index(out_linear, p->dims, p->ndims, out_coords);

            for (long d = 0; d < p->D; d++) {
                float sum = p->bias ? p->bias[d] : 0.0f;

                for (long kernel_linear = 0; kernel_linear < kernel_volume; kernel_linear++) {
                    long tmp = kernel_linear;
                    long src_linear = 0;
                    int valid = 1;

                    for (long axis = p->ndims; axis-- > 0;) {
                        long k_axis = tmp % p->K;
                        long src_coord;

                        tmp /= p->K;
                        src_coord = out_coords[axis] - (p->K - 1 - k_axis);
                        if (src_coord < 0 || src_coord >= p->dims[axis]) {
                            valid = 0;
                            break;
                        }
                        src_linear += src_coord * spatial_strides[axis];
                    }

                    if (!valid) continue;
                    sum += p->input[src_linear * p->D + d] *
                           p->kernel[kernel_linear * p->D + d];
                }

                p->output[out_linear * p->D + d] = sum;
            }
        }
    }

    free(spatial_strides);
}

/* ============================================================
 * Backward Wavefront
 * ============================================================ */

void convnd_backward_wavefront(ConvNDParams *p, const KMWavefrontPlan *plan) {
    long total_spatial;
    long kernel_volume;
    long *spatial_strides;

    if (!p || !p->input || !p->kernel || !p->dy ||
        !p->dinput || !p->dkernel ||
        !p->dims || p->ndims <= 0 || p->D <= 0 || p->K <= 0) {
        return;
    }
    if (!km_wavefront_plan_matches_dims(plan, p->dims, p->ndims)) return;

    total_spatial = product(p->dims, p->ndims);
    kernel_volume = convnd_kernel_volume(p->ndims, p->K);
    spatial_strides = (long *)malloc((size_t)p->ndims * sizeof(long));
    if (!spatial_strides) return;

    convnd_make_row_major_strides(p->dims, p->ndims, spatial_strides);
    memset(p->dinput, 0, (size_t)(total_spatial * p->D) * sizeof(float));
    memset(p->dkernel, 0, (size_t)(kernel_volume * p->D) * sizeof(float));
    if (p->dbias) memset(p->dbias, 0, (size_t)p->D * sizeof(float));

    for (long level = plan->max_level; level >= 0; level--) {
        const long *level_offsets = km_wavefront_plan_level_offsets(plan, level);
        long level_size = km_wavefront_plan_level_size(plan, level);
        if (!level_offsets || level_size < 0) break;

        for (long point = 0; point < level_size; point++) {
            long out_linear = level_offsets[point];
            long out_coords[p->ndims];

            convnd_unravel_index(out_linear, p->dims, p->ndims, out_coords);

            for (long d = 0; d < p->D; d++) {
                float grad = p->dy[out_linear * p->D + d];

                if (p->dbias) p->dbias[d] += grad;

                for (long kernel_linear = 0; kernel_linear < kernel_volume; kernel_linear++) {
                    long tmp = kernel_linear;
                    long src_linear = 0;
                    int valid = 1;

                    for (long axis = p->ndims; axis-- > 0;) {
                        long k_axis = tmp % p->K;
                        long src_coord;

                        tmp /= p->K;
                        src_coord = out_coords[axis] - (p->K - 1 - k_axis);
                        if (src_coord < 0 || src_coord >= p->dims[axis]) {
                            valid = 0;
                            break;
                        }
                        src_linear += src_coord * spatial_strides[axis];
                    }

                    if (!valid) continue;
                    p->dkernel[kernel_linear * p->D + d] +=
                        grad * p->input[src_linear * p->D + d];
                    p->dinput[src_linear * p->D + d] +=
                        grad * p->kernel[kernel_linear * p->D + d];
                }
            }
        }
    }

    free(spatial_strides);
}

/* ============================================================
 * Entry point unifié — avec dispatch automatique GPU
 * ============================================================ */

void convnd(ConvNDParams *p, ConvNDMode mode) {
    KMWavefrontPlan *plan;

    if (!p || !p->dims || p->ndims <= 0) return;

    /* Initialize backend on first call */
    static int backend_initialized = 0;
    if (!backend_initialized) {
        kmamba_backend_init();
        backend_initialized = 1;
    }

    /* Automatic GPU dispatch if available */
    KMambaBackend backend = kmamba_backend_select();

#ifdef KMAMBA_BUILD_CUDA
    if (backend == KMAMBA_BACKEND_GPU) {
        /* Try GPU first */
        int gpu_ok = 1;
        if (mode & CONVND_FORWARD) {
            if (om_convnd_forward(p) != 0) gpu_ok = 0;
        }
        if (gpu_ok && (mode & CONVND_BACKWARD)) {
            if (om_convnd_backward(p) != 0) gpu_ok = 0;
        }
        if (gpu_ok) return; /* GPU success */
        /* Fall back to CPU on GPU failure */
    }
#endif

    /* CPU implementation */
    plan = km_wavefront_plan_create(p->dims, p->ndims);
    if (!plan) return;

    if (mode & CONVND_FORWARD) {
        convnd_forward_wavefront(p, plan);
    }

    if (mode & CONVND_BACKWARD) {
        convnd_backward_wavefront(p, plan);
    }

    km_wavefront_plan_free(plan);
}
