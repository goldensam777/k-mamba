#include "scan_nd.h"

#include <math.h>
#include <stdlib.h>

#include "scan.h"
#include "wavefront_plan.h"

static int scannd_build_strides(const long *dims, long ndims, long *strides) {
    long stride = 1;

    if (!dims || !strides || ndims <= 0) return 0;

    for (long axis = ndims - 1; axis >= 0; axis--) {
        strides[axis] = stride;
        stride *= dims[axis];
    }

    return 1;
}

static void scannd_offset_to_idx(long offset,
                                 const long *dims,
                                 const long *strides,
                                 long ndims,
                                 long *idx) {
    if (!dims || !strides || !idx || ndims <= 0) return;

    for (long axis = 0; axis < ndims; axis++) {
        idx[axis] = (offset / strides[axis]) % dims[axis];
    }
}

int scannd_validate(const ScanNDParams *p) {
    long total_points;

    if (!p || !p->dims) return 0;
    if (p->ndims <= 0 || p->D <= 0 || p->M <= 0) return 0;
    if (!p->x || !p->A || !p->B || !p->C || !p->delta || !p->h || !p->y) return 0;
    if (!wavefront_nd_validate_dims(p->dims, p->ndims)) return 0;

    total_points = wavefront_nd_total_points(p->dims, p->ndims);
    if (total_points <= 0) return 0;

    return 1;
}

int scannd_ref_with_plan(ScanNDParams *p, const KMWavefrontPlan *plan) {
    long total_points;
    long *strides;
    long *idx;

    if (!scannd_validate(p)) return -1;
    if (!km_wavefront_plan_matches_dims(plan, p->dims, p->ndims)) return -1;

    total_points = plan->total_points;
    if (total_points <= 0) return -1;

    strides = (long *)malloc((size_t)p->ndims * sizeof(long));
    idx = (long *)malloc((size_t)p->ndims * sizeof(long));
    if (!strides || !idx) {
        free(strides);
        free(idx);
        return -1;
    }
    if (!scannd_build_strides(p->dims, p->ndims, strides)) {
        free(strides);
        free(idx);
        return -1;
    }

    for (long level = 0; level <= plan->max_level; level++) {
        const long *level_offsets = km_wavefront_plan_level_offsets(plan, level);
        long level_size = km_wavefront_plan_level_size(plan, level);
        if (!level_offsets || level_size < 0) {
            free(strides);
            free(idx);
            return -1;
        }

        for (long point = 0; point < level_size; point++) {
            long offset = level_offsets[point];

            if (offset < 0 || offset >= total_points) {
                free(strides);
                free(idx);
                return -1;
            }

            scannd_offset_to_idx(offset, p->dims, strides, p->ndims, idx);

            for (long d = 0; d < p->D; d++) {
                float x_val = p->x[offset * p->D + d];
                float dt_bar = 0.0f;
                float y_acc = 0.0f;

                for (long axis = 0; axis < p->ndims; axis++) {
                    dt_bar += p->delta[(axis * total_points + offset) * p->D + d];
                }
                dt_bar /= (float)p->ndims;

                for (long m = 0; m < p->M; m++) {
                    long dm = d * p->M + m;
                    long pdm = offset * p->D * p->M + dm;
                    float h_new = dt_bar * p->B[pdm] * x_val;

                    for (long axis = 0; axis < p->ndims; axis++) {
                        if (idx[axis] > 0) {
                            long prev_offset = offset - strides[axis];
                            float dt_axis = p->delta[(axis * total_points + offset) * p->D + d];
                            float a_val = p->A[(axis * p->D + d) * p->M + m];
                            float decay = expf(dt_axis * a_val);
                            long prev_pdm = prev_offset * p->D * p->M + dm;
                            h_new += decay * p->h[prev_pdm];
                        }
                    }

                    p->h[pdm] = h_new;
                    y_acc += p->C[pdm] * h_new;
                }

                p->y[offset * p->D + d] = y_acc;
            }
        }
    }

    free(strides);
    free(idx);
    return 0;
}

int scannd_ref(ScanNDParams *p) {
    KMWavefrontPlan *plan;
    int rc;

    if (!scannd_validate(p)) return -1;

    plan = km_wavefront_plan_create(p->dims, p->ndims);
    if (!plan) return -1;

    rc = scannd_ref_with_plan(p, plan);
    km_wavefront_plan_free(plan);
    return rc;
}

int scannd(ScanNDParams *p) {
    if (!scannd_validate(p)) return -1;

    if (p->ndims == 1) {
        ScanParams p1 = {
            .x = (float *)p->x,
            .A = (float *)p->A,
            .B = (float *)p->B,
            .C = (float *)p->C,
            .delta = (float *)p->delta,
            .h = p->h,
            .y = p->y,
            .L = p->dims[0],
            .D = p->D,
            .M = p->M
        };
        scan1d(&p1);
        return 0;
    }

    if (p->ndims == 2) {
        long total_points = wavefront_nd_total_points(p->dims, p->ndims);
        Scan2DParams p2;

        if (total_points <= 0) return -1;

        p2.x = (float *)p->x;
        p2.A1 = (float *)p->A;
        p2.A2 = (float *)(p->A + p->D * p->M);
        p2.B = (float *)p->B;
        p2.C = (float *)p->C;
        p2.delta1 = (float *)p->delta;
        p2.delta2 = (float *)(p->delta + total_points * p->D);
        p2.h = p->h;
        p2.y = p->y;
        p2.d1 = p->dims[0];
        p2.d2 = p->dims[1];
        p2.D = p->D;
        p2.M = p->M;

        scan2d(&p2);
        return 0;
    }

    return scannd_ref(p);
}
