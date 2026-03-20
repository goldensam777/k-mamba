#include "scan_nd.h"

#include <math.h>
#include <stdlib.h>

#include "scan.h"
#include "wavefront_nd.h"

typedef struct {
    ScanNDParams *p;
    long total_points;
    long *strides;
} ScanNDRefCtx;

static int scannd_build_strides(const long *dims, long ndims, long *strides) {
    long stride = 1;

    if (!dims || !strides || ndims <= 0) return 0;

    for (long axis = ndims - 1; axis >= 0; axis--) {
        strides[axis] = stride;
        stride *= dims[axis];
    }

    return 1;
}

static int scannd_visit(const long *idx,
                        long ndims,
                        long level,
                        long ordinal_in_level,
                        void *user) {
    ScanNDRefCtx *ctx = (ScanNDRefCtx *)user;
    ScanNDParams *p;
    long offset;

    (void)level;
    (void)ordinal_in_level;

    if (!ctx || !ctx->p || !idx) return -1;
    p = ctx->p;

    offset = wavefront_nd_row_major_offset(p->dims, idx, ndims);
    if (offset < 0 || offset >= ctx->total_points) return -1;

    for (long d = 0; d < p->D; d++) {
        float x_val = p->x[offset * p->D + d];
        float dt_bar = 0.0f;
        float y_acc = 0.0f;

        for (long axis = 0; axis < ndims; axis++) {
            dt_bar += p->delta[(axis * ctx->total_points + offset) * p->D + d];
        }
        dt_bar /= (float)ndims;

        for (long m = 0; m < p->M; m++) {
            long dm = d * p->M + m;
            long pdm = offset * p->D * p->M + dm;
            float h_new = dt_bar * p->B[pdm] * x_val;

            for (long axis = 0; axis < ndims; axis++) {
                if (idx[axis] > 0) {
                    long prev_offset = offset - ctx->strides[axis];
                    float dt_axis = p->delta[(axis * ctx->total_points + offset) * p->D + d];
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

    return 0;
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

int scannd_ref(ScanNDParams *p) {
    ScanNDRefCtx ctx;
    long total_points;
    long *strides;
    int rc;

    if (!scannd_validate(p)) return -1;

    total_points = wavefront_nd_total_points(p->dims, p->ndims);
    if (total_points <= 0) return -1;

    strides = (long *)malloc((size_t)p->ndims * sizeof(long));
    if (!strides) return -1;
    if (!scannd_build_strides(p->dims, p->ndims, strides)) {
        free(strides);
        return -1;
    }

    ctx.p = p;
    ctx.total_points = total_points;
    ctx.strides = strides;

    rc = wavefront_nd_for_each_level(p->dims, p->ndims, NULL, scannd_visit, &ctx);
    free(strides);
    return (rc == 0) ? 0 : -1;
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
