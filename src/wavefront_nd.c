#include "wavefront_nd.h"

#include <limits.h>
#include <stdlib.h>

static int safe_add_long(long a, long b, long *out) {
    if (!out) return 0;
    if ((b > 0 && a > LONG_MAX - b) || (b < 0 && a < LONG_MIN - b)) return 0;
    *out = a + b;
    return 1;
}

static int safe_mul_long(long a, long b, long *out) {
    if (!out) return 0;
    if (a < 0 || b < 0) return 0;
    if (a == 0 || b == 0) {
        *out = 0;
        return 1;
    }
    if (a > LONG_MAX / b) return 0;
    *out = a * b;
    return 1;
}

static long max_long(long a, long b) {
    return (a > b) ? a : b;
}

static long min_long(long a, long b) {
    return (a < b) ? a : b;
}

static int build_suffix_caps(const long *dims, long ndims, long *suffix_caps) {
    long suffix = 0;

    if (!dims || !suffix_caps || ndims <= 0) return 0;

    suffix_caps[ndims] = 0;
    for (long axis = ndims - 1; axis >= 0; axis--) {
        long cap = dims[axis] - 1;
        if (cap < 0) return 0;
        if (!safe_add_long(suffix, cap, &suffix)) return 0;
        suffix_caps[axis] = suffix;
    }
    return 1;
}

static long count_level_recursive(const long *dims,
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

    lo = max_long(0, remaining - suffix_caps[axis + 1]);
    hi = min_long(dims[axis] - 1, remaining);

    for (long value = lo; value <= hi; value++) {
        long sub = count_level_recursive(dims, suffix_caps, ndims, axis + 1, remaining - value);
        if (sub < 0 || !safe_add_long(count, sub, &count)) return -1;
    }

    return count;
}

typedef struct {
    const long *dims;
    const long *suffix_caps;
    long ndims;
    long level;
    long ordinal;
    long *idx;
    WavefrontNDVisitFn visit;
    void *user;
    int stop_code;
} WavefrontEmitCtx;

static void emit_level_recursive(WavefrontEmitCtx *ctx, long axis, long remaining) {
    long lo;
    long hi;

    if (!ctx || ctx->stop_code) return;

    if (axis == ctx->ndims - 1) {
        if (remaining >= 0 && remaining < ctx->dims[axis]) {
            int rc;
            ctx->idx[axis] = remaining;
            rc = ctx->visit(ctx->idx, ctx->ndims, ctx->level, ctx->ordinal, ctx->user);
            ctx->ordinal++;
            if (rc != 0) ctx->stop_code = rc;
        }
        return;
    }

    lo = max_long(0, remaining - ctx->suffix_caps[axis + 1]);
    hi = min_long(ctx->dims[axis] - 1, remaining);

    for (long value = lo; value <= hi; value++) {
        ctx->idx[axis] = value;
        emit_level_recursive(ctx, axis + 1, remaining - value);
        if (ctx->stop_code) return;
    }
}

static int wavefront_nd_for_level_impl(const long *dims,
                                       long ndims,
                                       long level,
                                       long *idx,
                                       const long *suffix_caps,
                                       WavefrontNDVisitFn visit,
                                       void *user) {
    WavefrontEmitCtx ctx;

    ctx.dims = dims;
    ctx.suffix_caps = suffix_caps;
    ctx.ndims = ndims;
    ctx.level = level;
    ctx.ordinal = 0;
    ctx.idx = idx;
    ctx.visit = visit;
    ctx.user = user;
    ctx.stop_code = 0;

    emit_level_recursive(&ctx, 0, level);
    return ctx.stop_code;
}

int wavefront_nd_validate_dims(const long *dims, long ndims) {
    if (!dims || ndims <= 0) return 0;
    for (long i = 0; i < ndims; i++) {
        if (dims[i] <= 0) return 0;
    }
    return 1;
}

long wavefront_nd_total_points(const long *dims, long ndims) {
    long total = 1;

    if (!wavefront_nd_validate_dims(dims, ndims)) return -1;

    for (long i = 0; i < ndims; i++) {
        if (!safe_mul_long(total, dims[i], &total)) return -1;
    }
    return total;
}

long wavefront_nd_max_level(const long *dims, long ndims) {
    long level = 0;

    if (!wavefront_nd_validate_dims(dims, ndims)) return -1;

    for (long i = 0; i < ndims; i++) {
        if (!safe_add_long(level, dims[i] - 1, &level)) return -1;
    }
    return level;
}

long wavefront_nd_row_major_offset(const long *dims, const long *idx, long ndims) {
    long offset = 0;

    if (!wavefront_nd_validate_dims(dims, ndims) || !idx) return -1;

    for (long axis = 0; axis < ndims; axis++) {
        if (idx[axis] < 0 || idx[axis] >= dims[axis]) return -1;
        if (!safe_mul_long(offset, dims[axis], &offset)) return -1;
        if (!safe_add_long(offset, idx[axis], &offset)) return -1;
    }

    return offset;
}

long wavefront_nd_level_size(const long *dims, long ndims, long level) {
    long max_level;
    long *suffix_caps;
    long count;

    if (!wavefront_nd_validate_dims(dims, ndims)) return -1;

    max_level = wavefront_nd_max_level(dims, ndims);
    if (max_level < 0 || level < 0 || level > max_level) return -1;

    suffix_caps = (long *)malloc((size_t)(ndims + 1) * sizeof(long));
    if (!suffix_caps) return -1;
    if (!build_suffix_caps(dims, ndims, suffix_caps)) {
        free(suffix_caps);
        return -1;
    }

    count = count_level_recursive(dims, suffix_caps, ndims, 0, level);
    free(suffix_caps);
    return count;
}

int wavefront_nd_for_level(const long *dims, long ndims, long level,
                           long *idx_scratch,
                           WavefrontNDVisitFn visit,
                           void *user) {
    long max_level;
    long *idx = idx_scratch;
    long *suffix_caps;
    int allocated_idx = 0;
    int rc;

    if (!wavefront_nd_validate_dims(dims, ndims) || !visit) return -1;

    max_level = wavefront_nd_max_level(dims, ndims);
    if (max_level < 0 || level < 0 || level > max_level) return -1;

    if (!idx) {
        idx = (long *)malloc((size_t)ndims * sizeof(long));
        if (!idx) return -1;
        allocated_idx = 1;
    }

    suffix_caps = (long *)malloc((size_t)(ndims + 1) * sizeof(long));
    if (!suffix_caps) {
        if (allocated_idx) free(idx);
        return -1;
    }
    if (!build_suffix_caps(dims, ndims, suffix_caps)) {
        free(suffix_caps);
        if (allocated_idx) free(idx);
        return -1;
    }

    rc = wavefront_nd_for_level_impl(dims, ndims, level, idx, suffix_caps, visit, user);

    free(suffix_caps);
    if (allocated_idx) free(idx);
    return rc;
}

int wavefront_nd_for_each_level(const long *dims, long ndims,
                                long *idx_scratch,
                                WavefrontNDVisitFn visit,
                                void *user) {
    long max_level;
    long *idx = idx_scratch;
    long *suffix_caps;
    int allocated_idx = 0;
    int rc = 0;

    if (!wavefront_nd_validate_dims(dims, ndims) || !visit) return -1;

    max_level = wavefront_nd_max_level(dims, ndims);
    if (max_level < 0) return -1;

    if (!idx) {
        idx = (long *)malloc((size_t)ndims * sizeof(long));
        if (!idx) return -1;
        allocated_idx = 1;
    }

    suffix_caps = (long *)malloc((size_t)(ndims + 1) * sizeof(long));
    if (!suffix_caps) {
        if (allocated_idx) free(idx);
        return -1;
    }
    if (!build_suffix_caps(dims, ndims, suffix_caps)) {
        free(suffix_caps);
        if (allocated_idx) free(idx);
        return -1;
    }

    for (long level = 0; level <= max_level; level++) {
        rc = wavefront_nd_for_level_impl(dims, ndims, level, idx, suffix_caps, visit, user);
        if (rc != 0) break;
    }

    free(suffix_caps);
    if (allocated_idx) free(idx);
    return rc;
}
