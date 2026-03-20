/*
 * test_wavefront_nd.c — Générateur de wavefront ND
 *
 * Vérifie :
 *   1. les tailles de niveaux en 2D / 3D
 *   2. la couverture exacte de tous les indices
 *   3. la cohérence du niveau sum(idx)
 *   4. l'arrêt anticipé via le callback
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wavefront_nd.h"

#define PASS_TAG "  [PASS]"
#define FAIL_TAG "  [FAIL]"

typedef struct {
    const long *dims;
    long ndims;
    long total_points;
    unsigned char *seen;
    long visits;
    long current_level;
    long expected_ordinal;
    int ok;
} CoverageCtx;

static long sum_idx(const long *idx, long ndims) {
    long s = 0;
    for (long i = 0; i < ndims; i++) s += idx[i];
    return s;
}

static int coverage_visit(const long *idx,
                          long ndims,
                          long level,
                          long ordinal_in_level,
                          void *user) {
    CoverageCtx *ctx = (CoverageCtx *)user;
    long offset;

    if (!ctx || !idx || ndims != ctx->ndims) {
        if (ctx) ctx->ok = 0;
        return 11;
    }

    if (level != ctx->current_level) {
        ctx->current_level = level;
        ctx->expected_ordinal = 0;
    }

    if (ordinal_in_level != ctx->expected_ordinal) {
        ctx->ok = 0;
        return 12;
    }
    ctx->expected_ordinal++;

    if (sum_idx(idx, ndims) != level) {
        ctx->ok = 0;
        return 13;
    }

    offset = wavefront_nd_row_major_offset(ctx->dims, idx, ndims);
    if (offset < 0 || offset >= ctx->total_points) {
        ctx->ok = 0;
        return 14;
    }

    if (ctx->seen[offset] != 0) {
        ctx->ok = 0;
        return 15;
    }

    ctx->seen[offset] = 1;
    ctx->visits++;
    return 0;
}

typedef struct {
    long seen;
    long stop_after;
    int code;
} StopCtx;

static int stop_after_n_visit(const long *idx,
                              long ndims,
                              long level,
                              long ordinal_in_level,
                              void *user) {
    StopCtx *ctx = (StopCtx *)user;
    (void)idx;
    (void)ndims;
    (void)level;
    (void)ordinal_in_level;

    ctx->seen++;
    if (ctx->seen >= ctx->stop_after) return ctx->code;
    return 0;
}

static int test_level_sizes_2d(void) {
    const long dims[2] = {3, 4};
    const long expected[] = {1, 2, 3, 3, 2, 1};
    const long max_level = 5;
    int ok = 1;

    printf("\n--- Tailles de niveaux 2D (3x4) ---\n");

    if (wavefront_nd_max_level(dims, 2) != max_level) {
        printf("%s max_level attendu=%ld obtenu=%ld\n",
               FAIL_TAG, max_level, wavefront_nd_max_level(dims, 2));
        ok = 0;
    }

    for (long level = 0; level <= max_level; level++) {
        long got = wavefront_nd_level_size(dims, 2, level);
        if (got != expected[level]) {
            printf("%s level=%ld attendu=%ld obtenu=%ld\n",
                   FAIL_TAG, level, expected[level], got);
            ok = 0;
        }
    }

    if (ok) printf("%s tailles 2D correctes\n", PASS_TAG);
    return ok;
}

static int test_level_sizes_3d(void) {
    const long dims[3] = {2, 3, 2};
    const long expected[] = {1, 3, 4, 3, 1};
    const long max_level = 4;
    int ok = 1;

    printf("\n--- Tailles de niveaux 3D (2x3x2) ---\n");

    if (wavefront_nd_max_level(dims, 3) != max_level) {
        printf("%s max_level attendu=%ld obtenu=%ld\n",
               FAIL_TAG, max_level, wavefront_nd_max_level(dims, 3));
        ok = 0;
    }

    for (long level = 0; level <= max_level; level++) {
        long got = wavefront_nd_level_size(dims, 3, level);
        if (got != expected[level]) {
            printf("%s level=%ld attendu=%ld obtenu=%ld\n",
                   FAIL_TAG, level, expected[level], got);
            ok = 0;
        }
    }

    if (ok) printf("%s tailles 3D correctes\n", PASS_TAG);
    return ok;
}

static int test_full_coverage(void) {
    const long dims[3] = {2, 3, 2};
    long total_points = wavefront_nd_total_points(dims, 3);
    CoverageCtx ctx;
    int rc;
    int ok = 1;

    printf("\n--- Couverture complète 3D (2x3x2) ---\n");

    ctx.dims = dims;
    ctx.ndims = 3;
    ctx.total_points = total_points;
    ctx.seen = (unsigned char *)calloc((size_t)total_points, sizeof(unsigned char));
    ctx.visits = 0;
    ctx.current_level = -1;
    ctx.expected_ordinal = 0;
    ctx.ok = 1;

    if (!ctx.seen) {
        printf("%s allocation seen\n", FAIL_TAG);
        return 0;
    }

    rc = wavefront_nd_for_each_level(dims, 3, NULL, coverage_visit, &ctx);
    if (rc != 0 || !ctx.ok) {
        printf("%s callback rc=%d ctx_ok=%d\n", FAIL_TAG, rc, ctx.ok);
        ok = 0;
    }

    if (ctx.visits != total_points) {
        printf("%s visites attendues=%ld obtenues=%ld\n",
               FAIL_TAG, total_points, ctx.visits);
        ok = 0;
    }

    for (long i = 0; i < total_points; i++) {
        if (ctx.seen[i] != 1) {
            printf("%s point offset=%ld non couvert correctement\n", FAIL_TAG, i);
            ok = 0;
            break;
        }
    }

    if (wavefront_nd_row_major_offset(dims, (const long[]){1, 2, 1}, 3) != 11) {
        printf("%s offset row-major attendu=11\n", FAIL_TAG);
        ok = 0;
    }

    if (ok) printf("%s couverture complète et offsets cohérents\n", PASS_TAG);

    free(ctx.seen);
    return ok;
}

static int test_early_stop(void) {
    const long dims[2] = {3, 4};
    StopCtx ctx = {.seen = 0, .stop_after = 2, .code = 77};
    int rc;
    int ok = 1;

    printf("\n--- Arrêt anticipé ---\n");

    rc = wavefront_nd_for_level(dims, 2, 3, NULL, stop_after_n_visit, &ctx);
    if (rc != 77) {
        printf("%s code arrêt attendu=77 obtenu=%d\n", FAIL_TAG, rc);
        ok = 0;
    }
    if (ctx.seen != 2) {
        printf("%s visites avant arrêt attendues=2 obtenues=%ld\n", FAIL_TAG, ctx.seen);
        ok = 0;
    }

    if (ok) printf("%s arrêt anticipé propagé correctement\n", PASS_TAG);
    return ok;
}

static int test_invalid_params(void) {
    const long bad_dims[2] = {3, 0};
    int ok = 1;

    printf("\n--- Paramètres invalides ---\n");

    if (wavefront_nd_validate_dims(bad_dims, 2) != 0) ok = 0;
    if (wavefront_nd_total_points(bad_dims, 2) != -1) ok = 0;
    if (wavefront_nd_max_level(bad_dims, 2) != -1) ok = 0;
    if (wavefront_nd_for_each_level(bad_dims, 2, NULL, coverage_visit, NULL) != -1) ok = 0;

    if (ok) printf("%s paramètres invalides rejetés\n", PASS_TAG);
    else    printf("%s validation paramètres invalides\n", FAIL_TAG);
    return ok;
}

int main(void) {
    int passed = 0;
    int total = 0;

    printf("=== Tests wavefront_nd ===\n");

    total++; passed += test_level_sizes_2d();
    total++; passed += test_level_sizes_3d();
    total++; passed += test_full_coverage();
    total++; passed += test_early_stop();
    total++; passed += test_invalid_params();

    printf("\n=== Résultat : %d/%d tests passés ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
