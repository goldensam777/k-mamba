#include <math.h>
#include <stdio.h>
#include <string.h>

#include "scan_nd.h"

#define PASS_TAG "  [PASS]"
#define FAIL_TAG "  [FAIL]"

static int nearly_equal(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static int test_scannd_validate_rejects_invalid(void) {
    const long dims[2] = {2, 0};
    float one = 1.0f;
    ScanNDParams p = {
        .dims = dims,
        .ndims = 2,
        .D = 1,
        .M = 1,
        .x = &one,
        .A = &one,
        .B = &one,
        .C = &one,
        .delta = &one,
        .h = &one,
        .y = &one
    };

    printf("\n--- Validation paramètres invalides ---\n");
    if (scannd_validate(&p) != 0) {
        printf("%s scannd_validate a accepté dims invalides\n", FAIL_TAG);
        return 0;
    }

    printf("%s scannd_validate rejette les dimensions invalides\n", PASS_TAG);
    return 1;
}

static int test_scannd_ref_matches_expected_2d(void) {
    const long dims[2] = {2, 2};
    float x[4] = {1.f, 1.f, 1.f, 1.f};
    float A[2] = {logf(2.f), logf(3.f)};
    float B[4] = {1.f, 1.f, 1.f, 1.f};
    float C[4] = {1.f, 1.f, 1.f, 1.f};
    float delta[8];
    float h[4] = {0.f, 0.f, 0.f, 0.f};
    float y[4] = {0.f, 0.f, 0.f, 0.f};
    const float expected[4] = {1.f, 4.f, 3.f, 18.f};
    ScanNDParams p;
    int ok = 1;

    for (int i = 0; i < 4; i++) {
        delta[i] = 1.f;
        delta[4 + i] = 1.f;
    }

    p.dims = dims;
    p.ndims = 2;
    p.D = 1;
    p.M = 1;
    p.x = x;
    p.A = A;
    p.B = B;
    p.C = C;
    p.delta = delta;
    p.h = h;
    p.y = y;

    printf("\n--- scannd_ref 2D attendu ---\n");

    if (scannd_ref(&p) != 0) {
        printf("%s scannd_ref a échoué sur le cas 2D simple\n", FAIL_TAG);
        return 0;
    }

    for (int i = 0; i < 4; i++) {
        if (!nearly_equal(y[i], expected[i], 1e-5f)) {
            printf("%s y[%d] attendu=%.6f obtenu=%.6f\n",
                   FAIL_TAG, i, expected[i], y[i]);
            ok = 0;
        }
    }

    if (ok) printf("%s scannd_ref reproduit le cas 2D de référence\n", PASS_TAG);
    return ok;
}

static int test_scannd_ref_matches_expected_3d(void) {
    const long dims[3] = {2, 2, 2};
    float x[8];
    float A[3] = {0.f, 0.f, 0.f};
    float B[8];
    float C[8];
    float delta[24];
    float h[8];
    float y[8];
    const float expected[8] = {1.f, 2.f, 2.f, 5.f, 2.f, 5.f, 5.f, 16.f};
    ScanNDParams p;
    int ok = 1;

    for (int i = 0; i < 8; i++) {
        x[i] = 1.f;
        B[i] = 1.f;
        C[i] = 1.f;
        h[i] = 0.f;
        y[i] = 0.f;
    }
    for (int i = 0; i < 24; i++) delta[i] = 1.f;

    p.dims = dims;
    p.ndims = 3;
    p.D = 1;
    p.M = 1;
    p.x = x;
    p.A = A;
    p.B = B;
    p.C = C;
    p.delta = delta;
    p.h = h;
    p.y = y;

    printf("\n--- scannd_ref 3D attendu ---\n");

    if (scannd_ref(&p) != 0) {
        printf("%s scannd_ref a échoué sur le cas 3D simple\n", FAIL_TAG);
        return 0;
    }

    for (int i = 0; i < 8; i++) {
        if (!nearly_equal(y[i], expected[i], 1e-5f)) {
            printf("%s y[%d] attendu=%.6f obtenu=%.6f\n",
                   FAIL_TAG, i, expected[i], y[i]);
            ok = 0;
        }
    }

    if (ok) printf("%s scannd_ref reproduit le cas 3D de référence\n", PASS_TAG);
    return ok;
}

static int test_scannd_dispatch_matches_ref_1d(void) {
    const long dims[1] = {4};
    float x[4] = {1.f, 1.f, 1.f, 1.f};
    float A[1] = {logf(2.f)};
    float B[4] = {1.f, 1.f, 1.f, 1.f};
    float C[4] = {1.f, 1.f, 1.f, 1.f};
    float delta[4] = {1.f, 1.f, 1.f, 1.f};
    float h_ref[4] = {0.f, 0.f, 0.f, 0.f};
    float y_ref[4] = {0.f, 0.f, 0.f, 0.f};
    float h_fast[4] = {0.f, 0.f, 0.f, 0.f};
    float y_fast[4] = {0.f, 0.f, 0.f, 0.f};
    ScanNDParams p_ref;
    ScanNDParams p_fast;
    int ok = 1;

    p_ref.dims = dims;
    p_ref.ndims = 1;
    p_ref.D = 1;
    p_ref.M = 1;
    p_ref.x = x;
    p_ref.A = A;
    p_ref.B = B;
    p_ref.C = C;
    p_ref.delta = delta;
    p_ref.h = h_ref;
    p_ref.y = y_ref;

    p_fast = p_ref;
    p_fast.h = h_fast;
    p_fast.y = y_fast;

    printf("\n--- dispatch scannd() vs référence 1D ---\n");

    if (scannd_ref(&p_ref) != 0 || scannd(&p_fast) != 0) {
        printf("%s échec exécution 1D\n", FAIL_TAG);
        return 0;
    }

    for (int i = 0; i < 4; i++) {
        if (!nearly_equal(y_ref[i], y_fast[i], 1e-5f)) {
            printf("%s y[%d] ref=%.6f fast=%.6f\n", FAIL_TAG, i, y_ref[i], y_fast[i]);
            ok = 0;
        }
    }

    if (ok) printf("%s dispatch 1D cohérent avec la référence\n", PASS_TAG);
    return ok;
}

static int test_scannd_dispatch_matches_ref_2d(void) {
    const long dims[2] = {2, 2};
    float x[4] = {1.f, 1.f, 1.f, 1.f};
    float A[2] = {logf(2.f), logf(3.f)};
    float B[4] = {1.f, 1.f, 1.f, 1.f};
    float C[4] = {1.f, 1.f, 1.f, 1.f};
    float delta[8];
    float h_ref[4] = {0.f, 0.f, 0.f, 0.f};
    float y_ref[4] = {0.f, 0.f, 0.f, 0.f};
    float h_fast[4] = {0.f, 0.f, 0.f, 0.f};
    float y_fast[4] = {0.f, 0.f, 0.f, 0.f};
    ScanNDParams p_ref;
    ScanNDParams p_fast;
    int ok = 1;

    for (int i = 0; i < 4; i++) {
        delta[i] = 1.f;
        delta[4 + i] = 1.f;
    }

    p_ref.dims = dims;
    p_ref.ndims = 2;
    p_ref.D = 1;
    p_ref.M = 1;
    p_ref.x = x;
    p_ref.A = A;
    p_ref.B = B;
    p_ref.C = C;
    p_ref.delta = delta;
    p_ref.h = h_ref;
    p_ref.y = y_ref;

    p_fast = p_ref;
    p_fast.h = h_fast;
    p_fast.y = y_fast;

    printf("\n--- dispatch scannd() vs backend 2D ---\n");

    if (scannd_ref(&p_ref) != 0 || scannd(&p_fast) != 0) {
        printf("%s échec exécution 2D\n", FAIL_TAG);
        return 0;
    }

    for (int i = 0; i < 4; i++) {
        if (!nearly_equal(y_ref[i], y_fast[i], 1e-5f)) {
            printf("%s y[%d] ref=%.6f fast=%.6f\n", FAIL_TAG, i, y_ref[i], y_fast[i]);
            ok = 0;
        }
    }

    if (ok) printf("%s dispatch 2D cohérent avec le backend ASM\n", PASS_TAG);
    return ok;
}

static int test_scannd_ref_with_plan_matches_plain_ref(void) {
    const long dims[3] = {2, 2, 2};
    float x[8];
    float A[3] = {0.f, 0.f, 0.f};
    float B[8];
    float C[8];
    float delta[24];
    float h_ref[8] = {0.f};
    float y_ref[8] = {0.f};
    float h_plan[8] = {0.f};
    float y_plan[8] = {0.f};
    ScanNDParams p_ref;
    ScanNDParams p_plan;
    KMWavefrontPlan *plan;
    int ok = 1;

    for (int i = 0; i < 8; i++) {
        x[i] = (float)(i + 1);
        B[i] = 1.f;
        C[i] = 1.f;
    }
    for (int i = 0; i < 24; i++) delta[i] = 1.f;

    p_ref.dims = dims;
    p_ref.ndims = 3;
    p_ref.D = 1;
    p_ref.M = 1;
    p_ref.x = x;
    p_ref.A = A;
    p_ref.B = B;
    p_ref.C = C;
    p_ref.delta = delta;
    p_ref.h = h_ref;
    p_ref.y = y_ref;

    p_plan = p_ref;
    p_plan.h = h_plan;
    p_plan.y = y_plan;

    printf("\n--- scannd_ref_with_plan() vs scannd_ref() ---\n");

    plan = km_wavefront_plan_create(dims, 3);
    if (!plan) {
        printf("%s création du plan wavefront\n", FAIL_TAG);
        return 0;
    }

    if (scannd_ref(&p_ref) != 0 || scannd_ref_with_plan(&p_plan, plan) != 0) {
        printf("%s échec exécution scannd_ref_with_plan\n", FAIL_TAG);
        km_wavefront_plan_free(plan);
        return 0;
    }

    for (int i = 0; i < 8; i++) {
        if (!nearly_equal(y_ref[i], y_plan[i], 1e-5f)) {
            printf("%s y[%d] ref=%.6f plan=%.6f\n", FAIL_TAG, i, y_ref[i], y_plan[i]);
            ok = 0;
        }
    }

    if (ok) printf("%s plan wavefront partagé cohérent pour scanND\n", PASS_TAG);
    km_wavefront_plan_free(plan);
    return ok;
}

int main(void) {
    int passed = 0;
    int total = 0;

    printf("=== Tests scan_nd ===\n");

    total++; passed += test_scannd_validate_rejects_invalid();
    total++; passed += test_scannd_ref_matches_expected_2d();
    total++; passed += test_scannd_ref_matches_expected_3d();
    total++; passed += test_scannd_dispatch_matches_ref_1d();
    total++; passed += test_scannd_dispatch_matches_ref_2d();
    total++; passed += test_scannd_ref_with_plan_matches_plain_ref();

    printf("\n=== Résultat: %d/%d tests passent ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
