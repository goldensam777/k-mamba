#include <math.h>
#include <stdio.h>
#include <string.h>

#include "kmamba.h"

#define PASS_TAG "  [PASS]"
#define FAIL_TAG "  [FAIL]"

static int nearly_equal(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static int test_convnd_full_expected_2d(void) {
    const long dims[2] = {2, 2};
    const float input[4] = {1.f, 2.f, 3.f, 4.f};
    const float kernel[4] = {1.f, 2.f, 3.f, 4.f};
    const float bias[1] = {0.5f};
    const float expected[4] = {4.5f, 11.5f, 14.5f, 30.5f};
    float output[4] = {0.f, 0.f, 0.f, 0.f};
    ConvNDFullParams p;
    int ok = 1;

    printf("\n--- convnd_full_ref 2D attendu ---\n");

    if (convnd_full_kernel_volume(2, 2) != 4) {
        printf("%s kernel_volume attendu=4 obtenu=%ld\n",
               FAIL_TAG, convnd_full_kernel_volume(2, 2));
        return 0;
    }

    p.input = input;
    p.kernel = kernel;
    p.bias = bias;
    p.output = output;
    p.dy = NULL;
    p.dinput = NULL;
    p.dkernel = NULL;
    p.dbias = NULL;
    p.dims = dims;
    p.ndims = 2;
    p.D = 1;
    p.K = 2;

    convnd_full_ref(&p, CONVND_FORWARD);

    for (int i = 0; i < 4; i++) {
        if (!nearly_equal(output[i], expected[i], 1e-5f)) {
            printf("%s y[%d] attendu=%.6f obtenu=%.6f\n",
                   FAIL_TAG, i, expected[i], output[i]);
            ok = 0;
        }
    }

    if (ok) printf("%s convnd_full_ref reproduit le cas 2D de référence\n", PASS_TAG);
    return ok;
}

static void build_factorized_full_kernel_2d(float *full_kernel,
                                            const float *separable_kernel,
                                            long K, long D) {
    for (long k0 = 0; k0 < K; k0++) {
        for (long k1 = 0; k1 < K; k1++) {
            long full_index = (k0 * K + k1) * D;
            long axis0_index = k0 * D;
            long axis1_index = (K + k1) * D;

            for (long d = 0; d < D; d++) {
                full_kernel[full_index + d] =
                    separable_kernel[axis0_index + d] *
                    separable_kernel[axis1_index + d];
            }
        }
    }
}

static int test_convnd_full_matches_separable_factorized_kernel(void) {
    const long dims[2] = {3, 4};
    enum { D = 2, K = 2, NDIMS = 2, TOTAL_SPATIAL = 12, TOTAL_FLOATS = TOTAL_SPATIAL * D };
    float input[TOTAL_FLOATS];
    float dy[TOTAL_FLOATS];
    float output_full[TOTAL_FLOATS];
    float output_sep[TOTAL_FLOATS];
    float dinput_full[TOTAL_FLOATS];
    float dinput_sep[TOTAL_FLOATS];
    float dbias_full[D];
    float dbias_sep[D];
    float dkernel_full[(K * K) * D];
    float dkernel_sep[NDIMS * K * D];
    float separable_kernel[NDIMS * K * D] = {
        1.0f, 0.5f,
        0.25f, -1.0f,
        0.5f, 1.5f,
        -2.0f, 0.75f
    };
    float full_kernel[(K * K) * D];
    float bias[D] = {0.25f, -0.75f};
    ConvNDFullParams full;
    ConvNDParams sep;
    int ok = 1;

    printf("\n--- convnd_full_ref vs convnd séparable factorisée ---\n");

    for (int i = 0; i < TOTAL_FLOATS; i++) {
        input[i] = (float)(i + 1) * 0.25f - 1.0f;
        dy[i] = (float)((i % 5) - 2) * 0.5f;
        output_full[i] = 0.0f;
        output_sep[i] = 0.0f;
        dinput_full[i] = 0.0f;
        dinput_sep[i] = 0.0f;
    }
    memset(dbias_full, 0, sizeof(dbias_full));
    memset(dbias_sep, 0, sizeof(dbias_sep));
    memset(dkernel_full, 0, sizeof(dkernel_full));
    memset(dkernel_sep, 0, sizeof(dkernel_sep));

    build_factorized_full_kernel_2d(full_kernel, separable_kernel, K, D);

    full.input = input;
    full.kernel = full_kernel;
    full.bias = bias;
    full.output = output_full;
    full.dy = dy;
    full.dinput = dinput_full;
    full.dkernel = dkernel_full;
    full.dbias = dbias_full;
    full.dims = dims;
    full.ndims = NDIMS;
    full.D = D;
    full.K = K;

    sep.input = input;
    sep.kernel = separable_kernel;
    sep.bias = bias;
    sep.output = output_sep;
    sep.dy = dy;
    sep.dinput = dinput_sep;
    sep.dkernel = dkernel_sep;
    sep.dbias = dbias_sep;
    sep.dims = dims;
    sep.ndims = NDIMS;
    sep.D = D;
    sep.K = K;

    convnd_full_ref(&full, CONVND_COMPLETE);
    convnd(&sep, CONVND_COMPLETE, NULL);

    for (int i = 0; i < TOTAL_FLOATS; i++) {
        if (!nearly_equal(output_full[i], output_sep[i], 1e-5f)) {
            printf("%s output[%d] full=%.6f sep=%.6f\n",
                   FAIL_TAG, i, output_full[i], output_sep[i]);
            ok = 0;
        }
        if (!nearly_equal(dinput_full[i], dinput_sep[i], 1e-5f)) {
            printf("%s dinput[%d] full=%.6f sep=%.6f\n",
                   FAIL_TAG, i, dinput_full[i], dinput_sep[i]);
            ok = 0;
        }
    }

    for (int d = 0; d < D; d++) {
        if (!nearly_equal(dbias_full[d], dbias_sep[d], 1e-5f)) {
            printf("%s dbias[%d] full=%.6f sep=%.6f\n",
                   FAIL_TAG, d, dbias_full[d], dbias_sep[d]);
            ok = 0;
        }
    }

    if (ok) printf("%s noyau plein factorisé cohérent avec la convND séparable\n", PASS_TAG);
    return ok;
}

static int test_convnd_full_with_plan_matches_plain_ref(void) {
    const long dims[2] = {3, 3};
    enum { D = 1, K = 2, TOTAL_FLOATS = 9 };
    float input[TOTAL_FLOATS];
    float output_ref[TOTAL_FLOATS];
    float output_plan[TOTAL_FLOATS];
    float kernel[K * K] = {1.f, -1.f, 0.5f, 2.f};
    float bias[1] = {0.25f};
    ConvNDFullParams p_ref;
    ConvNDFullParams p_plan;
    KMWavefrontPlan *plan;
    int ok = 1;

    printf("\n--- convnd_full_ref_with_plan() vs convnd_full_ref() ---\n");

    for (int i = 0; i < TOTAL_FLOATS; i++) {
        input[i] = (float)(i + 1) * 0.5f;
        output_ref[i] = 0.0f;
        output_plan[i] = 0.0f;
    }

    p_ref.input = input;
    p_ref.kernel = kernel;
    p_ref.bias = bias;
    p_ref.output = output_ref;
    p_ref.dy = NULL;
    p_ref.dinput = NULL;
    p_ref.dkernel = NULL;
    p_ref.dbias = NULL;
    p_ref.dims = dims;
    p_ref.ndims = 2;
    p_ref.D = D;
    p_ref.K = K;

    p_plan = p_ref;
    p_plan.output = output_plan;

    plan = km_wavefront_plan_create(dims, 2);
    if (!plan) {
        printf("%s création du plan wavefront\n", FAIL_TAG);
        return 0;
    }

    convnd_full_ref(&p_ref, CONVND_FORWARD);
    convnd_full_ref_with_plan(&p_plan, plan, CONVND_FORWARD);

    for (int i = 0; i < TOTAL_FLOATS; i++) {
        if (!nearly_equal(output_ref[i], output_plan[i], 1e-5f)) {
            printf("%s y[%d] ref=%.6f plan=%.6f\n",
                   FAIL_TAG, i, output_ref[i], output_plan[i]);
            ok = 0;
        }
    }

    if (ok) printf("%s plan wavefront partagé cohérent pour convND full\n", PASS_TAG);
    km_wavefront_plan_free(plan);
    return ok;
}

int main(void) {
    int ok = 1;

    ok &= test_convnd_full_expected_2d();
    ok &= test_convnd_full_matches_separable_factorized_kernel();
    ok &= test_convnd_full_with_plan_matches_plain_ref();

    if (!ok) {
        printf("\nconvND full tests: FAILED\n");
        return 1;
    }

    printf("\nconvND full tests: PASSED\n");
    return 0;
}
