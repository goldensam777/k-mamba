#include <stdio.h>
#include <string.h>

#include "kmamba.h"

#define PASS_TAG "  [PASS]"
#define FAIL_TAG "  [FAIL]"

static int test_block_builds_implicit_1d_plan(void) {
    MBConfig cfg = {
        .dim = 8,
        .state_size = 4,
        .seq_len = 12,
        .mimo_rank = 1,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    MambaBlock *block;
    int ok = 1;

    printf("\n--- mamba_block_create : topologie implicite 1D ---\n");

    block = mamba_block_create(&cfg);
    if (!block) {
        printf("%s mamba_block_create a échoué pour la forme implicite 1D\n",
               FAIL_TAG);
        return 0;
    }

    if (!block->wavefront_plan) {
        printf("%s aucun plan wavefront mis en cache dans le bloc\n", FAIL_TAG);
        ok = 0;
    }
    if (block->config.spatial_ndims != 1) {
        printf("%s spatial_ndims attendu=1, obtenu=%ld\n",
               FAIL_TAG, block->config.spatial_ndims);
        ok = 0;
    }
    if (block->config.spatial_dims[0] != (long)cfg.seq_len) {
        printf("%s spatial_dims[0] attendu=%zu, obtenu=%ld\n",
               FAIL_TAG, cfg.seq_len, block->config.spatial_dims[0]);
        ok = 0;
    }
    if (!km_wavefront_plan_matches_dims(block->wavefront_plan,
                                        block->config.spatial_dims,
                                        block->config.spatial_ndims)) {
        printf("%s le plan mis en cache ne correspond pas à la forme du bloc\n",
               FAIL_TAG);
        ok = 0;
    }

    if (ok) {
        printf("%s plan 1D implicite [seq_len] créé et validé\n", PASS_TAG);
    }

    mamba_block_free(block);
    return ok;
}

static int test_block_rejects_shape_mismatch(void) {
    MBConfig cfg = {
        .dim = 8,
        .state_size = 4,
        .seq_len = 12,
        .mimo_rank = 1,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f,
        .spatial_ndims = 2,
        .spatial_dims = {3, 5}
    };
    MambaBlock *block;

    printf("\n--- mamba_block_create : rejet des formes ND incohérentes ---\n");

    block = mamba_block_create(&cfg);
    if (block) {
        printf("%s la création aurait dû échouer pour 3x5 != seq_len=12\n",
               FAIL_TAG);
        mamba_block_free(block);
        return 0;
    }

    printf("%s la forme ND invalide est rejetée proprement\n", PASS_TAG);
    return 1;
}

static int test_kmamba_propagates_topology_and_conv_config(void) {
    KMambaConfig cfg = {
        .vocab_size = 256,
        .dim = 8,
        .state_size = 4,
        .seq_len = 12,
        .n_layers = 1,
        .mimo_rank = 1,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f,
        .spatial_ndims = 2,
        .spatial_dims = {3, 4},
        .use_convnd = 1,
        .convnd_K = 3,
        .convnd_ndims = 0
    };
    KMamba *model;
    MambaBlock *block;
    int ok = 1;

    printf("\n--- kmamba_create : propagation topologie ND + convND ---\n");

    model = kmamba_create(&cfg);
    if (!model) {
        printf("%s kmamba_create a échoué pour une topologie 2D valide\n",
               FAIL_TAG);
        return 0;
    }

    block = model->layers[0];
    if (!block) {
        printf("%s aucun bloc créé dans le modèle\n", FAIL_TAG);
        kmamba_free(model);
        return 0;
    }

    if (model->cfg.spatial_ndims != 2 ||
        model->cfg.spatial_dims[0] != 3 ||
        model->cfg.spatial_dims[1] != 4) {
        printf("%s la topologie normalisée n'est pas conservée dans KMamba\n",
               FAIL_TAG);
        ok = 0;
    }
    if (block->config.spatial_ndims != 2 ||
        block->config.spatial_dims[0] != 3 ||
        block->config.spatial_dims[1] != 4) {
        printf("%s la topologie ND n'est pas propagée au bloc\n", FAIL_TAG);
        ok = 0;
    }
    if (block->config.convnd_ndims != 2) {
        printf("%s convnd_ndims attendu=2 par dérivation, obtenu=%ld\n",
               FAIL_TAG, block->config.convnd_ndims);
        ok = 0;
    }
    if (!block->convnd_kernel || !block->convnd_bias) {
        printf("%s les buffers ConvND n'ont pas été alloués\n", FAIL_TAG);
        ok = 0;
    }
    if (!km_wavefront_plan_matches_dims(block->wavefront_plan,
                                        block->config.spatial_dims,
                                        block->config.spatial_ndims)) {
        printf("%s le plan du bloc ne correspond pas à la topologie propagée\n",
               FAIL_TAG);
        ok = 0;
    }

    if (ok) {
        printf("%s KMamba propage la topologie ND et dérive convnd_ndims proprement\n",
               PASS_TAG);
    }

    kmamba_free(model);
    return ok;
}

int main(void) {
    int total = 0;
    int passed = 0;

    printf("=== Tests Topologie MambaBlock / KMamba ===\n");

    total++; passed += test_block_builds_implicit_1d_plan();
    total++; passed += test_block_rejects_shape_mismatch();
    total++; passed += test_kmamba_propagates_topology_and_conv_config();

    printf("\nRésultat: %d/%d tests passés\n", passed, total);
    return (passed == total) ? 0 : 1;
}
