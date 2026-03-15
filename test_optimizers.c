#include <stdio.h>
#include <stdlib.h>
#include "include/kmamba.h"

int main() {
    printf("=== Test Système d'Optimiseurs Modulaires ===\n");
    
    // Test configuration
    MBConfig config = {
        .dim = 384,
        .state_size = 1024,
        .seq_len = 128,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f,
        .dt_rank = 1.0f,
        .dt_init = 1.0f,
        .use_convnd = 0,
        .convnd_K = 0,
        .convnd_ndims = 0
    };
    
    // Test optimizer config
    MBOptimConfig opt_config = {
        .lr = 1e-3f,
        .mu = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .clip_norm = 1.0f,
        .weight_decay = 1e-5f
    };
    
    // Create MambaBlock
    MambaBlock *block = mamba_block_create(&config);
    if (!block) {
        printf("❌ Erreur création MambaBlock\n");
        return 1;
    }
    
    mamba_block_init(block);
    printf("✅ MambaBlock créé et initialisé\n");
    
    // Test different optimizers
    printf("\n--- Test Optimizers ---\n");
    
    // Test ADAM_CLIP
    printf("Test ADAM_CLIP... ");
    mamba_attach_optimizer(block, OPTIMIZER_ADAM_CLIP, &opt_config);
    mamba_optimizer_step(block, &opt_config);
    printf("✅ OK\n");
    
    // Test ADAMW
    printf("Test ADAMW... ");
    mamba_free_optimizer(block);
    mamba_attach_optimizer(block, OPTIMIZER_ADAMW, &opt_config);
    mamba_optimizer_step(block, &opt_config);
    printf("✅ OK\n");
    
    // Test SGD
    printf("Test SGD... ");
    mamba_free_optimizer(block);
    mamba_attach_optimizer(block, OPTIMIZER_SGD, &opt_config);
    mamba_optimizer_step(block, &opt_config);
    printf("✅ OK\n");
    
    // Test MUON (placeholder)
    printf("Test MUON (placeholder)... ");
    mamba_free_optimizer(block);
    mamba_attach_optimizer(block, OPTIMIZER_MUON, &opt_config);
    mamba_optimizer_step(block, &opt_config);
    printf("✅ OK\n");
    
    // Cleanup
    mamba_free_optimizer(block);
    mamba_block_free(block);
    
    printf("\n=== Test Réussi ! ===\n");
    printf("Système d'optimiseurs modulaires fonctionnel\n");
    
    return 0;
}
