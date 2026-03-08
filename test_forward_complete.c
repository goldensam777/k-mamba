#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "optimatrix.h"

int main() {
    printf("🚀 Test Forward Pass Mamba Complet\n");
    
    // Créer un modèle simple
    printf("\n📋 Création du modèle Mamba...\n");
    MambaModel *model = mamba_create(4, 2);  // D=4, M=2
    
    if (!model) {
        printf("❌ Erreur de création du modèle\n");
        return 1;
    }
    
    mamba_print_info(model);
    
    // Données de test simples
    printf("\n📊 Génération des données de test...\n");
    long seq_len = 8;
    float *input = malloc(seq_len * 4 * sizeof(float));
    
    // Créer une séquence avec motif
    for (long t = 0; t < seq_len; t++) {
        for (long d = 0; d < 4; d++) {
            // Motif : croissant pour chaque feature
            input[t * 4 + d] = 0.1f * (t + d + 1);
        }
    }
    
    printf("   Séquence générée : %ld timesteps × %ld features\n", seq_len, 4);
    printf("   Exemple input[0]: %.3f, input[1]: %.3f, input[2]: %.3f, input[3]: %.3f\n",
           input[0], input[1], input[2], input[3]);
    
    // Test du forward pass
    printf("\n🔮 Test du forward pass...\n");
    float *output = mamba_forward(model, input, seq_len);
    
    if (!output) {
        printf("❌ Erreur lors du forward pass\n");
        mamba_destroy(model);
        free(input);
        return 1;
    }
    
    printf("✅ Forward pass réussi !\n");
    printf("   Output[0]: %.6f\n", output[0]);
    printf("   Output[1]: %.6f\n", output[1]);
    printf("   Output[2]: %.6f\n", output[2]);
    printf("   Output[3]: %.6f\n", output[3]);
    printf("   Output[4]: %.6f\n", output[4]);
    printf("   Output[5]: %.6f\n", output[5]);
    printf("   Output[6]: %.6f\n", output[6]);
    printf("   Output[7]: %.6f\n", output[7]);
    
    // Vérifier que l'état a été mis à jour
    printf("\n📈 État caché final:\n");
    printf("   h[0]: %.6f\n", model->h[0]);
    printf("   h[1]: %.6f\n", model->h[1]);
    printf("   h[2]: %.6f\n", model->h[2]);
    printf("   h[3]: %.6f\n", model->h[3]);
    printf("   h[4]: %.6f\n", model->h[4]);
    printf("   h[5]: %.6f\n", model->h[5]);
    printf("   h[6]: %.6f\n", model->h[6]);
    printf("   h[7]: %.6f\n", model->h[7]);
    
    // Test avec une séquence plus longue
    printf("\n🔮 Test avec séquence plus longue...\n");
    long long_seq_len = 32;
    float *long_input = malloc(long_seq_len * 4 * sizeof(float));
    
    for (long t = 0; t < long_seq_len; t++) {
        for (long d = 0; d < 4; d++) {
            long_input[t * 4 + d] = 0.05f * (float)sin(t * 0.1f + d);
        }
    }
    
    float *long_output = mamba_forward(model, long_input, long_seq_len);
    if (long_output) {
        printf("✅ Forward pass long réussi !\n");
        printf("   Premier output: %.6f\n", long_output[0]);
        printf("   Dernier output: %.6f\n", long_output[(long_seq_len-1)*4]);
        free(long_output);
    }
    
    // Nettoyer
    free(input);
    free(long_input);
    free(output);
    mamba_destroy(model);
    
    printf("\n✅ Test forward pass terminé avec succès !\n");
    return 0;
}
