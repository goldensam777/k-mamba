#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "optimatrix.h"

int main() {
    printf("🚀 Test API Entraînement Intégré\n");
    
    // Créer un modèle simple
    printf("\n📋 Création du modèle Mamba...\n");
    MambaModel *model = mamba_create(4, 2);  // D=4, M=2
    
    if (!model) {
        printf("❌ Erreur de création du modèle\n");
        return 1;
    }
    
    mamba_print_info(model);
    
    // Données d'entraînement simples
    printf("📊 Génération des données d'entraînement...\n");
    long batch_size = 8;
    long seq_len = 16;
    long total_size = batch_size * seq_len * 4;  // D=4
    
    float *X_train = malloc(total_size * sizeof(float));
    float *y_train = malloc(total_size * sizeof(float));
    
    // Initialiser avec des données synthétiques
    srand(42);
    for (long i = 0; i < total_size; i++) {
        X_train[i] = (float)rand() / RAND_MAX;
        y_train[i] = X_train[i] * 0.5f + 0.1f * (float)rand() / RAND_MAX;  // Target bruité
    }
    
    // Entraînement
    printf("\n🏋‍♂️ Lancement de l'entraînement...\n");
    mamba_train(model, X_train, y_train, batch_size, seq_len, 20, 0.01f);
    
    // Prédiction
    printf("\n🔮 Test de prédiction...\n");
    float *test_input = malloc(seq_len * 4 * sizeof(float));
    for (long i = 0; i < seq_len * 4; i++) {
        test_input[i] = (float)rand() / RAND_MAX;
    }
    
    float *prediction = mamba_forward(model, test_input, seq_len);
    if (prediction) {
        printf("✅ Prédiction réussie\n");
        printf("   Input[0]: %.4f → Output[0]: %.4f\n", test_input[0], prediction[0]);
        printf("   Input[1]: %.4f → Output[1]: %.4f\n", test_input[1], prediction[1]);
        printf("   Input[2]: %.4f → Output[2]: %.4f\n", test_input[2], prediction[2]);
        free(prediction);
    }
    
    // Test de sauvegarde/chargement
    printf("\n💾 Test de sauvegarde/chargement...\n");
    mamba_save(model, "test_model.bin");
    
    MambaModel *loaded_model = mamba_load("test_model.bin");
    if (loaded_model) {
        printf("✅ Modèle chargé avec succès\n");
        printf("   D: %ld, M: %ld\n", loaded_model->D, loaded_model->M);
        mamba_destroy(loaded_model);
    }
    
    // Nettoyer
    printf("\n🧹 Nettoyage...\n");
    free(X_train);
    free(y_train);
    free(test_input);
    mamba_destroy(model);
    
    printf("✅ Test terminé avec succès !\n");
    return 0;
}
