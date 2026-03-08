#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "optimatrix.h"

// Générer des données synthétiques pour classification binaire
void generate_binary_classification_data(float **X_train, float **y_train, 
                                   float **X_test, float **y_test,
                                   long train_size, long test_size, 
                                   long seq_len, long D) {
    long total_train = train_size * seq_len * D;
    long total_test = test_size * seq_len * D;
    
    // Allouer
    *X_train = malloc(total_train * sizeof(float));
    *y_train = malloc(train_size * seq_len * sizeof(float));
    *X_test = malloc(total_test * sizeof(float));
    *y_test = malloc(test_size * seq_len * sizeof(float));
    
    // Générer des données avec motif simple
    srand(123);  // Seed différent pour test
    
    // Données d'entraînement
    for (long i = 0; i < train_size; i++) {
        for (long t = 0; t < seq_len; t++) {
            for (long d = 0; d < D; d++) {
                long idx = i * seq_len * D + t * D + d;
                // Créer des motifs : si d=0 et t>5, alors classe 1
                if (d == 0 && t > seq_len/2) {
                    (*X_train)[idx] = 0.8f + 0.1f * (float)rand() / RAND_MAX;
                    (*y_train)[i * seq_len + t] = 1.0f;  // Classe positive
                } else {
                    (*X_train)[idx] = 0.2f + 0.1f * (float)rand() / RAND_MAX;
                    (*y_train)[i * seq_len + t] = 0.0f;  // Classe négative
                }
            }
        }
    }
    
    // Données de test (légèrement différentes)
    for (long i = 0; i < test_size; i++) {
        for (long t = 0; t < seq_len; t++) {
            for (long d = 0; d < D; d++) {
                long idx = i * seq_len * D + t * D + d;
                if (d == 0 && t > seq_len/2) {
                    (*X_test)[idx] = 0.85f + 0.1f * (float)rand() / RAND_MAX;
                    (*y_test)[i * seq_len + t] = 1.0f;
                } else {
                    (*X_test)[idx] = 0.25f + 0.1f * (float)rand() / RAND_MAX;
                    (*y_test)[i * seq_len + t] = 0.0f;
                }
            }
        }
    }
}

int main() {
    printf("🧪 Test Classification Binaire avec Mamba Training\n");
    
    // Paramètres du test
    long D = 8;           // Dimension des features
    long M = 2;           // Dimension de l'état
    long seq_len = 16;     // Longueur de séquence
    long train_size = 100; // Taille du training set
    long test_size = 50;   // Taille du test set
    int epochs = 30;       // Nombre d'époques
    float lr = 0.01f;     // Learning rate
    
    printf("📊 Configuration:\n");
    printf("   Features: D=%ld\n", D);
    printf("   État caché: M=%ld\n", M);
    printf("   Séquence: %ld\n", seq_len);
    printf("   Training: %ld échantillons\n", train_size);
    printf("   Test: %ld échantillons\n", test_size);
    printf("   Époques: %d\n", epochs);
    printf("   Learning rate: %.3f\n\n", lr);
    
    // Générer les données
    float *X_train, *y_train, *X_test, *y_test;
    generate_binary_classification_data(&X_train, &y_train, &X_test, &y_test,
                                   train_size, test_size, seq_len, D);
    
    printf("📈 Données générées:\n");
    printf("   Classes: 0 (négative), 1 (positive)\n");
    printf("   Motif: feature[0] > seuil quand temps > mi-séquence\n");
    printf("   Distribution: ~50%% chaque classe\n\n");
    
    // Créer et entraîner le modèle
    printf("\n🚀 Création et entraînement du modèle...\n");
    MambaModel *model = mamba_create(D, M);
    if (!model) {
        printf("❌ Erreur de création du modèle\n");
        return 1;
    }
    
    // Entraîner
    mamba_train(model, X_train, y_train, train_size, seq_len, epochs, lr);
    
    // Marquer comme entraîné
    model->is_trained = 1;
    model->best_epoch = epochs - 1;  // Dernière epoch
    
    printf("\n📊 Évaluation sur le jeu de test...\n");
    mamba_evaluate(model, X_test, y_test, test_size, seq_len);
    
    // Test avec quelques exemples
    printf("\n🔮 Exemples de prédictions:\n");
    for (int i = 0; i < 5; i++) {
        float *input = &X_test[i * seq_len * D];
        float *true_labels = &y_test[i * seq_len];
        float *predictions = mamba_forward(model, input, seq_len);
        
        if (predictions) {
            printf("   Échantillon %d:\n", i + 1);
            printf("     Input[0]: %.3f → Pred: %.3f (True: %.1f)\n", 
                   input[0], predictions[0], true_labels[0]);
            printf("     Input[7]: %.3f → Pred: %.3f (True: %.1f)\n", 
                   input[7], predictions[7], true_labels[7]);
            printf("     Input[15]: %.3f → Pred: %.3f (True: %.1f)\n", 
                   input[15], predictions[15], true_labels[15]);
            
            free(predictions);
        }
    }
    
    // Sauvegarder le modèle
    printf("\n💾 Sauvegarde du modèle...\n");
    mamba_save(model, "bissimamba_model.bin");
    
    // Afficher les métriques finales
    printf("\n🏆 Résultats finaux:\n");
    mamba_print_metrics(model);
    
    // Nettoyer
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);
    mamba_destroy(model);
    
    printf("\n✅ Test classification binaire terminé !\n");
    return 0;
}
