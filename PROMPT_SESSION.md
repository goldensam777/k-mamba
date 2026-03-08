# Prompt pour Continuer le Développement Optimatrix

## 📋 **Contexte Actuel**

### **Projet** : Optimatrix - Moteur de calcul haute performance pour Mamba N-dimensionnel
### **Objectif** : Implémenter un forward pass Mamba complet avec sélectivité

### **État Actuel**
- ✅ **Structure MambaModel** : API d'entraînement simplifiée
- ✅ **Forward pass avancé** : Implémentation avec selective scan, gates, exp() approximation
- ✅ **Backward pass** : M>1 générique avec gradients complets
- ✅ **Métriques** : Accuracy, precision, recall, F1, performance
- ✅ **Tests** : BissiMamba fonctionnel, classification binaire

### **Problème Actuel**
Le forward pass Mamba complet est implémenté mais des erreurs de compilation persistent :
```
SelectiveScanParams' has no member named 'input'
```

### **Fichiers Clés**
- `src/mamba_forward.c` : Forward pass avec selective scan
- `src/mamba_training.c` : API d'entraînement intégrée
- `include/optimatrix.h` : Structures et déclarations
- `test_forward_complete.c` : Test du forward pass

## 🎯 **Prochaines Étapes à Continuer**

### **1. Corriger les erreurs de compilation**
```c
// Problème : La structure SelectiveScanParams n'a pas le champ 'input'
// Solution : Ajouter le champ manquant ou corriger les références
typedef struct {
    float *A, *B, *C, *Δ;
    float *h_prev, *h_curr, *output;
    float *input;  // Champ manquant à ajouter
    long L, D, M;
} SelectiveScanParams;
```

### **2. Finaliser le forward pass**
```c
// À vérifier :
- Approximation exp() : polynôme d'ordre 3 suffisant ?
- Gate mechanism : sigmoid vs tanh ?
- Performance : vectorisation AVX2 possible ?
- Stabilité numérique : overflow/underflow ?
```

### **3. Intégration complète**
```c
// Connecter forward et backward pass dans un entraînement cohérent
mamba_train() {
    // Forward pass
    float *output = mamba_forward_complete(model, X, seq_len);
    
    // Calcul de la loss
    float loss = compute_loss(output, y, seq_len * D);
    
    // Backward pass
    ScanBackwardMParams params = {
        // ... configuration complète
    };
    scan1d_backward_m_generic(&params);
    
    // Mise à jour des poids
    update_weights(model, gradients);
}
```

### **4. Tests avancés**
```c
// Tests de régression :
- Forward pass vs référence BissiMamba
- Backward pass vs gradients théoriques
- Entraînement complet sur données réelles (MNIST, text)
```

### **5. Optimisations**
```c
// Vectorisation AVX2 du selective scan
// Parallélisation OpenMP pour M>1
// Cache-friendly memory layout
// FMA instructions pour les calculs critiques
```

## 🚀 **Questions pour la Session Suivante**

1. **Priorité** : Corriger d'abord les erreurs de compilation du forward pass ?
2. **Architecture** : Optimiser la performance avant l'intégration complète ?
3. **Testing** : Se concentrer sur les tests unitaires ou l'intégration BissiMamba ?
4. **Documentation** : Mettre à jour README avec les nouvelles fonctionnalités ?

## 📝 **Commandes de Développement**

```bash
# Compiler le forward pass
make all
gcc -no-pie -mavx2 -I include test_forward_complete.c obj/*.o -o test_forward_complete -lm

# Tester les erreurs
./test_forward_complete

# Debug avec gdb
gdb -batch -ex run -ex bt --args ./test_forward_complete
```

## 🎯 **Objectif Final**

Créer **le moteur Mamba le plus rapide et complet** pour CPU x86-64, avec :
- Forward pass sélectif fonctionnel
- Backward pass optimisé
- Entraînement intégré
- Métriques complètes
- Tests robustes

**Prêt à continuer le développement !** 🚀
