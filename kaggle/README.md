# k-mamba CUDA sur Kaggle

Entraînez votre modèle Mamba **~500M paramètres** sur GPU Tesla T4/P100 gratuitement.

**Config** : VOCAB=32K, DIM=1024, STATE=2048, LAYERS=24, SEQ=1024

**Tokenizer** : BPE 32K (Rust)

## 🚀 Démarrage rapide

### Option 1: Notebook (Recommandé)

1. Créez un nouveau notebook sur [Kaggle](https://www.kaggle.com/code)
2. Activez le GPU : `Settings` → `Accelerator` → `GPU T4 x2`
3. Uploadez `kmamba_kaggle.ipynb`
4. Exécutez les cellules

### Option 2: Script Shell

Dans une cellule de notebook Kaggle :

```bash
!wget https://raw.githubusercontent.com/goldensam777/k-mamba/main/kaggle/setup_kaggle.sh
!chmod +x setup_kaggle.sh
!./setup_kaggle.sh
```

## 📊 Configuration Kaggle

| Paramètre | Valeur |
|-----------|--------|
| GPU | Tesla T4 (16GB) ou P100 (16GB) |
| CPU | 4 cœurs |
| RAM | 32 GB |
| Session max | 9 heures |

## 📁 Structure

```
kaggle/
├── kmamba_kaggle.ipynb    # Notebook complet
├── setup_kaggle.sh        # Script de setup
└── README.md              # Ce fichier
```

## ⚙️ Configuration modèle

Dans le notebook, ajustez ces variables :

```python
CONFIG = {
    'vocab_size': 32768,      # BPE 32K (tokenizer Rust)
    'dim': 1024,              # 1024 dimensions
    'state_size': 2048,       # 2048 états SSM
    'n_layers': 24,           # 24 blocs Mamba
    'seq_len': 1024,          # Contexte 1024 tokens
    'batch_size': 4,          # 4 séquences (limite 16GB VRAM)
    'lr': 5e-4,               # Learning rate
    'total_tokens': 10_000_000_000,  # 10B tokens (Chinchilla 500M×20)
}
```

**VRAM requise** : ~12-14 GB

## 📚 Données

### Option A: Dataset Kaggle

1. Cliquez sur `Add Data` (barre latérale droite)
2. Cherchez un corpus texte français
3. Utilisez le chemin : `/kaggle/input/NOM_DATASET/fichier.txt`

### Option B: Téléchargement direct

```python
!wget -O data/corpus.txt "URL_DU_CORPUS"
```

### Option C: Upload manuel

1. Cliquez sur `Upload` dans le panneau de droite
2. Sélectionnez votre fichier `.txt`
3. Utilisez le chemin : `/kaggle/input/...`

## 💾 Sauvegarde

Les checkpoints sont automatiquement sauvegardés dans `/kaggle/output/` pour téléchargement.

## ⏱️ Temps d'entraînement estimé

| Tokens | GPU | Temps estimé | Epochs |
|--------|-----|--------------|--------|
| 100M | T4 | ~2h | 10 |
| 1B | T4 | ~20h | 100 |
| 10B | T4 | ~200h (x20 sessions) | 1000 |

## 🔧 Troubleshooting

**Erreur: CUDA not available**
```bash
# Vérifier GPU
!nvidia-smi
```

**Build échoue**
```bash
# Vérifier dépendances
!gcc --version
!nvcc --version
!nasm --version
```

**Out of memory (OOM)**
- Réduire `batch_size` (ex: 4 → 2)
- Réduire `seq_len` (ex: 1024 → 512)
- Activer mixed precision (déjà actif dans kmamba_cuda)

## 📝 Notes

- La session Kaggle expire après 9h d'inactivité
- Sauvegardez régulièrement vos checkpoints
- Utilisez `Save Version` pour conserver le notebook

## 🔗 Liens

- [Kaggle GPUs](https://www.kaggle.com/docs/gpu)
- [k-mamba GitHub](https://github.com/goldensam777/k-mamba)
