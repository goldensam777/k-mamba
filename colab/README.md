# k-mamba CUDA sur Google Colab

Entraînez votre modèle Mamba **~500M paramètres** gratuitement sur Google Colab (GPU Tesla T4).

## 🚀 Démarrage rapide

### Option 1: Notebook (Recommandé)

1. Ouvrez [Google Colab](https://colab.research.google.com/)
2. Uploadez `kmamba_colab_500M.ipynb` : Fichier → Charger le notebook
3. Activez le GPU : Runtime → Change runtime type → GPU
4. Exécutez les cellules (Runtime → Run all)

### Option 2: Script shell

```python
!wget https://raw.githubusercontent.com/goldensam777/k-mamba/main/colab/setup_colab.sh
!chmod +x setup_colab.sh && ./setup_colab.sh
```

## 📊 Configuration Colab

| Paramètre | Valeur |
|-----------|--------|
| GPU | Tesla T4 (16GB VRAM) |
| CPU | 2 cœurs Intel Xeon |
| RAM | 12 GB |
| Session max | 12 heures (peut varier) |
| Disk | 78 GB |

## 🎯 Configuration modèle (500M params)

```python
VOCAB_SIZE = 32768    # BPE 32K (tokenizer Rust)
DIM = 1024            # 1024 dimensions
STATE_SIZE = 2048     # 2048 états SSM
N_LAYERS = 24         # 24 blocs Mamba
SEQ_LEN = 1024        # Contexte 1024 tokens
BATCH_SIZE = 4        # Limite T4 16GB
```

**VRAM estimée** : ~12-14 GB

## 📁 Structure

```
colab/
├── kmamba_colab_500M.ipynb  # Notebook complet
├── setup_colab.sh           # Script de setup
└── README.md                # Ce fichier
```

## 📚 Données

### Option A: Upload manuel (recommandé)

1. Cliquez sur l'icône 📁 (fichiers) dans la barre latérale gauche
2. Faites glisser votre fichier `.txt` dans le panneau
3. Le fichier sera dans `/content/k-mamba/data/`

### Option B: Téléchargement direct

```python
!wget -O data/corpus.txt "URL_DU_CORPUS"
```

### Option C: Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
# Copier depuis Drive
!cp /content/drive/MyDrive/corpus.txt data/
```

## 💾 Sauvegarde checkpoint

### Méthode 1: Téléchargement direct

```python
from google.colab import files
files.download('kmamba_500M.bin')
```

### Méthode 2: Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
!cp kmamba_500M.bin /content/drive/MyDrive/
```

## ⏱️ Temps d'entraînement estimé (T4)

| Tokens | Temps estimé | Epochs | Checkpoints |
|--------|--------------|--------|-------------|
| 100M | ~3h | 10 | 2-3 |
| 1B | ~30h | 100 | 10-20 |
| 10B | ~300h | 1000 | 100+ |

**Note** : Colab peut interrompre les sessions longues. Sauvegardez régulièrement.

## 🔧 Troubleshooting

**Pas de GPU**
```python
# Vérifier
!nvidia-smi
# Si pas de GPU : Runtime → Change runtime type → GPU
```

**Out of memory**
- Réduire `batch_size` : 4 → 2
- Réduire `seq_len` : 1024 → 512
- Réduire `dim` : 1024 → 512

**Session interrompue**
- Utiliser Google Drive pour sauvegarder checkpoints
- Relancer et reprendre depuis le dernier checkpoint

**Build échoue**
```python
# Vérifier installations
!gcc --version
!nvcc --version
!source $HOME/.cargo/env && rustc --version
```

## 📝 Notes importantes

- **Session limit** : Colab peut interrompre après ~12h d'inactivité ou usage intensif
- **GPU limit** : Usage GPU gratuit limité (peut être indisponible temporairement)
- **Sauvegarde** : Toujours sauvegarder checkpoints sur Drive ou télécharger
- **Reconnect** : Si déconnecté, rerun les cellules depuis le début

## 🔗 Liens

- [Google Colab](https://colab.research.google.com/)
- [k-mamba GitHub](https://github.com/goldensam777/k-mamba)
- [Colab GPU FAQ](https://research.google.com/colaboratory/faq.html)
