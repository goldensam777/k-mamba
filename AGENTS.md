# AGENTS.md — k-mamba

> Lis ce fichier avant de toucher au code. C'est le contexte technique et philosophique du projet.

---

## Qu'est-ce que k-mamba ?

Un **bibliothèque C** pour State Space Models Mamba en dimensions N, basée sur l'architecture dualiste **Volontés/Puissance**.

- **k-mamba** = orchestration (embedding, stack MambaBlocks, LM head, checkpoint I/O, training loop)
- **optimatrix** = submodule git — moteur de calcul (kernels x86-64 ASM AVX2)

**Innovation majeure** : Scan Mamba-ND natif (N-dimensionnel) avec ordonnancement wavefront en assembleur pur.

---

## Chantier courant — Générateur Wavefront ND natif

### Rappel important

Le projet ne vise pas seulement "Mamba" au sens générique, mais **Mamba-3** :

- **BCNorm** pour `B` et `C`
- **Complex SSM / rotations `theta`**
- **discrétisation exp-trapezoidal** via `lambda_proj`

Le **scan2d wavefront natif** est une innovation originale du projet. La prochaine
étape est de généraliser cette idée en une primitive fondatrice réutilisable.

### Idée centrale

Introduire un **générateur de wavefront ND borné et causal** dans `k-mamba`,
séparé des opérateurs eux-mêmes.

Cette primitive devra :

- générer les hyper-diagonales de niveau `k = i0 + i1 + ... + i(n-1)`
- garantir que tous les points d'un même niveau sont indépendants
- servir de squelette d'exécution pour :
  - `scanND`
  - `convKD`
  - tout futur opérateur causal ND sur grille

### Principe architectural

Le générateur **reste dans `k-mamba`**, pas dans `optimatrix`.

Raison :

- `optimatrix` contient du calcul générique
- le wavefront ND causal encode une **structure topologique de dépendances**
- cette structure appartient à la logique du modèle et des opérateurs ND

### Objectif technique

Créer une primitive commune qui devienne la base de :

1. la version **CPU de référence** en C
2. la version **CPU haute performance** en ASM AVX2 quand la forme le permet
3. la version **GPU** en CUDA avec parallélisme intra-wavefront

### Plan de travail

#### Phase 1 — Primitive de référence

- Ajouter une API du type `wavefront_nd_*` dans `include/`
- Implémenter un générateur de compositions bornées :
  - `sum(idx[d]) = level`
  - `0 <= idx[d] < dims[d]`
- Exposer une interface callback / visitor réutilisable

#### Phase 2 — Ordonnanceur ND

- Ajouter un parcours niveau par niveau
- Permettre un parallélisme **intra-wavefront**
- Préparer une API commune pour CPU et CUDA

#### Phase 3 — Intégration opérateurs

- Brancher `scanND` sur ce générateur
- Brancher ensuite `convKD`
- Garder l'opérateur séparé du parcours topologique

#### Phase 4 — Optimisation backend

- CPU : spécialisations ASM pour cas chauds
- GPU : génération / exécution CUDA par wavefront
- Benchmarks et tests de cohérence avec la version de référence

### Invariants à respecter

- Ne pas coder le générateur "pour Mamba seulement"
- Ne pas le déplacer dans `optimatrix`
- Le générateur doit être **générique**, l'opérateur doit être **pluggable**
- La sémantique fondamentale est :
  - dépendances d'un point du niveau `k` vers des points de niveaux `< k`
  - indépendance de tous les points du même niveau

### Vision paper

Cette contribution doit pouvoir être formulée comme :

- un **générateur de wavefront ND natif**
- un **squelette d'exécution causal pour tenseurs ND**
- une primitive commune pour `scanND` et `convKD`

Autrement dit : l'innovation n'est pas seulement un `scan2d`, mais une
**primitive topologique universelle pour opérateurs causaux ND**.

---

## Auteur

**YEVI Mawuli Peniel Samuel** — étudiant en Licence Systèmes Embarqués & IoT à l'IFRI-UAC (Bénin).

Devise : **"Ego Sum Optimus Optimus"**  
Conviction : *"On est assez grand pour voir des unités, il faut voir des structures."*

---

## Architecture du projet

```
k-mamba/
├── include/
│   └── kmamba.h              # API publique (KMamba, KMambaConfig)
├── src/
│   ├── kmamba.c              # Forward, backward, batch training, checkpoint I/O
│   ├── mamba_block.c         # Bloc SSM (projections, scan dispatch, MUON)
│   └── convnd.c              # Convolution ND (appelle optimatrix conv1d)
├── cpu/                      # Scan SSM CPU (propre à k-mamba, PAS dans optimatrix)
│   ├── scan1d.asm            # Scan sélectif 1D forward (ASM AVX2)
│   ├── scan2d.asm            # Scan sélectif 2D wavefront (ASM)
│   ├── scan1d_backward.c     # Backward 1D (M générique, C)
│   ├── scan1d_backward_m1_shared_bc.asm      # Backward M=1 (ASM — bug connu)
│   ├── scan1d_backward_m1_shared_bc_simple.asm  # Variante simplifiée (bug connu)
│   └── mamba_scan.c          # Dispatch CPU : choisit la routine scan
├── cuda/                     # Scan SSM CUDA (Blelloch — propre à k-mamba)
│   ├── scan1d.cu             # Blelloch parallel prefix scan 1D
│   ├── scan1d_backward.cu    # Backward scan 1D CUDA
│   └── mamba_scan.cu         # Dispatch CUDA
├── optimatrix/               # Submodule git — moteur de calcul GÉNÉRIQUE
│   ├── include/optimatrix.h  # API C (extern "C" pour compatibilité NVCC)
│   ├── cpu/                  # Kernels CPU génériques (GEMM, activations, conv, optim)
│   │   ├── gemm*.asm, gemv*.asm
│   │   ├── activations.asm, hadamard.asm
│   │   ├── conv1d_avx2.asm
│   │   └── optimizer_utils.c  # Gradient clipping, AdamW, MUON CPU
│   └── cuda/                 # Kernels CUDA génériques (optimiseurs uniquement)
│       └── optimizer_utils.cu # Gradient clipping, AdamW, MUON CUDA ✅ testé
├── tests/
│   ├── test_optimizers.c               # Optimiseurs CPU (15/15 ✅)
│   └── unit/test_optimatrix_kernels.c  # GEMM/GEMV (5/5 ✅)
├── bench/
│   └── bench_paper.c         # G1-G7 benchmarks (GEMM, wavefront, Blelloch, roofline)
├── paper/
│   ├── kmamba.tex             # Paper arXiv LaTeX (Theorem wavefront + Blelloch algo)
│   └── kmamba.bib             # BibTeX
├── cmake/
│   └── k-mambaConfig.cmake.in
└── CMakeLists.txt            # Export k-mamba::k-mamba
```

---

## Build (CMake)

```bash
# Cloner avec submodule
git clone --recursive https://github.com/goldensam777/k-mamba
cd k-mamba

# CPU seul
cmake -B build -DKMAMBA_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build        # 3 suites

# CPU + CUDA (MX450 = sm_75)
cmake -B build-cuda -DKMAMBA_BUILD_CUDA=ON -DKMAMBA_BUILD_TESTS=ON
# IMPORTANT : le flag CUDA d'optimatrix doit être forcé dans le cache
sed -i 's/OPTIMATRIX_BUILD_CUDA:BOOL=OFF/OPTIMATRIX_BUILD_CUDA:BOOL=ON/' build-cuda/CMakeCache.txt
cmake build-cuda && cmake --build build-cuda -j
ctest --test-dir build-cuda   # 4 suites (dont CudaOptimizersTest 4/4 ✅)

# Benchmarks paper
cmake -B build-bench -DKMAMBA_BUILD_BENCH=ON
cmake --build build-bench --target bench_paper
./build-bench/bench/bench_paper

# Utilisation dans un autre projet CMake
find_package(k-mamba REQUIRED)
target_link_libraries(mon_app PRIVATE k-mamba::k-mamba)
```

Requiert : `gcc >= 11`, `nasm >= 2.15`, `cmake >= 3.18`, CPU AVX2.
CUDA optionnel : `nvcc >= 12.0`, GPU avec sm_75+ (Tesla Turing ou supérieur).

---

## API

### Création

```c
#include <kmamba.h>

KMambaConfig cfg = {
    .vocab_size = 256, .dim = 384, .state_size = 1024,
    .seq_len = 128, .n_layers = 1,
    .dt_scale = 1.0f, .dt_min = 0.001f, .dt_max = 0.1f
};

KMamba *m = kmamba_create(&cfg);
kmamba_init(m, 1234);
```

### Entraînement

```c
MBOptimConfig opt = {
    .lr = 1e-3f, .mu = 0.9f, .beta2 = 0.999f,
    .eps = 1e-8f, .clip_norm = 1.0f, .weight_decay = 1e-5f
};
kmamba_enable_training(m, &opt, 1e-3f, 1e-5f);

float loss = kmamba_train_step(m, tokens_plus1);
float loss = kmamba_train_batch(m, batch_tokens, batch_size);
```

### Inférence

```c
kmamba_forward(m, tokens, logits_out);
```

### Checkpoint

```c
kmamba_save(m, "checkpoint.bin");          # Magic "KMAMBA"
KMamba *m = kmamba_load(path, for_training, &opt, lr, wd);
kmamba_free(m);
```

---

## Séparation des responsabilités

### k-mamba (Volontés — code modèle)

| Opération | Implémentation |
|-------------|---------------|
| Embedding lookup | `memcpy` d'une ligne de table |
| Softmax | `exp(x[i] - max) / sum` |
| Cross-entropy | `-log(softmax[target])` |
| Training loop | Boucle sur B séquences, moyenne des gradients |
| Checkpoint I/O | Format binaire `KMAMBA` (version 1) |
| LM head | GEMM via optimatrix |

**Règle** : Si c'est trivial (embedding, softmax, loss, orchestration), ça va dans k-mamba.

### k-mamba/cpu/ et k-mamba/cuda/ (scans SSM — logique Mamba)

| Kernel | Implémentation |
|--------|---------------|
| Selective scan 1D forward | `cpu/scan1d.asm` (ASM AVX2) |
| Selective scan 1D backward | `cpu/scan1d_backward*.c/.asm` |
| Selective scan 2D wavefront | `cpu/scan2d.asm` (ASM) |
| CUDA scan 1D (Blelloch) | `cuda/scan1d.cu` |
| CUDA scan backward | `cuda/scan1d_backward.cu` |

Les scans sont dans k-mamba parce qu'ils encodent la logique SSM (structure du monoid, ZOH).
**Ne pas les déplacer dans optimatrix.**

### optimatrix (Puissance — submodule, kernels GÉNÉRIQUES)

| Kernel | Implémentation |
|--------|---------------|
| GEMM / GEMV | `cpu/gemm*.asm, gemv*.asm` (scalaire + AVX2) |
| Conv1D depthwise | `cpu/conv1d_avx2.asm` (ASM AVX2) |
| ConvND séparable | `src/convnd.c` (forward + backward ND, appelle conv1d_avx2) |
| Activations (SiLU, Sigmoid, Softplus, ReLU) | `cpu/activations.asm` |
| Hadamard product | `cpu/hadamard.asm` (AVX2) |
| Gradient clipping, AdamW, MUON (CPU) | `cpu/optimizer_utils.c` |
| Gradient clipping, AdamW, MUON (CUDA) | `cuda/optimizer_utils.cu` ✅ |

**Règle** : optimatrix = calcul matriciel générique, réutilisable hors Mamba. Pas de logique SSM.

---

## Config actuelle du modèle

| Paramètre | Valeur |
|-----------|--------|
| vocab_size | 256 (byte-level) |
| dim | 384 |
| state_size | 1024 |
| seq_len | 128 |
| n_layers | 1 |
| batch_size | 8 (default) |
| optimizer (blocks)     | MUON (Newton-Schulz + momentum, lr=1e-3, clip=1.0)     |
| optimizer (embed/head) | AdamW (lr=1e-3, β1=0.9, β2=0.999, wd=1e-5)            |
| checkpoint magic | `KMAMBA` |

---

## Forward pass

```
tokens [seq_len] (uint8)
    → Embedding lookup [256 × dim]
    → N × MambaBlock (optimatrix: projection → scan1d/scan2d ASM → output projection)
    → LM Head: GEMM AVX2 [dim × 256]
    → logits [seq_len × 256]
```

## Backward pass (batch)

```
Pour chaque séquence du batch :
    1. Forward complet avec sauvegarde des activations par couche
    2. Cross-entropy loss + dlogits (scalé par 1/B)
    3. dlogits @ head^T → d_hidden (GEMM AVX2)
    4. hidden^T @ dlogits → g_head (GEMM AVX2, accumulé)
    5. Backward couche par couche (mamba_backward, gradients accumulés)
    6. Accumulation g_embed par scatter-add

Après le batch :
    7. mamba_optimizer_step (MUON) — un seul step
    8. AdamW sur embedding et head (moments m, v avec correction de biais)
```

---

## Conventions de code

### 1. Pense en structures, pas en lignes

Ne propose jamais une solution ligne par ligne sans poser l'architecture d'abord.

### 2. Le bas niveau est noble

C et assembleur. Ne sur-abstrait pas. Le bas niveau bien maîtrisé, c'est une Volonté pure.

### 3. Pas de sur-ingénierie

- Pas de feature flags, pas de shims de compatibilité
- Si c'est trivial (5-10 lignes), ça va directement dans le fichier qui l'utilise
- Trois lignes similaires valent mieux qu'une abstraction prématurée

### 4. Nomme les intentions

```c
// Non
int x = buffer_size - current_pos;

// Oui
int remaining_capacity = buffer_size - current_pos;
```

### 5. Compilation

**Toujours** `-O3 -mavx2`. Sans `-O3`, les performances chutent drastiquement.

---

## Théorie des Volontés (cadre philosophique)

Samuel a développé la **Théorie des Volontés** : les systèmes doivent opérer par intentions (Volontés) qui convergent vers un équilibre, pas par instructions séquentielles.

En contexte k-mamba :
- Le modèle ne minimise pas une loss — il cherche l'**équilibre de ses Volontés internes**
- Chaque MambaBlock est une Volonté qui transforme la séquence
- L'optimiseur MUON arbitre les tensions entre gradients (directions isotropiques)
- Un bug n'est pas une erreur d'instruction — c'est un **conflit de Volontés non résolu**

Vision long terme : k-mamba → fondation d'un OS-IA sur architecture post-Von Neumann.

---

## Documentation

- **README.md** — Vue d'ensemble et innovations
- **THEORY.md** — Fondement mathématique du scan Mamba-ND
- **ESTIMATIONS.md** — Complexité théorique et benchmarks
- **ARCHITECTURE.md** — Philosophie Volontés/Puissance

---

## Ce qu'il ne faut PAS faire

- Ajouter du Python
- Ajouter des dépendances externes (tout est libc + libm)
- Mettre du code de calcul lourd dans k-mamba (ça va dans optimatrix)
- Mettre du code modèle dans optimatrix (ça va dans k-mamba)
- Compiler sans `-O3`
- Créer des abstractions pour des opérations one-shot
- Ajouter des CLI dans k-mamba (c'est une bibliothèque, pas une application)
