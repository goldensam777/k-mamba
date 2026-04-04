# AGENTS.md — k-mamba

> Lis ce fichier avant de toucher au code. C'est le contexte technique et philosophique du projet.

---

## Qu'est-ce que k-mamba ?

Une **bibliothèque C zero-dependency** pour State Space Models Mamba en dimensions N.

**Philosophie** : Aucune dépendance externe — juste `gcc`, `nasm` et `libc/libm`.

**Innovations** : 
1. **Scan Mamba-ND natif** (N-dimensionnel) avec ordonnancement wavefront
2. **Générateur de wavefront ND** — primitive topologique générique réutilisable par scanND et convND
3. **Couche topologique commune** — normalisation des dimensions, strides, indexation ND
4. **Kernels inline** — GEMM, activations, MUON/AdamW en C pur (pas de BLAS externe)

---

## Architecture (zero-dependency)

```
k-mamba/
├── include/
│   ├── kmamba.h              # API publique
│   ├── kmamba_kernels.h      # Kernels zero-dependency (GEMM, activations, optimizers)
│   ├── km_topology.h         # Couche topologique ND
│   ├── wavefront_nd.h        # Générateur wavefront ND
│   ├── wavefront_plan.h      # Plans exécutables
│   ├── scan_nd.h             # Interface scan ND
│   └── convnd.h              # Interface convND
├── src/
│   ├── kmamba.c              # Forward, backward, training loop
│   ├── mamba_block.c         # Bloc SSM (projections, scan dispatch)
│   ├── km_topology.c         # Normalisation topologique
│   ├── wavefront_nd.c        # Générateur wavefront
│   ├── wavefront_plan.c      # Plans wavefront
│   ├── scan_nd.c             # Scan ND
│   └── convnd.c              # Convolution ND
├── kernels/                  # Kernels compute zero-dependency
│   ├── gemm_f32.c            # GEMM/GEMV en C pur
│   ├── activations_f32.c     # SiLU, ReLU, Sigmoid, Softplus
│   ├── elementwise_f32.c     # Hadamard, vector ops
│   ├── optimizer_f32.c       # Gradient clip, Newton-Schulz, MUON, AdamW
│   └── init_f32.c            # Xavier/Kaiming init
├── cpu/                      # Scan SSM en assembleur
│   ├── scan1d.asm            # Scan 1D AVX2
│   ├── scan2d.asm            # Scan 2D wavefront
│   └── mamba_scan.c          # Dispatch CPU
├── cuda/                     # Scan SSM CUDA (optionnel)
│   ├── scan1d.cu
│   └── mamba_scan.cu
├── Makefile                  # Build simple (pas de CMake)
└── build.sh                  # Script build style Karpathy
```

---

## Build (Zero Dependency)

```bash
# Build simple
make

# Ou avec le script
./build.sh

# Clean
make clean
```

**Requiert** : `gcc >= 11`, `nasm >= 2.15`, `libc`, `libm`

**Pas de** : CMake, OpenBLAS, optimatrix, Python, PyTorch

---

## Kernels Inline

Tous les kernels sont maintenant dans `kernels/` — C pur sans dépendances :

| Opération | Fichier | Fonction |
|-----------|---------|----------|
| GEMM | `gemm_f32.c` | `gemm_f32()`, `gemv_f32()` |
| Activations | `activations_f32.c` | `silu_f32()`, `relu_f32()` |
| Elementwise | `elementwise_f32.c` | `hadamard_f32()`, `vec_add_f32()` |
| Optimizers | `optimizer_f32.c` | `adamw_step_f32()`, `muon_update_mat_f32()` |
| Init | `init_f32.c` | `init_xavier_uniform_f32()` |

---

## API

```c
#include <kmamba.h>

// Création
KMambaConfig cfg = {
    .vocab_size = 256, .dim = 384, .state_size = 1024,
    .seq_len = 128, .n_layers = 1
};
KMamba *m = kmamba_create(&cfg);

// Forward
kmamba_forward(m, tokens, logits_out);

// Training
MBOptimConfig opt = {.lr = 1e-3f, .clip_norm = 1.0f};
kmamba_enable_training(m, &opt, 1e-3f, 1e-5f);
float loss = kmamba_train_step(m, tokens_plus1);
```

---

## Séparation Volontés/Puissance

| Couche | Rôle | Localisation |
|--------|------|--------------|
| **Volontés** | Orchestration modèle | `src/kmamba.c`, `src/mamba_block.c` |
| **Topologie** | ND indexing, wavefront | `src/km_topology.c`, `src/wavefront_*.c` |
| **Puissance** | Kernels compute | `kernels/*.c`, `cpu/*.asm` |

---

## Conventions

1. **Zero dependency** — Pas de bibliothèque externe, pas de package manager
2. **Inline kernels** — Fonctions simples en C, pas de BLAS complexe
3. **Makefile simple** — 20 lignes, pas de CMake
4. **NASM + C** — Assembleur pour hot paths, C pour le reste

---

## Ce qu'il ne faut PAS faire

- Ajouter des dépendances externes (OpenBLAS, MKL, etc.)
- Utiliser CMake ou autre build system complexe
- Créer des abstractions prématurées
- Dépendre de Python ou PyTorch

---

## Auteur

**YEVI Mawuli Peniel Samuel** — IFRI-UAC (Bénin)

Devise : **"Ego Sum Optimus Optimus"**
