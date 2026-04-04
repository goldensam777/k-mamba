<p align="center">
  <img src="figures/k_mamba_banner.svg" alt="K-Mamba banner" width="100%" />
</p>

# k-mamba

**Bibliothèque C zero-dependency pour Mamba-ND natif.**

ScanND + ConvND unifiés sous un même squelette wavefront parallèle.

[![Build](https://img.shields.io/badge/build-makefile-blue)](Makefile)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table des matières

- [Innovations](#innovations)
- [Structure](#structure)
- [Build](#build)
- [API](#api)
- [Documentation](#documentation)

---

## Innovations

### 1. Mamba-ND natif (N-dimensionnel)

Extension native de Mamba 1D vers N dimensions via **récurrence simultanée** :

```math
h(n) = Σ_{k=1}^{N} A_k · h(n − e_k) + B(n) · x(n)
```

| Opérateur | Implémentation | Wavefront | Parallélisme |
|-----------|---------------|-----------|--------------|
| **Scan 1D** | ASM AVX2 | N/A | Séquentiel |
| **Scan 2D** | ASM AVX2 | Anti-diagonale | Intra-diagonale |
| **Scan ND** | C pur | Géométrique implicite | OpenMP optionnel |
| **ConvND** | C pur | Wavefront unifié | OpenMP optionnel |

### 2. Unification Wavefront

**ScanND** et **ConvND** partagent le même squelette topologique :

- Même générateur de wavefront (`KMWavefrontPlan`)
- Même ordonnancement niveau par niveau
- Même parallélisme intra-niveau (OpenMP)

```c
// ConvND wavefront parallèle
for (long level = 0; level <= plan->max_level; level++) {
    #ifdef _OPENMP
    #pragma omp parallel for  // ← Parallélisme intra-niveau
    #endif
    for (long point = 0; point < level_size; point++) {
        // Convolution dense ND
    }
}
```

### 3. MUON natif CPU

Implémentation C pure de l'optimiseur MUON :

- Newton-Schulz (5 itérations)
- Momentum Nesterov + gradient clipping
- AdamW avec weight decay
- **Zero dependency**

### 4. Zero Dependency

- **CPU** : `gcc`, `nasm`, `libc`, `libm`
- **Build** : Makefile simple (pas CMake)
- **Kernels** : C pur inline (pas BLAS externe)
- **Optionnel** : OpenMP, CUDA

---

## Structure

```
k-mamba/
├── include/
│   ├── kmamba.h              # API publique
│   ├── kmamba_kernels.h      # Kernels inline
│   ├── km_topology.h         # Topologie ND
│   ├── wavefront_nd.h        # Générateur wavefront
│   ├── wavefront_plan.h      # Plans exécutables
│   ├── scan_nd.h             # Interface scan
│   └── convnd.h              # ConvND wavefront unifiée
├── src/
│   ├── kmamba.c              # Orchestration
│   ├── mamba_block.c         # Bloc SSM
│   ├── km_topology.c         # Topologie
│   ├── wavefront_nd.c        # Wavefront
│   ├── wavefront_plan.c      # Plans
│   ├── scan_nd.c             # Scan ND
│   └── convnd.c              # ConvND wavefront parallèle
├── kernels/                   # Kernels inline C pur
├── cpu/                       # ASM AVX2
├── cuda/                      # CUDA (optionnel)
├── Makefile
└── build.sh
```

---

## Build

### Prérequis

- `gcc >= 11`
- `nasm >= 2.15`
- `libc`, `libm`
- CPU AVX2 (Intel Haswell+ / AMD Ryzen+)

### Compilation

```bash
cd k-mamba
make
```

### Output

```
libkmamba.a   # Bibliothèque statique
```

---

## API

### Création

```c
#include <kmamba.h>

KMambaConfig cfg = {
    .vocab_size = 256,
    .dim        = 384,
    .state_size = 1024,
    .seq_len    = 128,
    .n_layers   = 1
};

KMamba *m = kmamba_create(&cfg);
kmamba_init(m, 1234);
```

### Entraînement

```c
MBOptimConfig opt = {.lr = 1e-3f, .clip_norm = 1.0f};
kmamba_enable_training(m, &opt, 1e-3f, 1e-5f);
float loss = kmamba_train_step(m, tokens_plus1);
```

### ConvND Wavefront

```c
#include <convnd.h>

ConvNDParams p = {
    .input = x, .kernel = k, .bias = b, .output = y,
    .dims = dims, .ndims = 2, .D = 64, .K = 3
};

convnd(&p, CONVND_FORWARD);   // Wavefront parallèle
```

---

## Documentation

- **[THEORY.md](THEORY.md)** — Fondement mathématique
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Volontés/Puissance
- **[AGENTS.md](AGENTS.md)** — Contexte technique

---

## Auteur

**YEVI Mawuli Peniel Samuel** — IFRI-UAC, Bénin

*Ego Sum Optimus Optimus*
