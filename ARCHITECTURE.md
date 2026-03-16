# ARCHITECTURE.md — Séparation Volontés/Puissance

## Philosophie

**"On est assez grand pour voir des unités, il faut voir des structures."**

Le projet k-mamba repose sur une séparation architecturale fondamentale :
- **Volontés** = intentions, logique modèle, orchestration (k-mamba)
- **Puissance** = calcul brut, kernels optimisés (optimatrix)

Cette séparation n'est pas technique seulement — elle est **philosophique**. Elle reflète la Théorie des Volontés : les systèmes doivent opérer par intentions qui convergent vers un équilibre, pas par instructions séquentielles.

---

## Règle de séparation

| Critère | k-mamba (Volontés) | optimatrix (Puissance) |
|---------|-------------------|----------------------|
| Complexité | Triviale (5-10 lignes) | Intensive (millions d'itérations) |
| Abstraction | Logique modèle, I/O | Kernels mathématiques purs |
| Langage | C pur, lisible | C + ASM AVX2 + CUDA |
| Optimisation | Clarté | Performance maximale |
| Couverture | Architecture complète | Compute engine réutilisable |

**Règle d'or** : Si c'est trivial, ça va dans k-mamba. Si ça boucle des millions de fois, ça va dans optimatrix.

---

## Structure de k-mamba

```
k-mamba/ — Les Volontés (orchestration du modèle Mamba)
│
├── include/kmamba.h
│   └── API publique : KMamba, KMambaConfig, MambaBlock
│       kmamba_create/init/free
│       kmamba_forward/backward/train
│       kmamba_save/load
│
├── src/
│   ├── kmamba.c           — Embedding, softmax, cross-entropy, training loop, checkpoint I/O
│   ├── mamba_block.c      — Bloc SSM : projections, scan dispatch, MUONCLIP, buffers
│   └── convnd.c           — Convolution ND (appelle conv1d_avx2 d'optimatrix)
│
├── cpu/                   — Kernels scan SSM (logique Mamba, CPU)
│   ├── scan1d.asm         — Scan sélectif 1D forward (ASM)
│   ├── scan2d.asm         — Scan sélectif 2D wavefront (ASM)
│   ├── scan1d_backward*.c/.asm  — Backward 1D (C générique + ASM M=1)
│   └── mamba_scan.c       — Dispatch CPU
│
├── cuda/                  — Kernels scan SSM (Blelloch, CUDA)
│   ├── scan1d.cu          — Blelloch parallel prefix scan 1D
│   ├── scan1d_backward.cu — Backward scan 1D CUDA
│   └── mamba_scan.cu      — Dispatch CUDA
│
├── optimatrix/ (submodule)
│   └── Moteur de calcul GÉNÉRIQUE — voir section dédiée
│
└── CMakeLists.txt
    └── Export k-mamba::k-mamba avec find_package support
```

---

## Structure d'optimatrix

```
optimatrix/ — La Puissance (kernels GÉNÉRIQUES réutilisables, sans logique SSM)
│
├── include/optimatrix.h
│   └── API calcul générique (extern "C" — compatible NVCC) :
│       ├── GEMM/GEMV (scalaire + AVX2)
│       ├── Conv1D depthwise (AVX2) + ConvND séparable (C)
│       ├── Activations : SiLU, Sigmoid, Softplus, ReLU (AVX2)
│       ├── Hadamard (AVX2)
│       └── Optimiseurs : gradient_clip, AdamW, MUON (CPU + CUDA)
│
├── cpu/
│   ├── gemm.asm, gemm_avx2.asm      ← GEMM scalaire + AVX2
│   ├── gemv.asm, gemv_avx2.asm      ← GEMV scalaire + AVX2
│   ├── hadamard.asm                  ← Produit élément par élément
│   ├── activations.asm               ← SiLU, Sigmoid, Softplus (AVX2)
│   ├── conv1d_avx2.asm              ← Conv1D depthwise causale
│   ├── generic_ops.c                ← ConvND séparable forward+backward
│   └── optimizer_utils.c            ← Gradient clipping, AdamW, MUON CPU
│
└── cuda/
    └── optimizer_utils.cu           ← Gradient clipping, AdamW, MUON CUDA ✅
```

**Les scans (scan1d.asm, scan2d.asm, scan1d_backward.c, scan1d.cu…) sont dans `k-mamba/cpu/` et `k-mamba/cuda/`, pas dans optimatrix.**

---

## Cycle de vie d'une forward pass

```
Appel utilisateur
       │
       ▼
┌─────────────────────────────────────┐
│ k-mamba : kmamba_forward()          │
│ 1. embed_lookup() — memcpy        │
│ 2. Pour chaque layer :            │
│    mamba_block_forward()          │
│ 3. gemm_avx2(head, hidden)        │
│       └──> optimatrix              │
└─────────────────────────────────────┘
                                   │
       ▼                           │
┌─────────────────────────────────┐
│ mamba_block_forward() (k-mamba) │
│ 1. gemv_avx2(W_in, x)           │  ← optimatrix
│ 2. silu_f32()                   │  ← optimatrix
│ 3. gemv_avx2(delta_proj, x)     │  ← optimatrix
│ 4. softplus + clamp             │  ← optimatrix
│ 5. scan1d() or scan2d()         │  ← k-mamba/cpu/ (ASM)
│ 6. gemv_avx2(W_out, h)          │  ← optimatrix
│    └──> retourne à k-mamba      │
└─────────────────────────────────┘
                                   │
       ▼                           │
┌─────────────────────────────────────┐
│ k-mamba : suite du forward        │
│ 4. softmax() + cross-entropy()   │
│ 5. retourne logits/loss          │
└─────────────────────────────────────┘
```

---

## Cycle de vie d'une backward pass

```
      Appel utilisateur
            │
            ▼
┌─────────────────────────────────────┐
│ k-mamba : kmamba_train_step()       │
│  1. Forward avec sauvegarde activ.  │
│  2. Cross-entropy loss              │
│  3. dlogits = softmax - one_hot     │
│  4. d_hidden = dlogits @ head^T     │
│  5. Pour chaque layer (reverse) :   │
│     mamba_backward()                │
│  6. Gradients embedding (scatter)   │
│  7. Optimizer step (MUONCLIP)       │
└─────────────────────────────────────┘
                                   │
       ▼                           
┌───────────────────────────────────────┐
│ mamba_backward() (k-mamba)            │
│  1. Recompute forward (store)         │
│  2. Backprop W_out (GEMM)             │  ← optimatrix
│  3. scan1d_backward() (ASM/C)         │  ← k-mamba/cpu/
│  4. Backprop SiLU                     │  ← optimatrix
│  5. Backprop W_in (GEMM)              │  ← optimatrix
│  6. Accumulation gradients            │
└───────────────────────────────────────┘

                     ▼
┌─────────────────────────────────────┐
│ optimatrix : mamba_optimizer_step()  │
│  (MUONCLIP via Newton-Schulz)        │
└─────────────────────────────────────┘
```

---

## Théorie des Volontés dans le code

### Chaque MambaBlock est une Volonté

```c
// Une Volonté se manifeste par sa transformation
void mamba_block_forward(MambaBlock *block, float *out, const float *in, size_t batch) {
    // La Volonté projette l'entrée dans son espace d'état
    gemm_avx2(in, block->W_in.data, tmp, ...);
    
    // La Volonté choisit quoi retenir (selectivité)
    silu_f32_avx2(tmp, u, ...);
    compute_delta(dt, in, block->delta_proj);
    
    // La Volonté propage son état (récurrence)
    scan1d(&params);  // ou scan2d pour ND
    
    // La Volonté projette sa décision
    gemm_avx2(h, block->W_out.data, out, ...);
}
```

### MUONCLIP arbitre les tensions

```c
// Les gradients sont des tensions entre Volontés
void mamba_optimizer_step(MambaBlock *block, MBOptimConfig *conf) {
    // Momentum = mémoire des tensions passées
    // Newton-Schulz = orthogonalisation des directions
    // → Directions isotropiques = équilibre des Volontés
}
```

### Un bug = conflit de Volontés

Dans la Théorie des Volontés, un bug n'est pas une erreur d'instruction.
C'est un **conflit de Volontés non résolu**.

Exemple : si deux MambaBlocks tentent de modifier la même mémoire,
c'est un conflit d'intentions — résolu par l'ordonnancement de k-mamba.

---

## Pourquoi cette séparation est puissante

### 1. Réutilisabilité

optimatrix peut être utilisé par d'autres projets (pas seulement Mamba) :
- Traitement d'images (ConvND séparable)
- Algèbre linéaire (GEMM/GEMV AVX2)
- Optimisation (gradient clipping, AdamW, MUON — CPU + CUDA)

### 2. Testabilité

Les kernels ASM peuvent être testés unitairement (phase 1-5 dans optimatrix).
k-mamba peut être testé avec des mocks.

### 3. Portabilité

Pour porter sur ARM NEON ou AVX-512 : modifier optimatrix uniquement.
k-mamba reste du C pur portable.

### 4. Clarté

Un chercheur peut lire k-mamba en une heure et comprendre l'architecture complète.
Les détails de calcul sont encapsulés dans optimatrix.

---

## Vision long terme

k-mamba est une brique fondatrice vers un **OS-IA post-Von Neumann** :
- Processus = Volontés (MambaBlocks)
- Communication = streams de tenseurs
- Scheduler = ordonnancement wavefront
- Mémoire = états persistants (h_t)

La séparation Volontés/Puissance préfigure cette architecture :
- Les Volontés sont les processus métier
- La Puissance est le moteur d'exécution

---

## Références

- **AGENTS.md** — Contexte technique et philosophique
- **THEORY.md** — Fondement mathématique Mamba-ND
- **ESTIMATIONS.md** — Complexité et benchmarks

---

## Auteur

**YEVI Mawuli Peniel Samuel**

*Ego Sum Optimus Optimus*
