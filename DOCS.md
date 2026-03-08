# DOCS.md — Documentation de la Charge de Calcul

## Notations

```
N    — nombre de dimensions spatiales
d_k  — taille de la dimension k  (k = 1..N)
D    — dimension du modèle (d_model)
M    — dimension de l'état caché (d_state)
K    — taille du noyau de convolution par axe
P    — nombre d'éléments total : P = d_1 × d_2 × ... × d_N
```

---

## 1. ConvND (Convolution N-dimensionnelle)

### Description
Applique un noyau de taille `K^N` sur un tenseur de taille `P × D`.
Capture le contexte local dans toutes les directions.

### Complexité
```
Calcul  : O(P · K^N · D_in · D_out)
Mémoire : O(K^N · D_in · D_out)  [noyau]
        + O(P · D)                 [tenseur]
```

### Opérations ASM requises
- Chargement de voisins en mémoire (accès strided ND)
- Produit scalaire par fenêtre (FMA — Fused Multiply-Add)
- Accumulation dans le tenseur de sortie

---

## 2. GEMV — Produit Matrice × Vecteur

### Description
Projection linéaire appliquée à chaque position :
`y = W · x` avec `W ∈ ℝ^{D_out × D_in}`, `x ∈ ℝ^{D_in}`

Utilisé pour : calcul de B(n), C(n), Δ(n).

### Complexité
```
Calcul  : O(P · D_in · D_out)   [une projection sur tout le tenseur]
Mémoire : O(D_in · D_out)       [matrice de poids]
```

### Opérations ASM requises
- Boucle sur les positions P (parallélisable)
- Produit scalaire W · x (AVX2 : 8 float32 en parallèle)

---

## 3. GEMM — Produit Matrice × Matrice

### Description
`C = A · B` avec `A ∈ ℝ^{m×k}`, `B ∈ ℝ^{k×n}`
Utilisé pour : in_proj, out_proj (batch de projections).

### Complexité
```
Calcul  : O(m · n · k)          [2mnk flops]
Mémoire : O(m·k + k·n + m·n)
```

### Opérations ASM requises
- Blocking (tiling) pour cache L1/L2
- FMA vectorisé (AVX2 / AVX-512)
- Accès row-major vs column-major (stride)

---

## 4. Hadamard ND — Produit élément par élément

### Description
`Z = X ⊙ Y` — multiplication terme à terme sur tenseurs de même forme.
Utilisé dans la gate de Mamba : `y = x ⊙ silu(z)`

### Complexité
```
Calcul  : O(P · D)
Mémoire : O(P · D)
```

### Opérations ASM requises
- Boucle simple sur P·D éléments
- VMULPS (AVX2 — 8 float32 en parallèle)

---

## 5. Selective Scan ND — Cœur du calcul

### Description
Résout la récurrence sur le DAG N-dimensionnel :

```
h(n) = Σ_{k=1}^{N} Ā_k(n) · h(n - e_k)  +  B̄(n) · x(n)
y(n) = C(n) · h(n)
```

C'est l'opération la plus coûteuse et la plus critique.

### Complexité
```
Calcul  : O(P · N · M²)          [N transitions d'état par position]
          + O(P · D · M)          [projection C · h]
Mémoire : O(P · M)               [états cachés h]
          + O(N · M²)            [matrices A_k]
```

### Contrainte de parallélisme
En 1D : le scan est séquentiel mais parallélisable via parallel prefix.
En ND : la frontière de calcul avance comme un **front d'onde** (wavefront).
Chaque "couche" du DAG peut être calculée en parallèle.

```
Ordre de calcul (2D exemple) :
  Diagonale 0 : h(0,0)
  Diagonale 1 : h(1,0), h(0,1)
  Diagonale 2 : h(2,0), h(1,1), h(0,2)
  ...
```

### Opérations ASM requises
- GEMV par position (A_k · h)
- Accumulation sur N prédécesseurs
- Ordonnancement wavefront (scheduling des dépendances)

### Backward 1D actuellement implémenté

Le dépôt expose aussi un backward 1D pour trois cas :

- `scan1d_backward` générique sur `[L, D, M]`
- spécialisation `M=1`
- spécialisation `M=1` avec `B/C` partagés par canal et `delta[t]` scalaire

Ce dernier cas correspond au chemin chaud utilisé par le LM Mamba CPU.
Il accepte aussi `A_diag[t,d]` pré-calculé pour éviter `expf()` dans le hot path.

---

## 6. Activations

### SiLU (Swish)
```
silu(x) = x · sigmoid(x) = x / (1 + exp(-x))
```
Complexité : O(P · D) — une passe sur le tenseur.

### Softplus
```
softplus(x) = log(1 + exp(x))
```
Utilisé pour Δ (doit être positif).

### Opérations ASM requises
- Approximation rapide de exp() (polynôme de Taylor ou table)
- VDIVPS pour la division vectorisée

---

## 7. RMSNorm

### Description
```
RMSNorm(x) = x / rms(x) · γ
rms(x) = sqrt(1/D · Σ x_i²)
```

### Complexité
```
Calcul  : O(P · D)
Mémoire : O(D)    [paramètres γ]
```

---

## Récapitulatif de la charge

| Opération | Complexité calcul | Priorité |
|---|---|---|
| Selective Scan ND | O(P·N·M²) | Critique |
| GEMM | O(m·n·k) | Haute |
| ConvND | O(P·K^N·D²) | Haute |
| GEMV | O(P·D²) | Moyenne |
| Hadamard | O(P·D) | Basse |
| Activations | O(P·D) | Basse |
| RMSNorm | O(P·D) | Basse |

---

## Stratégie d'implémentation ASM

```
Phase 1 — Fondations scalaires
  └── GEMV, GEMM (correctes, sans SIMD)

Phase 2 — Vectorisation
  └── AVX2 : 8×float32 en parallèle
  └── FMA  : a = b*c + a en une instruction

Phase 3 — Selective Scan ND
  └── Wavefront scheduling
  └── Parallélisme intra-diagonale

Phase 4 — ConvND
  └── Accès strided optimisés
  └── Tiling pour cache

Phase 5 — Intégration
  └── Interface C (extern "C") pour appel depuis Python/C++
```

---

## Format des tenseurs en mémoire

Stockage **row-major** (C-order), continu :

```
T[n_1][n_2]...[n_N][d] = base + (n_1·s_1 + n_2·s_2 + ... + n_N·s_N + d) * sizeof(elem)
```

Où `s_k = d_{k+1} · ... · d_N · D` est le stride de la dimension k.

Alignement **32 bytes** requis pour AVX2.
