# THEORY.md — Fondement Theorique de k-mamba

## 1. Mamba 1D — Ce qui existe

### Le modele continu (SSM classique)

```math
x'(t) = A * x(t) + B * u(t)
y(t)  = C * x(t)
```

- `x(t) in R^N`  — etat cache
- `u(t) in R^D`  — entree
- `A in R^{N x N}` — transition d'etat
- `B in R^{N x D}` — matrice d'entree
- `C in R^{D x N}` — matrice de sortie

### Discretisation (ZOH)

```math

A_b = exp(delta * A)
```

```math
B_bar = (delta * A)^{-1} * (exp(delta * A) - I) * delta * B

```

Recurrence discrete :

```math
h_t = A_bar * h_{t-1} + B_bar * x_t

```

```math
    y_t = C * h_t
```

### Selectivite (Mamba)

B, C, et delta **dependent de l'entree** :

```math
B_t = f_B(x_t)
```

```math
C_t = f_C(x_t)
```

```math
delta_t = softplus(f_delta(x_t))
```

Le modele choisit **quoi retenir** a chaque pas — d'ou "selective scan".

### Le Conv1D dans Mamba

Avant le scan, une Conv1D capture le contexte local :

```math
x' = Conv1D(x)
```

Elle glisse un noyau de taille K le long de l'axe sequence.

---

## 2. Mamba ND — L'extension

### Intuition

Si le scan 1D fait evoluer l'etat le long d'une sequence,
le scan ND fait evoluer l'etat dans un espace a N dimensions.

### Formulation

Pour un tenseur `X in R^{d_1 x d_2 x ... x d_N x D}` :

```math
h(n) = sum_{k=1}^{N} A_k * h(n - e_k)  +  B(n) * x(n)
y(n) = C(n) * h(n)
```

ou `e_k` est le vecteur unite dans la dimension k.

### Structure des dependances

En 1D : `h_1 -> h_2 -> h_3 -> h_4`

En 2D :

```math
h(i-1,j) \
            -> h(i,j)
h(i,j-1) /
```

En ND : chaque point depend de N predecesseurs — DAG N-dimensionnel.

### ConvND separable avant le scan

```plain
  X' = ConvND(X)   chaine de Conv1D axe par axe
```

Capture le contexte local dans toutes les directions
avant que le scan selectif ne propage l'etat.

L'implementation est separable depthwise : un noyau [K, D] par axe,
applique du dernier axe au premier. Le backward propage les gradients
en ordre inverse avec recompute des intermediaires.

---

## 3. Pourquoi c'est different de VMamba et Mamba-ND

**VMamba (2024)** : 4 scans 1D dans 4 directions (gauche-droite, droite-gauche,
haut-bas, bas-haut). C'est 4 x Mamba1D, pas un vrai Mamba2D.

**Mamba-ND (Li et al., 2024)** : scans 1D alternes le long de chaque axe
par couche. Toujours des scans 1D reordonnes, pas une recurrence native ND.

**k-mamba** : recurrence **simultanée** dans toutes les dimensions.
L'etat a la position (i,j) depend de h(i-1,j) ET h(i,j-1) au meme pas.
L'ordonnancement wavefront (diagonales anti) resout les dependances
tout en exposant du parallelisme intra-diagonale.

---

## 4. Le MambaBlock complet

Le MambaBlock dans k-mamba implemente le pipeline Mamba de bout en bout :

```text
input [seq_len x dim]
  |
  v
W_in : dim -> state_size (projection lineaire)
  |
  v
SiLU (activation gate)
  |
  v
delta_proj + softplus + clamp -> dt_t (pas de temps adaptatif)
  |
  v
Selective Scan (1D ou 2D wavefront)
  h_t = exp(dt * A) * h_{t-1} + dt * B * u_t
  y_t = C * h_t
  |
  v
W_out : state_size -> dim (projection de sortie)
  |
  v
output [seq_len x dim]
```

Le backward traverse ce pipeline en sens inverse, accumulant les gradients
pour W_in, W_out, A_log, B, C, delta_proj. L'optimiseur MUONCLIP
(momentum + gradient clipping + weight decay) met a jour les poids.

---

## 5. Ce que k-mamba fournit

```plain
1. Orchestration ND   — stack MambaBlocks, embedding, LM head
2. Checkpoint I/O     — format binaire avec magic "KMAMBA"
3. Training loop      — batch training avec accumulation gradients
4. optimatrix         — kernels de calcul (GEMM, scan, conv, MUONCLIP)
```

k-mamba orchestre, optimatrix calcule.

---

## 6. Architecture Volontés/Puissance

### k-mamba (Volontés)
Responsable de l'orchestration du modèle Mamba :
- Création et gestion des MambaBlocks
- Embedding lookup et LM head
- Softmax, cross-entropy, loss computation
- Training loop sur batches
- Checkpoint I/O (format binaire "KMAMBA")
- Appel des kernels optimatrix

### optimatrix (Puissance)
Fournit les primitives de calcul haute performance :
- Kernels algébriques (GEMM/GEMV, Hadamard)
- Activations vectorisées (SiLU, Sigmoid, Softplus)
- Scans sélectifs (1D, 2D wavefront)
- Convolution 1D depthwise
- Rétropropagation complète des scans

Cette séparation permet :
- **Réutilisabilité** : optimatrix peut servir à d'autres architectures
- **Clarté** : logique modèle (k-mamba) séparée du calcul (optimatrix)
- **Performance** : kernels optimisés ASM AVX2 sans compromis

---

## References

- Gu & Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.

- Liu et al. (2024). VMamba: Visual State Space Models.

- Li et al. (2024). Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data.

- Kosson et al. (2024). Newton-Schulz orthogonalization (cited in MUON paper).

- Moonshot AI (2025). MUON optimizer (arXiv:2502.16982).
