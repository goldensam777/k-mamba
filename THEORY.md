# THEORY.md — Fondement Théorique d'optimatrix

## 1. Mamba 1D — Ce qui existe

### Le modèle continu (SSM classique)

```math
ẋ(t) = A · x(t) + B · u(t)
y(t) = C · x(t)
```

- `x(t) ∈ ℝ^N`  — état caché (mémoire du système)
- `u(t) ∈ ℝ^D`  — entrée à l'instant t
- `A ∈ ℝ^{N×N}` — matrice de transition d'état
- `B ∈ ℝ^{N×D}` — matrice d'entrée
- `C ∈ ℝ^{D×N}` — matrice de sortie

### Discrétisation (ZOH — Zero Order Hold)

Pour pouvoir traiter des séquences discrètes :

```math
Ā = exp(Δ · A)
B̄ = (ΔA)^{-1} · (exp(ΔA) - I) · ΔB
```

Ce qui donne la récurrence discrète :

```math
h_t = Ā · h_{t-1} + B̄ · x_t
y_t = C · h_t
```

### Ce que Mamba ajoute : la sélectivité

Dans Mamba, `B`, `C`, et `Δ` **dépendent de l'entrée** :

```math
B_t = f_B(x_t)
C_t = f_C(x_t)
Δ_t = softplus(f_Δ(x_t))
```

Le modèle choisit **quoi retenir** à chaque pas — d'où "selective scan".

### Le Conv1D dans Mamba

Avant le scan sélectif, une Conv1D capture le contexte local :

```math
x' = Conv1D(x)    sur l'axe séquence
```

Elle glisse un noyau de taille k le long d'un seul axe.
C'est une approximation locale de la dérivée discrète —
elle relie chaque position à ses voisines immédiates.

---

## 2. Mamba ND — L'extension

### Intuition

Si Conv1D glisse sur 1 axe, ConvND glisse sur N axes simultanément.
Si le scan 1D fait évoluer l'état le long d'une séquence,
le scan ND fait évoluer l'état dans un espace à N dimensions.

### Formulation

Pour un tenseur d'entrée `X ∈ ℝ^{d_1 × d_2 × ... × d_N × D}` :

L'état en un point `n = (n_1, n_2, ..., n_N)` est :

```math
h(n) = Σ_{k=1}^{N}  A_k · h(n - e_k)  +  B(n) · x(n)
y(n) = C(n) · h(n)
```

où :

- `e_k` est le vecteur unité dans la dimension k
- `A_k ∈ ℝ^{M×M}` est la matrice de transition pour l'axe k
- `h(n) ∈ ℝ^M` est l'état caché au point n
- `B(n)`, `C(n)` sont sélectifs (dépendent de l'entrée)

### Structure des dépendances

En 1D : chaque point dépend d'un seul prédécesseur.

```matrix
h_1 → h_2 → h_3 → h_4
```

En 2D : chaque point dépend de deux prédécesseurs.

```matrix
h(i-1,j)
    ↘
       h(i,j)
    ↗
h(i,j-1)
```

En ND : chaque point dépend de N prédécesseurs — un par dimension.
La structure de dépendance est un **DAG N-dimensionnel**.

### ConvND avant le scan

```matrix
X' = ConvND(X)    noyau de taille (k_1, k_2, ..., k_N)
```

Capture le contexte local dans toutes les directions
avant que le scan sélectif ne propage l'état.

### Cas particuliers

| N | Données | Cas d'usage |
|---|---------|-------------|
| 1 | séquences | texte, audio, ADN |
| 2 | matrices | images, spectrogrammes |
| 3 | volumes | vidéo, IRM, physique 3D |
| N | tenseurs | graphes, physique ND |

---

## 3. Pourquoi c'est différent de VMamba

VMamba (2024) approche le 2D en **scannant dans 4 directions** séquentielles.
C'est 4 × Mamba1D, pas un vrai Mamba2D.

Le vrai Mamba ND résout la récurrence **simultanément** dans toutes les dimensions.
La difficulté : le DAG de dépendances rend le calcul parallèle non-trivial.

---

## 4. Ce qu'optimatrix doit fournir

Pour que ce Mamba ND existe, il faut :

```list
1. ConvND          — convolution N-dimensionnelle
2. GEMV / GEMM     — projections linéaires (in_proj, out_proj, dt_proj)
3. Hadamard ND     — produit élément par élément sur tenseurs
4. Selective Scan ND — le cœur : récurrence dans le DAG ND
5. Activations     — SiLU, sigmoid, softplus
6. Normalization   — RMSNorm
```

Ces 6 opérations constituent le noyau d'optimatrix.
