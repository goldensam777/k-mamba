# THEORY.md - Fondement mathematique de k-mamba

## 1. Idee centrale

Le coeur theorique de `k-mamba` n'est pas seulement "un scan 2D" ou
"une convND". La contribution generale est :

- une topologie causale ND sur grille
- un generateur de wavefront borne
- un squelette d'execution commun pour les operateurs causaux ND

Autrement dit :

- `scanND` est une premiere instance de cette topologie
- `convND` native simultanee est une seconde instance
- le generateur `wavefront_nd_*` est la primitive fondatrice commune

Le projet ne se limite donc pas a etendre Mamba en ND. Il pose une structure
topologique universelle pour les operateurs causaux ND sur tenseurs.

---

## 2. Rappel : Mamba 1D

### 2.1 SSM continu

Un State Space Model lineaire continu s'ecrit :

```math
\dot{h}(t) = A h(t) + B u(t), \qquad y(t) = C h(t)
```

avec :

- `h(t) in R^M` : etat latent
- `u(t) in R^D` : entree
- `A in R^{M x M}` : dynamique d'etat
- `B in R^{M x D}` : couplage entree -> etat
- `C in R^{D x M}` : projection etat -> sortie

### 2.2 Discretisation selective

Apres discretisation de type ZOH, puis selectivite Mamba, on obtient :

```math
\delta_t = softplus(W_delta x_t), \qquad
B_t = f_B(x_t), \qquad
C_t = f_C(x_t)
```

et la recurrence :

```math
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \qquad y_t = C_t h_t
```

Dans la representation scalaire par canal `d` et etat `m` :

```math
a_t^{d,m} = e^{\delta_t^d A^{d,m}}, \qquad
b_t^{d,m} = \delta_t^d B_t^{d,m} x_t^d
```

```math
h_t^{d,m} = a_t^{d,m} h_{t-1}^{d,m} + b_t^{d,m}
```

```math
y_t^d = \sum_m C_t^{d,m} h_t^{d,m}
```

### 2.3 Monoid du scan SSM

La recurrence 1D admet la representation :

```math
(a_t, b_t) \quad \text{telle que} \quad h_t = a_t h_{t-1} + b_t
```

avec l'operateur de composition :

```math
(a_1, b_1) \otimes (a_2, b_2) = (a_1 a_2, a_2 b_1 + b_2)
```

Cet operateur est associatif et son neutre est `(1, 0)`.
Le scan Mamba est donc un scan prefixe sur un monoid non commutatif.

---

## 3. Extension ND native

### 3.1 Tenseur d'entree

Soit :

```math
X \in R^{d_1 \times d_2 \times \cdots \times d_N \times D}
```

et une position :

```math
\mathbf{n} = (n_1, n_2, \ldots, n_N)
```

avec `0 <= n_k < d_k`.

### 3.2 Recurrence ND

La recurrence ND native de `k-mamba` est :

```math
h(\mathbf{n}) =
\sum_{k=1}^{N} \bar{A}_k(\mathbf{n}) h(\mathbf{n} - \mathbf{e}_k)
 + \bar{B}(\mathbf{n}) x(\mathbf{n})
```

```math
y(\mathbf{n}) = C(\mathbf{n}) h(\mathbf{n})
```

ou :

- `e_k` est le vecteur unite de l'axe `k`
- `h(n) = 0` des qu'un indice sort du domaine
- `bar(A)_k(n)` code la propagation depuis le predecesseur sur l'axe `k`

La difference fondamentale avec `VMamba` et `Mamba-ND` est que les `N`
predecesseurs sont agreges simultanement au meme pas de recurrence. Ce n'est
pas une alternance de scans 1D ; c'est une dynamique ND unique.

---

## 4. Primitive topologique : le wavefront ND

### 4.1 Niveau topologique

On definit le niveau d'un indice ND par :

```math
\ell(\mathbf{n}) = \sum_{k=1}^{N} n_k
```

Le niveau maximal d'une grille `d_1 x ... x d_N` est :

```math
\ell_{max} = \sum_{k=1}^{N} (d_k - 1)
```

Le `wavefront` de niveau `s` est :

```math
\mathcal{D}_s = \{ \mathbf{n} \mid \ell(\mathbf{n}) = s \}
```

Ce sont les hyper-diagonales ND.

### 4.2 Operateurs causaux bornes

On introduit maintenant la classe generale des operateurs causaux ND.

#### Definition

Un operateur ND sur grille est dit causal borne s'il existe un support fini
`R subset N^N` tel que la valeur en `n` ne depend que de :

- donnees externes `x(n-r)` pour `r in R`
- et, pour les variables recurrentes, uniquement d'etats `h(n-r)` avec `r != 0`

et chaque dependance invalide hors du domaine vaut zero.

Deux cas particuliers nous interessent :

1. recurrence d'etat :

```math
h(\mathbf{n}) = \Phi(\{h(\mathbf{n} - \mathbf{r})\}_{\mathbf{r} \in R_h},
                     \{x(\mathbf{n} - \mathbf{r})\}_{\mathbf{r} \in R_x})
```

avec `0 notin R_h`

2. convolution locale :

```math
z(\mathbf{n}) = \sum_{\mathbf{r} \in R_x} K(\mathbf{r}) x(\mathbf{n} - \mathbf{r})
```

ou `R_x` est borne et causal

### 4.3 Lemma fondamental

#### Lemma

Si `r in N^N` et `r != 0`, alors pour tout indice valide `n-r` :

```math
\ell(\mathbf{n} - \mathbf{r}) = \ell(\mathbf{n}) - \ell(\mathbf{r}) < \ell(\mathbf{n})
```

#### Demonstration

Comme `r in N^N`, toutes les composantes de `r` sont positives ou nulles.
Comme `r != 0`, il existe au moins une composante strictement positive.
Donc :

```math
\ell(\mathbf{r}) = \sum_k r_k > 0
```

Par linearite de la somme :

```math
\ell(\mathbf{n} - \mathbf{r}) = \sum_k (n_k - r_k)
= \sum_k n_k - \sum_k r_k
= \ell(\mathbf{n}) - \ell(\mathbf{r})
```

et comme `ell(r) > 0`, on obtient :

```math
\ell(\mathbf{n} - \mathbf{r}) < \ell(\mathbf{n})
```

CQFD.

### 4.4 Corollaire d'independance intra-wavefront

#### Corollaire

Pour toute recurrence ND causale bornee, deux points distincts d'un meme niveau
ne peuvent pas dependre l'un de l'autre.

#### Demonstration

Soient `n1` et `n2` deux indices tels que :

```math
\ell(\mathbf{n}_1) = \ell(\mathbf{n}_2)
```

Supposons que `n1` depende de `n2`. Alors il existe `r != 0` tel que :

```math
\mathbf{n}_2 = \mathbf{n}_1 - \mathbf{r}
```

Le lemme precedent impose alors :

```math
\ell(\mathbf{n}_2) < \ell(\mathbf{n}_1)
```

ce qui contredit `ell(n1) = ell(n2)`.

Donc les points d'un meme wavefront sont mutuellement independants.

CQFD.

### 4.5 Theoreme du squelette topologique

#### Theoreme

Tout operateur ND causal borne dont les dependances recurrentes vont vers des
niveaux strictement inferieurs peut etre execute correctement par parcours des
niveaux :

```math
0, 1, 2, \ldots, \ell_{max}
```

et, a chaque niveau `s`, tous les points de `D_s` peuvent etre traites en
parallele.

#### Demonstration

- Par le lemme, toute dependance recurrente de `n` pointe vers un niveau
  strictement plus petit que `ell(n)`.
- Donc, quand on commence le calcul du niveau `s`, toutes les donnees
  recurrentes requises ont deja ete produites aux niveaux `< s`.
- Par le corollaire, aucun point de `D_s` ne depend d'un autre point de `D_s`.

Le parcours par wavefront est donc un ordre topologique valide du DAG causal,
et le parallelisme intra-wavefront est exact.

CQFD.

---

## 5. Pourquoi cela couvre a la fois scanND et convND

### 5.1 ScanND

Le `scanND` entre dans le cas recurrent. Les dependances portent sur :

```math
h(\mathbf{n} - \mathbf{e}_1), \ldots, h(\mathbf{n} - \mathbf{e}_N)
```

Chaque vecteur `e_k` est non nul, donc toute dependance va vers un niveau
strictement inferieur. Le wavefront n'est pas un heuristique : c'est l'ordre
topologique exact.

### 5.2 ConvND native simultanee

La `convND` que nous voulons pour `K-Mamba` n'est pas une chaine de Conv1D
separables. C'est une convolution ND native a noyau plein :

```math
z(\mathbf{n}) = \sum_{\mathbf{r} \in [0, K-1]^N} K(\mathbf{r}) x(\mathbf{n} - \mathbf{r})
```

Le point essentiel est que :

- les axes se voient dans le meme noyau `K(r_1, ..., r_N)`
- l'operateur est simultane, pas factorise
- les dependances portent sur l'entree `x`, pas sur les sorties `z`

Donc mathematiquement, une telle convolution peut etre evaluee dans n'importe
quel ordre puisque `x` est une donnee lue seule.

Mais si son support est causal borne, elle est compatible avec le meme
squelette wavefront que le scan :

- non parce qu'elle en a besoin pour etre correcte
- mais parce que le wavefront fournit une discipline topologique commune
- et une API d'ordonnancement unifiee pour tous les operateurs causaux ND

En pratique :

- pour `scanND`, le wavefront est necessaire
- pour `convND`, le wavefront est un ordonnancement commun volontaire

---

## 6. ConvND simultanee versus convND separable

### 6.1 Convolution separable

La version separable axe par axe suppose :

```math
K(\mathbf{r}) = \prod_{k=1}^{N} k_k(r_k)
```

ou, en implementation, une composition de `N` convolutions 1D.

Avantages :

- cout reduit
- implementation simple
- bon chemin rapide

Limite :

- les interactions inter-axes sont indirectes
- le noyau ND n'est pas appris comme une entite unique

### 6.2 Convolution simultanee native

La version simultanee apprend directement :

```math
K : [0, K-1]^N -> R
```

ou canal par canal en depthwise :

```math
K : [0, K-1]^N x [0, D-1] -> R
```

Cette forme est strictement plus expressive que la separabilite. Elle est la
bonne primitive si l'on veut que les axes "se voient" dans un meme voisinage
local ND.

---

## 7. Le generateur de wavefront ND dans le code

Le generateur `wavefront_nd_*` expose exactement cette structure :

- validation des dimensions
- nombre total de points
- niveau maximal
- taille d'un niveau
- iteration d'un niveau
- iteration de tous les niveaux
- offset row-major

La semantique fondamentale est :

```text
niveau(idx) = idx[0] + idx[1] + ... + idx[ndims-1]
```

et :

```text
si les dependances recurrentes vont vers des niveaux < k,
tous les points du niveau k sont independants
```

Ce generateur appartient a `k-mamba`, pas a `optimatrix`, parce qu'il encode
une topologie causale de modele, pas un simple kernel numerique.

---

## 8. Vision K-Mamba

Le "vrai K-Mamba" vise donc deux primitives natives ND partageant le meme
ordonnancement topologique :

1. `scanND`
   dynamique d'etat, memoire longue, recurrence causale

2. `convND`
   interaction locale bornée, noyau ND simultane, axes vus ensemble

La structure generale d'un bloc hybride devient :

```math
u(\mathbf{n}) = ConvND(x)(\mathbf{n})
```

```math
h(\mathbf{n}) =
\sum_{k=1}^{N} \bar{A}_k(\mathbf{n}) h(\mathbf{n} - \mathbf{e}_k)
+ \bar{B}(\mathbf{n}) u(\mathbf{n})
```

```math
y(\mathbf{n}) = C(\mathbf{n}) h(\mathbf{n})
```

ou, dans une version a deux branches :

```math
y(\mathbf{n}) = Psi(ScanND(x)(\mathbf{n}), ConvND(x)(\mathbf{n}))
```

Le point theorique central n'est pas la forme exacte de `Psi`.
Le point central est que les deux branches vivent sur la meme topologie causale
ND et peuvent partager le meme generateur de wavefront.

---

## 9. Consequence architecturale

Le generateur de wavefront ND devient la primitive mere.

Au-dessus :

- `scanND` branche recurrence
- `convND` branche locale
- demain `convKD`, operateurs implicites, ou tout autre operateur causal ND

En dessous :

- version C de reference
- version CPU specialisee
- version GPU

La these generale du projet peut alors se formuler proprement :

`k-mamba` n'implante pas seulement un `scan2d`, mais une primitive topologique
universelle pour les operateurs causaux ND sur grille, dont `scanND` et
`convND` sont deux instances fondamentales.

---

## References

- Gu and Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
- Liu et al. (2024). VMamba: Visual State Space Models.
- Li et al. (2024). Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data.
- Blelloch (1990). Prefix Sums and Their Applications.
