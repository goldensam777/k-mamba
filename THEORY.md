# THEORY.md - Fondement mathematique de k-mamba

## 0. Théorie générale : Systèmes causaux ND et topologie wavefront

### 0.1 Positionnement scientifique

Ce projet formule une théorie des **systèmes causaux multi-dimensionnels sur grilles régulières**, fondée sur quatre piliers :

**Pilier I : Topologie causale comme primitive fondatrice**

Le niveau wavefront ND, défini par $l(n) = Σ n_i$, constitue l'ordonnancement topologique universel pour tout opérateur causal sur grille N-dimensionnelle. Ce n'est pas une heuristique d'implémentation mais une propriété structurelle du DAG causal sous-jacent.

**Pilier II : Unification des opérateurs via le squelette topologique**

ScanND (récurrence d'état avec mémoire longue) et ConvND (convolution locale à noyau dense) partagent le même squelette d'exécution wavefront. Seul le calcul élémentaire par position diffère :
- ScanND : $h(n) = Σ A_k · h(n-e_k) + B(n)·x(n)$
- ConvND : $z(n) = Σ K(r)·x(n-r)$ pour $r ∈ [0,K-1]^N$

Cette unification révèle que la complexité algorithmique réside dans la topologie d'ordonnancement, non dans le calcul local.

**Pilier III : Parallélisme structurel intra-niveau**

Le parallélisme exploitable est une propriété géométrique exacte : tous les points d'un même niveau wavefront sont mutuellement indépendants (Corollaire 4.3). Ce parallélisme est borné par la largeur du niveau, atteignant Θ(d) pour une grille d×d en 2D.

**Pilier IV : Architecture Volontés/Puissance**

Séparation philosophique entre intention (logique modèle, orchestration dans k-mamba) et calcul (kernels optimisés dans kernels/). Cette dualité reflète la Théorie des Volontés : les systèmes opèrent par intentions convergentes vers un équilibre, où chaque MambaBlock est une Volonté qui transforme la séquence, et MUON arbitre les tensions entre gradients.

### 0.2 Contribution originale

Contrairement à l'état de l'art (VMamba, Mamba-ND) qui décompose la dimensionnalité en scans 1D séquentiels, cette théorie propose deux innovations majeures :

**1. Récurrence ND simultanée.** La récurrence native $h(n) = Σ A_k·h(n-e_k) + B(n)·x(n)$ remplace les compositions de scans 1D.

**2. Convolution ND unifiée par wavefront.** La convolution dense $z(n) = Σ K(r)·x(n-r)$ partage le même squelette topologique wavefront que le scan, avec parallélisme intra-niveau. Cette unification théorique remplace les approches séparables ou séquentielles traditionnelles.

Le wavefront devient la **primitive mère** partagée par tous les opérateurs causaux ND.

### 0.3 Théorème central (Caractérisation exacte)

**Théorème (Classe des opérateurs causaux wavefront-exécutables).**

Soit $O$ un opérateur sur grille ND régulière $G = [0,d₁)×⋯×[0,dₙ)$. Les propositions suivantes sont équivalentes :

**(i)** $O$ est exécutable par parcours wavefront niveau par niveau, avec parallélisme intra-niveau exact.

**(ii)** Le graphe de dépendances de $O$ est un sous-graphe du DAG causal défini par l'ordre partiel $m ≺ n$ ssi $l(m) < l(n)$.

**(iii)** Pour tout point $n$, les dépendances de $O$ ne pointent que vers des points de niveau strictement inférieur ($l(m) < l(n)$), et il n'existe pas de dépendances entre points d'un même niveau.

**Preuve.**

*(i) ⇒ (ii)* : Par construction du parcours wavefront, tout point $n$ au niveau $s$ n'est calculé qu'après tous les niveaux $< s$. Donc toute dépendance de $n$ pointe nécessairement vers un niveau strictement inférieur. Le graphe de dépendances est bien un sous-graphe du DAG causal défini par $l(m) < l(n)$. ∎

*(ii) ⇒ (iii)* : Immédiat. Si le graphe de dépendances est un sous-graphe du DAG $l(m) < l(n)$, alors (a) toute dépendance pointe vers un niveau strictement inférieur, et (b) deux points du même niveau $s$ ne peuvent être liés car cela contredirait $l(m) < l(n)$ avec $l(m) = l(n) = s$. ∎

*(iii) ⇒ (i)* : Construisons le parcours wavefront et montrons qu'il est correct.

- **Initialisation.** Le niveau `0` contient uniquement l'origine $0$. Ce point n'a aucun prédécesseur dans la grille ($n - e_k$ serait hors domaine pour tout $k$). Il est calculable sans dépendance. ✓

- **Hérédité.** Supposons que tous les niveaux `0, 1, …, s-1` ont été calculés correctement. Soit `n` un point du niveau `s`, i.e. `l(n) = s`. Par (iii), toute dépendance de `n` pointe vers un point `m` avec `l(m) < s`. Donc `m` appartient à un niveau déjà calculé. La valeur `O[n]` est donc calculable. ✓

- **Parallélisme intra-niveau.** Soient `n₁, n₂` deux points distincts du niveau `s`. Par (iii), il n'existe pas de dépendance entre eux. Leurs calculs sont mutuellement indépendants et parallélisables. ✓

*Conclusion.* Le parcours niveau par niveau `0, 1, …, l_max` est un ordre topologique valide du DAG causal, avec parallélisme intra-niveau exact. `O` est wavefront-exécutable au sens de (i). ∎

**Corollaire.** La classe des opérateurs causaux ND sur grilles régulières coïncide exactement avec la classe des opérateurs wavefront-exécutables.

---

## 0.4 Positionnement critique : axiomes violés

Notre théorie procède par **violation constructive** de deux axiomes implicites de l'état de l'art.

### Axiome 1 : Causalité = Séquentialité (Gu et Dao, 2023)

**Le présupposé.** Dans la théorie originale de Mamba, la causalité temporelle implique une dépendance séquentielle : `h_t` ne dépend que de `h_{t-1}`. La récurrence est une chaîne linéaire.

**La généralisation.** En dimension N, nous montrons que la **causalité n'implique pas la séquentialité**. Un état `h(i,j)` dépend de deux prédécesseurs simultanés `(i-1,j)` et `(i,j-1)`, mais ces dépendances respectent un ordre partiel strict (le wavefront).

**Le résultat.** L'ordre partiel expose un parallélisme structurel exact : tous les points d'un niveau wavefront sont indépendants et traitables en parallèle. La théorie de Gu et Dao est le cas limite N=1 où le parallélisme intra-niveau se réduit à un seul point.

**Conséquence.** Nous déplaçons la complexité : ce n'est plus la dimension qui crée la complexité (par décomposition séquentielle), c'est la profondeur topologique `l(n)` qui structure le calcul parallèle.

---

### Axiome 2 : Convolution ND = Séparable (État de l'art classique)

**Le présupposé.** Une convolution ND efficace doit être décomposée en N convolutions 1D séquentielles (séparabilité). La convolution dense `K^N` serait prohibitive.

**La violation.** Nous montrons que la convolution ND **dense** (noyau complet, non séparable) devient viable grâce au wavefront. Le même squelette topologique qui rend le scanND parallèle s'applique à la convolution.

**Le mécanisme.**
- Convolution séparable classique : N passes 1D séquentielles, complexité `N·K·d^N`.
- Convolution dense wavefront : une passe wavefront, parallélisme intra-niveau, complexité `K^N·d^N` mais avec parallélisme Θ(d).

**Pourquoi c'est préférable.**
1. **Interactions croisées.** Une convolution dense capture les interactions diagonales `(i-1, j-1)` que la séparabilité ignore.
2. **Unification théorique.** ScanND et ConvND partagent le même squelette — ce n'est plus deux algorithmes distincts, mais deux instances d'une primitive topologique commune.
3. **Parallélisme structurel.** Le parallélisme n'est pas une heuristique (tiling, blocking), c'est une propriété géométrique du DAG causal.

---

### Synthèse : la primitive wavefront comme fondement

Ces deux violations convergent vers une même primitive : le **wavefront ND** comme ordonnancement topologique universel pour les opérateurs causaux sur grilles régulières.

| Axiome violé | Avant (séparable/séquentiel) | Après (wavefront unifié) |
|--------------|------------------------------|--------------------------|
| Causalité | Chaîne linéaire 1D | Ordre partiel ND |
| Convolution | N passes 1D | Une passe wavefront parallèle |
| Parallélisme | Inter-batch seulement | Intra-niveau exact |
| Unification | Aucune (scan vs conv distincts) | Squelette commun |

**Thèse centrale.** Le wavefront n'est pas une optimisation d'implémentation. C'est la structure topologique fondatrice qui révèle que les opérateurs causaux ND sont fondamentalement parallèles, simultanés, et unifiables.

---

## 1. Idée centrale

Le coeur théorique de `k-mamba` n'est pas seulement "un scan 2D" ou
"une convND". La contribution générale est :

- une topologie causale ND sur grille
- un générateur de wavefront borné
- un squelette d'exécution commun pour les opérateurs causaux ND

Autrement dit :

- `scanND` est une première instance de cette topologie
- `convND` native wavefront est une seconde instance
- le générateur `wavefront_nd_*` est la primitive fondatrice commune

Le projet ne se limite donc pas à étendre Mamba en ND. Il pose une structure
topologique universelle pour les opérateurs causaux ND sur tenseurs.

---

## 2. Rappel : Mamba 1D

### 2.1 SSM continu

Un State Space Model linéaire continu s'écrit :

```math
\dot{h}(t) = A h(t) + B u(t), \qquad y(t) = C h(t)
```

avec :

- `h(t) ∈ R^M` : état latent
- `u(t) ∈ R^D` : entrée
- `A ∈ R^{M×M}` : dynamique d'état
- `B ∈ R^{M×D}` : couplage entrée → état
- `C ∈ R^{D×M}` : projection état → sortie

### 2.2 Discrétisation sélective

Après discrétisation de type ZOH, puis sélectivité Mamba, on obtient :

```math
δ_t = softplus(W_δ x_t), \qquad B_t = f_B(x_t), \qquad C_t = f_C(x_t)
```

et la récurrence :

```math
h_t = Ā_t h_{t-1} + B̄_t x_t, \qquad y_t = C_t h_t
```

Dans la représentation scalaire par canal `d` et état `m` :

```math
a_t^{d,m} = e^{δ_t^d A^{d,m}}, \qquad b_t^{d,m} = δ_t^d B_t^{d,m} x_t^d
```

```math
h_t^{d,m} = a_t^{d,m} h_{t-1}^{d,m} + b_t^{d,m}
```

```math
y_t^d = Σ_m C_t^{d,m} h_t^{d,m}
```

### 2.3 Monoïde du scan SSM

La récurrence 1D admet la représentation :

```math
(a_t, b_t) \quad \text{telle que} \quad h_t = a_t h_{t-1} + b_t
```

avec l'opérateur de composition :

```math
(a_1, b_1) ⊗ (a_2, b_2) = (a_1 a_2,\ a_2 b_1 + b_2)
```

Cet opérateur est associatif et son neutre est `(1, 0)`.
Le scan Mamba est donc un scan préfixe sur un monoïde non commutatif.

---

## 3. Extension ND native

### 3.1 Tenseur d'entrée

Soit :

```math
X ∈ R^{d_1 × d_2 × ⋯ × d_N × D}
```

avec :

- `d_1, …, d_N` : dimensions spatiales
- `D` : profondeur (canaux)

### 3.2 Récurrence ND simultanée

L'extension native de Mamba vers N dimensions s'écrit :

```math
h(n) = Σ_{k=1}^{N} Ā_k(n) · h(n - e_k) + B̄(n) · x(n)
```

```math
y(n) = C(n) · h(n)
```

avec :

- `n = (n_1, …, n_N)` : coordonnées ND
- `e_k` : vecteur unitaire de l'axe k
- `A_k` : dynamique d'état selon axe k (input-dependent)
- `B` : couplage entrée → état
- `C` : projection état → sortie

### 3.3 Différence avec l'état de l'art

- **VMamba (2024)** : 4 scans 1D successifs dans 4 directions (cross-scan)
- **Mamba-ND (Li et al.)** : scans 1D alternés par couche
- **k-mamba** : **récurrence ND simultanée**, pas de décomposition séquentielle

---

## 4. Le wavefront comme ordre topologique

### 4.1 Définition du niveau

Pour un point `n = (n_1, …, n_N)` dans une grille ND, on définit :

```math
l(n) = n_1 + n_2 + ⋯ + n_N
```

C'est la **distance de Manhattan** depuis l'origine, ou la **profondeur topologique**
dans le DAG causal.

### 4.2 Lemme de causalité

**Lemme** : Si un opérateur ND causal dépend de positions `n - e_k`
(vecteurs unitaires), alors toute dépendance récurrente de `n` pointe vers
un niveau strictement inférieur.

**Démonstration** :

```math
l(n - e_k) = (n_1 + ⋯ + n_N) - 1 = l(n) - 1 < l(n)
```

CQFD. ∎

### 4.3 Corollaire d'indépendance

**Corollaire.** Deux points distincts `n₁, n₂` du même niveau `l(n₁) = l(n₂) = k` sont mutuellement indépendants.

*Preuve.* Supposons `n₁` dépend de `n₂`. Alors `n₁ = n₂ + eⱼ` pour un axe `j`. Donc `l(n₁) = l(n₂) + 1 = k + 1 ≠ k`. Contradiction. ∎

**Conséquence.** Tous les points d'un même niveau wavefront sont parallélisables.

### 4.4 Théorème du squelette topologique

**Théorème** :

Tout opérateur ND causal borné dont les dépendances récurrentes vont vers des
niveaux strictement inférieurs peut être exécuté correctement par parcours des
niveaux `0, 1, 2, …, l_max`, et à chaque niveau `s`, tous les points de `D_s`
peuvent être traités en parallèle.

**Démonstration** :

- Par le lemme, toute dépendance récurrente de `n` pointe vers un niveau
  strictement plus petit que `l(n)`.
- Donc, quand on commence le calcul du niveau `s`, toutes les données
  récurrentes requises ont déjà été produites aux niveaux `< s`.
- Par le corollaire, aucun point de `D_s` ne dépend d'un autre point de `D_s`.

Le parcours par wavefront est donc un ordre topologique valide du DAG causal,
et le parallélisme intra-wavefront est exact. ∎

---

## 5. Convolution ND native wavefront

### 5.1 Formulation

La convolution ND que nous implémentons est une convolution native à noyau
plein, dense, avec ordonnancement wavefront :

```math
z(n) = Σ_{r ∈ [0,K-1]^N} K(r) · x(n - r)
```

Le noyau est un tenseur ND complet :

```math
K : [0, K-1]^N × [0, D-1] → R
```

### 5.2 La convolution ND comme décision architecturale unifiée

Bien que la convolution ND dense ne nécessite pas théoriquement d'ordre
particulier (car elle lit seulement `x` qui est déjà connu), le choix du
wavefront comme ordonnancement commun est une **décision architecturale**,
pas une nécessité logique isolée.

Dans le contexte du bloc hybride ND (Section 6), où ConvND et ScanND
coexistent sur le même domaine causal, cet ordonnancement devient
**nécessaire** pour garantir la causalité du bloc composé.

**Pourquoi ce choix est préférable :**

| Aspect | Bénéfice |
|--------|----------|
| **Unification** | Même API, même plan que scanND |
| **Parallélisme** | Intra-niveau parfaitement parallélisable |
| **Cache** | Accès spatialement cohérents par niveau |
| **Prévisibilité** | Ordonnancement déterministe, debuggable |
| **Extensibilité** | Tout futur opérateur causal ND utilisera le même squelette |

La complexité algorithmique ne réside pas dans la nature (récurrente ou
convolutive) de l'opérateur, mais dans la **topologie du DAG causal
sous-jacent**. C'est ce que l'unification wavefront établit.

---

## 6. Vision K-Mamba unifiée

Le "vrai K-Mamba" vise deux primitives natives ND partageant le même
ordonnancement topologique wavefront :

1. `scanND` : dynamique d'état, mémoire longue, récurrence causale
2. `convND` : interaction locale bornée, noyau ND simultané, axes vus ensemble

La structure générale d'un bloc hybride :

```math
u(n) = ConvND(x)(n)
```

```math
h(n) = Σ_{k=1}^{N} Ā_k(n) · h(n - e_k) + B̄(n) · u(n)
```

```math
y(n) = C(n) · h(n)
```

Ou, dans une version à deux branches :

```math
y(n) = Ψ(ScanND(x)(n),\ ConvND(x)(n))
```

Le point théorique central n'est pas la forme exacte de `Ψ`.
Le point central est que les deux branches vivent sur la **même topologie causale
ND** et partagent le **même générateur de wavefront**.

---

## 7. Le générateur de wavefront ND dans le code

Le générateur `wavefront_nd_*` expose exactement cette structure :

- validation des dimensions
- nombre total de points
- niveau maximal
- taille d'un niveau
- itération d'un niveau
- itération de tous les niveaux
- offset row-major

La sémantique fondamentale est :

```text
niveau(idx) = idx[0] + idx[1] + ... + idx[ndims-1]
```

et :

```text
si les dépendances récurrentes vont vers des niveaux < k,
tous les points du niveau k sont indépendants
```

Ce générateur appartient à `k-mamba` parce qu'il encode une topologie causale
de modèle, pas un simple kernel numérique.

---

## 8. Conséquence architecturale

Le générateur de wavefront ND devient la **primitive mère**.

Au-dessus :

- `scanND` branche récurrence
- `convND` branche locale (wavefront parallèle)
- demain `convKD`, opérateurs implicites, ou tout autre opérateur causal ND

En dessous :

- version C de référence (séquentiel)
- version CPU spécialisée (ASM AVX2)
- version GPU (CUDA)

La thèse générale du projet peut alors se formuler proprement :

`k-mamba` n'implante pas seulement un `scan2d`, mais une **primitive topologique
universelle pour les opérateurs causaux ND sur grille**, dont `scanND` et
`convND` sont deux instances fondamentales partageant le même squelette
d'ordonnancement wavefront.

---

## 9. Protocole de validation empirique

**Statut : à réaliser.** Cette section décrit le protocole qui validera
(ou infirmera) les affirmations théoriques du document.

### 9.1 Configuration expérimentale

**Tâche.** Prédiction de pixel sur images MNIST 28×28 (D=64, M=16).

**Configurations comparées :**

| Approche | Architecture | Dépendances |
|----------|-------------|-------------|
| **Baseline** | Flatten 28×28 → 784, scan 1D séquentiel | Chaîne 1D simple |
| **K-Mamba** | Conv2D + Scan2D wavefront unifié | 2 prédécesseurs simultanés + convolution locale |

### 9.2 Hypothèses falsifiables

**H1 — Parallélisme structurel.**

Sur une grille 28×28, l'implémentation wavefront de `scanND` atteint un
speedup ≥ 20× sur 28 cœurs par rapport à une implémentation séquentielle
naïve (parcours row-major).

*Critère d'échec.* Speedup < 20× sur 28 cœurs. Interprétation : le
parallélisme intra-niveau existe théoriquement mais est annulé par les coûts
de synchronisation inter-niveaux ou de scheduling. La théorie du parallélisme
structurel exact reste valide, mais son bénéfice pratique est conditionnel
à l'overhead d'ordonnancement.

---

**H2 — Qualité de modélisation.**

Un modèle K-Mamba combinant `scanND` et `convND` sur squelette wavefront
commun atteint une perplexité sur MNIST pixel-prediction inférieure d'au
moins 10% à un baseline `flatten + scan1D` de même nombre de paramètres.

*Critère d'échec.* PPL baseline ≤ PPL K-Mamba ou écart < 10%. Interprétation :
la récurrence ND simultanée ne capture pas d'interactions croisées
supplémentaires par rapport à la composition de scans 1D — ce qui remettrait
en question l'avantage empirique de la récurrence ND native face à la
décomposition séquentielle.

---

**H3 — Unification wavefront pour ConvND.**

La convolution dense wavefront (3×3, noyau complet) atteint un temps
d'inférence comparable (≤ 2×) à une convolution séparable classique sur CPU,
tout en produisant des feature maps différenciables (norme de la différence
> ε sur un batch test).

*Critère d'échec.* Facteur > 2× en temps ou feature maps indiscernables.
Interprétation dans les deux cas : soit l'unification wavefront est trop
coûteuse pour ConvND dense, soit la densité du noyau n'apporte pas
d'information supplémentaire mesurable — dans les deux cas, l'argument
d'unification est affaibli empiriquement.

### 9.3 Interprétation globale

L'échec simultané de H1 et H2 invaliderait la thèse centrale du document.
L'échec de H3 seul affaiblirait l'unification ConvND/ScanND sans remettre
en cause la théorie du scan ND simultané.

---

## Références

- Gu and Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
- Liu et al. (2024). VMamba: Visual State Space Models.
- Li et al. (2024). Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data.
- Blelloch (1990). Prefix Sums and Their Applications.
