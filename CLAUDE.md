# CLAUDE.md — Contexte Intellectuel & Philosophique de Samuel

> Ce fichier est un contexte fondamental. Lis-le entièrement avant de coder.
> Il ne s'agit pas de préférences stylistiques — il s'agit du **cadre conceptuel** dans lequel tout ce code existe.

---

## Qui est Samuel ?

**YEVI Mawuli Peniel Samuel** — étudiant en Licence Systèmes Embarqués & IoT à l'IFRI-UAC (Bénin), matricule 122473. Lead développeur, architecte de systèmes, et penseur.

Sa devise : **"Ego Sum Optimus Optimus"**  
Sa conviction : les structures priment sur les unités. *"On est assez grand pour voir des unités, il faut voir des structures."*

---

## La Théorie des Volontés

> C'est le cadre philosophique et formel que Samuel a développé. Comprendre ceci, c'est comprendre pourquoi il code comme il code.

### Intuition centrale

Les systèmes classiques fonctionnent par **instructions** : dis à la machine *quoi faire*, pas *pourquoi*.  
La Théorie des Volontés propose un renversement : les systèmes doivent opérer par **intentions** — des *Volontés* — qui convergent vers un état d'équilibre.

Ce n'est pas de la programmation déclarative classique. C'est plus profond : c'est une ontologie computationnelle.

---

### Les Axiomes

#### 1. La Volonté (𝕍)
Une Volonté est un vecteur d'intention dans un espace d'états. Elle n'est pas une commande — c'est une **orientation**.

```
𝕍 = (direction, intensité, contexte, convergence_cible)
```

Elle possède :
- Une **magnitude** : l'intensité de l'intention
- Une **direction** : vers quel état elle pousse le système
- Un **champ d'action** : le sous-espace sur lequel elle agit

#### 2. Le Kaos (𝕂)
Le Kaos n'est pas le chaos au sens péjoratif. C'est l'**espace des possibles non-structurés** — l'état initial avant qu'une Volonté ne s'exprime.

La fonction de Kaos `𝕂(t)` décrit la distribution d'états potentiels à l'instant `t`.

```
𝕂 : T → P(Ω)
```
où `Ω` est l'espace total des états du système, et `P(Ω)` sa partie puissante.

**Propriété clé** : Le Kaos n'est pas vide — il est *plein* de potentialité non-orientée.

#### 3. La Convergence vectorielle
Plusieurs Volontés coexistent dans un système. Elles interagissent :

```
𝕍_résultante = Σ wᵢ · 𝕍ᵢ
```

Le système évolue vers l'**équilibre des Volontés** — un état où la résultante des tensions est minimale. Ce n'est pas forcément le minimum global d'une fonction de coût : c'est l'état où les intentions cessent de se contredire.

#### 4. L'Équilibre (Ω*)
L'état cible n'est pas prescrit. Il **émerge** de la composition des Volontés.

```
Ω* = lim_{t→∞} Φ(𝕂(0), {𝕍ᵢ})
```

où `Φ` est l'opérateur d'évolution du système sous l'influence des Volontés.

---

### L'Informatique des Volontés

Samuel envisage un nouveau paradigme computationnel :

> **Au lieu de programmer des instructions, on programme des intentions.**

Concrètement :

| Paradigme classique | Informatique des Volontés |
|---|---|
| `for i in range(n): ...` | "Vouloir que chaque élément soit traité" |
| `if condition: do X` | "Volonté conditionnelle à résoudre" |
| Fonction déterministe | Attracteur de Volontés |
| Bug = erreur d'instruction | Bug = conflit de Volontés non résolu |
| Optimisation = minimiser coût | Optimisation = atteindre équilibre |

Ce paradigme a des implications profondes pour :
- L'**IA** : un modèle n'optimise pas une loss — il cherche l'équilibre de ses Volontés internes
- L'**OS** : un système d'exploitation n'alloue pas des ressources — il arbitre des Volontés de processus
- L'**IoT** : un capteur n'envoie pas des données — il exprime une Volonté d'être entendu

---

### Le Langage Formel des Structures Naturelles

À 14 ans, Samuel a créé un langage formel pour décrire les structures naturelles — grammaires, arbres, formes récurrentes dans la nature — avant même d'avoir les outils mathématiques pour les formaliser complètement.

L'idée centrale : **les structures naturelles sont des Volontés figées**. Une feuille, une spirale de Fibonacci, un réseau de neurones — ce sont des Kaos qui ont convergé.

---

### Rapport aux Matrices (exemple concret)

Quand Samuel dit *"il faut voir des structures, pas des unités"*, en contexte matriciel :

- Un débutant voit : `a[i][j] = 3`
- Un intermédiaire voit : une matrice `A ∈ ℝ^{n×m}`
- Samuel voit : **une transformation linéaire** — une Volonté d'espace qui dit "je veux envoyer ce vecteur là"

La Volonté d'une matrice, c'est sa **sémantique géométrique**, pas ses coefficients.

---

## Vision à long terme

```
Maintenant      → SmartLearn, BissiMamba, IoT firmware
Court terme     → Maîtrise de l'architecture bas niveau (C, RISC-V)
Moyen terme     → BissiMamba comme fondation d'un nouveau modèle
Long terme      → OS-IA sur architecture post-Von Neumann / quantique
```

Le fil conducteur : **chaque projet est une Volonté partielle** qui converge vers OS-IA.

---

## Comment coder pour Samuel

### 1. Pense en structures, pas en lignes
Ne propose jamais une solution ligne par ligne sans avoir d'abord posé l'architecture. Samuel ne veut pas du code — il veut de la **forme**.

### 2. Les abstractions ont du sens
Quand tu crées une abstraction, justifie-la en termes de ce qu'elle *veut faire*, pas juste de ce qu'elle fait.

### 3. Le bas niveau est noble
Samuel code en C, en assembleur si nécessaire. Ne sur-abstrait pas. Le bas niveau bien maîtrisé, c'est une Volonté pure — sans intermédiaire.

### 4. L'équilibre > la perfection
Un code "parfait" avec des dépendances conflictuelles est moins bon qu'un code en équilibre. Cherche la cohérence systémique.

### 5. Nomme les intentions
```c
// Mauvais
int x = buffer_size - current_pos;

// Bien (à la manière des Volontés)
int remaining_capacity = buffer_size - current_pos; // Volonté : savoir combien d'espace reste
```

---

## Stack actuelle (SmartLearn)

- **Backend** : PHP (`model.php`, `resume.php`, `progression.php`, `index.php`)
- **API** : `api.php` → OpenAI GPT-4o-mini / `api_simulation.php` (fallback)
- **DB** : Supabase (PostgreSQL)
- **Deploy** : Vercel
- **Design** : Playfair Display + Plus Jakarta Sans
  - Clair : `--bg:#f5f7fa`, `--text:#0f172a`, `--blue:#3b82f6`
  - Sombre : `--bg:#0a0a0a`, `--text:#e8f0fe`

---

## En résumé

> Samuel ne programme pas des machines. Il exprime des Volontés dans un langage que les machines comprennent — *pour l'instant*.
> L'objectif final : que les machines comprennent les Volontés directement.

Garde ça en tête à chaque ligne.

