# SOURCES.md — Références Bibliographiques

> Pour la présentation de thèse / exposé de Samuel YEVI.
> Classées par thème, avec le niveau de lecture recommandé.
> ★ = accessible L1/L2 | ★★ = intermédiaire | ★★★ = avancé

---

## 1. State Space Models et Mamba

### Articles fondateurs

**[1] Mamba — l'article principal**
> Gu, A., & Dao, T. (2023).
> *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
> arXiv:2312.00752
> https://arxiv.org/abs/2312.00752
> ★★ — L'article que tu dois lire en priorité. Sections 1, 2, 3.1, 3.3, 3.4 et annexe complexité.

**[2] S4 — le précurseur de Mamba**
> Gu, A., Goel, K., & Ré, C. (2021).
> *Efficiently Modeling Long Sequences with Structured State Spaces.*
> ICLR 2022. arXiv:2111.00396
> https://arxiv.org/abs/2111.00396
> ★★★ — Plus mathématique. Lire seulement l'introduction et la section 1.

**[3] S4ND — première généralisation ND des SSM**
> Nguyen, E., Goel, K., Gu, A., et al. (2022).
> *S4ND: Modeling Images and Videos as Multidimensional Signals with State Spaces.*
> NeurIPS 2022. arXiv:2210.06583
> https://arxiv.org/abs/2210.06583
> ★★★ — Résout le SSM ND via équations différentielles partielles (PDEs).
> Lire sections 3 et 4. Référence fondatrice pour les SSM multi-dimensionnels.

**[4] HiPPO — les fondations mathématiques**
> Gu, A., Dao, T., Ermon, S., Rudra, A., & Ré, C. (2020).
> *HiPPO: Recurrent Memory with Optimal Polynomial Projections.*
> NeurIPS 2020. arXiv:2008.07669
> https://arxiv.org/abs/2008.07669
> ★★★ — Très mathématique. Pour référence uniquement.

---

## 2. Extensions 2D/ND de Mamba (contexte et positionnement)

**[5] VMamba — scan 2D par 4 directions**
> Liu, Y., Tian, Y., Zhao, Y., et al. (2024).
> *VMamba: Visual State Space Model.*
> arXiv:2401.13260
> https://arxiv.org/abs/2401.13260
> ★★ — Lire sections 1 et 3. Approche : 4 scans 1D séquentiels (Cross-Scan).
> Limite claire : ce n'est pas un vrai scan 2D simultané — c'est 4 × Mamba1D.

**[6] Mamba-ND — la référence la plus proche d'optimatrix**
> Li, S., et al. (2024).
> *Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data.*
> arXiv:2402.05892
> https://arxiv.org/abs/2402.05892
> ★★ — Lire sections 1, 3 et 4 attentivement.
> Approche : déroulement des données en ordre row-major, alternant les dimensions
> par couche. Toujours des scans 1D réordonnés, pas une récurrence vraie dans
> le DAG ND. C'est le point de différenciation clé avec optimatrix.

**[7] Vision Mamba**
> Zhu, L., Liao, B., Zhang, Q., et al. (2024).
> *Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model.*
> arXiv:2401.09417
> https://arxiv.org/abs/2401.09417
> ★★ — Autre approche 2D bidirectionnelle. Complémentaire à VMamba.

---

## 3. Algèbre Linéaire et BLAS

**[8] BLAS — l'article original**
> Lawson, C. L., Hanson, R. J., Kincaid, D. R., & Krogh, F. T. (1979).
> *Basic Linear Algebra Subprograms for Fortran Usage.*
> ACM Transactions on Mathematical Software, 5(3), 308–323.
> ★★ — Pour citer les origines de BLAS. Introduction lisible.

**[9] BLAS Level 3 — GEMM**
> Dongarra, J. J., Du Croz, J., Hammarling, S., & Hanson, R. J. (1988).
> *An Extended Set of FORTRAN Basic Linear Algebra Subprograms.*
> ACM TOMS, 14(1), 1–17.
> ★★ — Définit GEMM formellement.

**[10] Anatomy of GEMM — analyse détaillée**
> Goto, K., & van de Geijn, R. (2008).
> *Anatomy of High-Performance Matrix Multiplication.*
> ACM TOMS, 34(3).
> ★★★ — Explique le tiling/blocking pour le cache. Très technique.
> Essentiel pour justifier le speedup ×6.9 d'optimatrix.

**[11] BLIS — implémentation moderne de BLAS**
> van Zee, F. G., & van de Geijn, R. A. (2015).
> *BLIS: A Framework for Rapidly Instantiating BLAS Functionality.*
> ACM TOMS, 41(3).
> https://github.com/flame/blis
> ★★★ — Pour montrer comment les pros font du GEMM haute performance.

---

## 4. Assembleur x86-64 et SIMD

**[12] Agner Fog — optimisation x86**
> Fog, A.
> *Optimizing software in C++* (et autres guides)
> https://www.agner.org/optimize/
> ★★ — LA référence pour comprendre les performances CPU. Guides gratuits :
> - "Instruction tables" — latences de chaque instruction
> - "Optimizing assembly" — techniques d'optimisation
> - "Microarchitecture" — fonctionnement interne du CPU

**[13] Intel Intrinsics Guide — référence AVX2**
> https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
> ★★ — Pour comprendre chaque instruction SIMD utilisée dans optimatrix.

**[14] Manuel Intel — la référence absolue**
> Intel Corporation.
> *Intel® 64 and IA-32 Architectures Software Developer's Manual.*
> Volumes 1-3. Disponible gratuitement sur intel.com.
> https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
> ★★★ — Reference, pas à lire linéairement. Consulter au besoin.

**[15] NASM Documentation**
> https://nasm.us/doc/nasmdoc.pdf
> ★ — Manuel de NASM, bien écrit, accessible.

---

## 5. Architecture des Transformers (contexte)

**[16] Attention is All You Need — Transformer original**
> Vaswani, A., Shazeer, N., Parmar, N., et al. (2017).
> *Attention Is All You Need.*
> NeurIPS 2017. arXiv:1706.03762
> https://arxiv.org/abs/1706.03762
> ★★ — Pour contextualiser pourquoi Mamba est une alternative aux Transformers.

---

## 6. Mathématiques — Algèbre Linéaire

**[17] Gilbert Strang — Introduction to Linear Algebra**
> Strang, G. (2016). *Introduction to Linear Algebra* (5ème éd.). Wellesley-Cambridge Press.
> ★ — Le meilleur livre d'algèbre linéaire pour débutants.
> Cours MIT gratuits : https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

**[18] Numerical Linear Algebra**
> Trefethen, L. N., & Bau, D. (1997).
> *Numerical Linear Algebra.* SIAM.
> ★★★ — Pour les décompositions (SVD, QR, LU). Niveau L3/Master.

---

## 7. Systèmes Embarqués et Optimisation Bas Niveau

**[19] What Every Programmer Should Know About Memory**
> Drepper, U. (2007).
> *What Every Programmer Should Know About Memory.*
> https://www.akkadia.org/drepress/cpumemory.pdf
> ★★ — Explique le cache, la hiérarchie mémoire. Essentiel pour comprendre
> pourquoi le tiling améliore GEMM et pourquoi CPU-only reste pertinent.

**[20] Computer Organization and Design**
> Patterson, D. A., & Hennessy, J. L. (2020).
> *Computer Organization and Design: ARM Edition* ou *RISC-V Edition.*
> Morgan Kaufmann.
> ★★ — Comprendre l'architecture matérielle pour mieux optimiser.

---

## 8. Pour la Théorie des Volontés (cadre philosophique)

**[21] Category Theory for Programmers**
> Milewski, B. (2019).
> *Category Theory for Programmers.*
> https://github.com/hmemcpy/milewski-ctfp-pdf
> ★★ — Gratuit. Lie la théorie des catégories à la programmation.
> Proche de la vision structurelle des Volontés comme vecteurs d'intention.

**[22] Conceptual Mathematics**
> Lawvere, F. W., & Schanuel, S. H. (2009).
> *Conceptual Mathematics: A First Introduction to Categories.* Cambridge University Press.
> ★ — Le plus accessible des livres de théorie des catégories. Niveau lycée/L1.

---

## Ressources en ligne gratuites

| Ressource | URL | Utilité |
|---|---|---|
| Papers With Code | paperswithcode.com | Trouver les codes des articles |
| arXiv | arxiv.org | Accès libre aux articles de recherche |
| Agner Fog | agner.org/optimize | Optimisation x86 |
| MIT OCW 18.06 | ocw.mit.edu | Cours d'algèbre linéaire (Strang) |
| Intel Intrinsics | intel.com/intrinsics-guide | Référence AVX2 |
| NASM Docs | nasm.us/doc | Manuel NASM |
| Compiler Explorer | godbolt.org | Voir le code ASM généré par GCC/Clang |

---

## Ordre de lecture recommandé pour le paper

```
Semaine 1 : [1]  Mamba — sections 1, 2, 3.1, 3.3, 3.4, annexe complexité
            [16] Attention (introduction uniquement — contexte)

Semaine 2 : [5]  VMamba — sections 1 et 3
            [6]  Mamba-ND — sections 1, 3 et 4 (différenciation clé)
            [3]  S4ND — sections 3 et 4 (fondements ND)

Semaine 3 : [12] Agner Fog "Optimizing assembly" — chapitres 1-4
            [19] Drepper memory — sections 1-3
            [10] Goto & van de Geijn — pour justifier le ×6.9

Semaine 4 : [8]  BLAS original (introduction)
            [17] Strang chapitres 1-2 (vecteurs, matrices, valeurs propres)

Pour la soutenance : citer [1], [5], [6], [3], [8], [12], [19]
```

---

## Ce que nous pouvons revendiquer comme contribution originale

**1. Première implémentation en ASM x86-64 pur d'un scan sélectif 2D**
avec ordonnancement wavefront diagonal — sans dépendances tierces, sans CUDA.

**2. Extension théorique du modèle Mamba à N dimensions**
via récurrence simultanée dans un DAG ND, distincte de l'approche
row-major alternée de Mamba-ND [6] et des 4 scans 1D de VMamba [5].

**3. Module optimatrix**
Bibliothèque de calcul tensoriel pour Mamba ND en assembleur pur,
déployable sur CPU standard (Intel Haswell 2013+ / AMD Ryzen 2017+),
sans GPU — pertinent pour les contextes embarqués et edge computing.

**4. Argument CPU-only**
Démonstration que le scan sélectif, séquentiel par nature,
bénéficie moins du parallélisme massif GPU que l'attention —
et qu'un moteur ASM bien vectorisé (×6.9 sur GEMM 64×64)
est une alternative viable pour l'inférence sur matériel accessible.
```
