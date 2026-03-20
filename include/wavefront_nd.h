#ifndef KMAMBA_WAVEFRONT_ND_H
#define KMAMBA_WAVEFRONT_ND_H

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * wavefront_nd.h — Générateur de wavefront ND borné et causal
 *
 * Primitive topologique générique pour tenseurs ND :
 *   niveau(idx) = idx[0] + idx[1] + ... + idx[ndims-1]
 *
 * Tous les points d'un même niveau sont indépendants et peuvent être traités
 * en parallèle si les dépendances de l'opérateur vont vers des niveaux < k.
 *
 * Cette API est pensée comme squelette commun pour scanND, convKD et tout
 * opérateur causal ND sur grille.
 * ============================================================================ */

typedef int (*WavefrontNDVisitFn)(
    const long *idx,
    long ndims,
    long level,
    long ordinal_in_level,
    void *user
);

/* Vérifie que dims[d] > 0 et ndims > 0. */
int wavefront_nd_validate_dims(const long *dims, long ndims);

/* Nombre total de points = produit des dimensions, ou -1 si invalide / overflow. */
long wavefront_nd_total_points(const long *dims, long ndims);

/* Niveau maximal = somme(dims[d] - 1), ou -1 si invalide / overflow. */
long wavefront_nd_max_level(const long *dims, long ndims);

/* Offset row-major d'un indice ND, ou -1 si invalide / overflow. */
long wavefront_nd_row_major_offset(const long *dims, const long *idx, long ndims);

/* Nombre de points dans un niveau donné, ou -1 si invalide. */
long wavefront_nd_level_size(const long *dims, long ndims, long level);

/* Itère un seul niveau.
 * idx_scratch peut être NULL ; sinon il doit pointer vers ndims éléments.
 * Retour :
 *   0  = parcours complet
 *   >0 = arrêt anticipé demandé par le callback (code retourné tel quel)
 *   -1 = paramètres invalides / allocation impossible
 */
int wavefront_nd_for_level(const long *dims, long ndims, long level,
                           long *idx_scratch,
                           WavefrontNDVisitFn visit,
                           void *user);

/* Itère tous les niveaux de 0 à max_level. Même convention de retour. */
int wavefront_nd_for_each_level(const long *dims, long ndims,
                                long *idx_scratch,
                                WavefrontNDVisitFn visit,
                                void *user);

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_WAVEFRONT_ND_H */
