#ifndef KMAMBA_SCAN_ND_H
#define KMAMBA_SCAN_ND_H

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * scan_nd.h — Scan sélectif ND de référence sur grille causale
 *
 * Layout mémoire :
 *   dims   : [ndims]
 *   x      : [prod(dims), D]
 *   A      : [ndims, D, M]
 *   B      : [prod(dims), D, M]
 *   C      : [prod(dims), D, M]
 *   delta  : [ndims, prod(dims), D]
 *   h      : [prod(dims), D, M]
 *   y      : [prod(dims), D]
 *
 * Récurrence :
 *   dt_bar(idx,d) = mean_axis delta(axis,idx,d)
 *
 *   h(idx,d,m) = dt_bar(idx,d) * B(idx,d,m) * x(idx,d)
 *              + sum_axis exp(delta(axis,idx,d) * A(axis,d,m))
 *                         * h(idx - e_axis, d, m)
 *
 *   y(idx,d)   = sum_m C(idx,d,m) * h(idx,d,m)
 *
 * Les dépendances suivent l'ordre causal borné donné par wavefront_nd :
 *   level(idx) = sum(idx[axis]).
 *
 * Cette définition réduit exactement à :
 *   - scan1d si ndims = 1
 *   - scan2d si ndims = 2, avec dt_bar = 0.5 * (delta1 + delta2)
 * ============================================================================ */

typedef struct {
    const long  *dims;
    long         ndims;
    long         D;
    long         M;
    const float *x;
    const float *A;
    const float *B;
    const float *C;
    const float *delta;
    float       *h;
    float       *y;
} ScanNDParams;

/* Vérifie que les pointeurs / tailles sont valides. */
int scannd_validate(const ScanNDParams *p);

/* Implémentation de référence (scalaire, ordonnancement wavefront ND). */
int scannd_ref(ScanNDParams *p);

/* Alias courant vers l'implémentation de référence. */
int scannd(ScanNDParams *p);

#ifdef __CUDACC__
/* Backend CUDA générique piloté par wavefronts. Les pointeurs x/A/B/C/delta/h/y
 * doivent être des device pointers ; dims reste côté hôte. */
int om_scannd_forward(ScanNDParams *p);
#endif

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_SCAN_ND_H */
