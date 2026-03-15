/* ============================================================
 * convnd.c — Convolution ND separable depthwise : Forward + Backward
 *
 * Architecture :
 *   Le calcul vectorise est dans conv1d_avx2.asm (noyau AVX2).
 *   Ici, on orchestre la chaine de Conv1D axe par axe.
 *
 *   Forward  : Conv1D du dernier axe au premier, puis biais.
 *   Backward : backprop a travers chaque axe en ordre inverse,
 *              avec intermediaires sauvegardes (workspace) ou recomputes.
 *
 * API unifiee :
 *   convnd(p, CONVND_FORWARD,  ws)  — forward, sauvegarde inter dans ws
 *   convnd(p, CONVND_BACKWARD, ws)  — backward, lit inter depuis ws
 *   convnd(p, CONVND_COMPLETE, NULL) — forward + backward enchaines
 *
 * Parametrisation :
 *   ConvNDParams contient dims, ndims, D, K, bias (NULL = sans).
 *   ConvNDWorkspace gere les intermediaires pour decoupler fwd/bwd.
 *   ConvNDMode choisit le mode d'execution.
 * ============================================================ */

#include <stdlib.h>
#include <string.h>
#include "optimatrix.h"

/* ============================================================
 * Helpers
 * ============================================================ */

/* Produit des elements d'un tableau (produit vide = 1) */
static long product(const long *arr, long n) {
    long p = 1;
    for (long i = 0; i < n; i++) p *= arr[i];
    return p;
}

/* ============================================================
 * apply_conv1d_axis — Forward Conv1D AVX2 le long d'un axe
 *
 * Deux cas :
 *   - Axe contigu (dernier spatial) : appel direct, zero copie
 *   - Axe stride (autre) : gather [L,D], conv1d, scatter
 * ============================================================ */
static void apply_conv1d_axis(const float *src, float *dst, const float *kernel,
                              const long *dims, long ndims, long axis,
                              long D, long K) {
    long L = dims[axis];
    long inner_count = product(dims + axis + 1, ndims - 1 - axis);
    long outer_count = product(dims, axis);

    if (inner_count == 1) {
        /* Axe contigu : chaque batch est [L, D] en memoire */
        for (long b = 0; b < outer_count; b++) {
            Conv1DParams cp = {
                (float *)(src + b * L * D),
                (float *)kernel, NULL,
                dst + b * L * D,
                L, D, K
            };
            conv1d_depthwise_avx2(&cp);
        }
    } else {
        /* Axe stride : gather, conv1d, scatter */
        float *line_in  = malloc((size_t)(L * D) * sizeof(float));
        float *line_out = malloc((size_t)(L * D) * sizeof(float));

        for (long o = 0; o < outer_count; o++) {
            for (long ic = 0; ic < inner_count; ic++) {
                /* Gather : extraire une ligne [L, D] le long de l'axe */
                for (long t = 0; t < L; t++) {
                    long idx = (o * L * inner_count + t * inner_count + ic) * D;
                    memcpy(line_in + t * D, src + idx, (size_t)D * sizeof(float));
                }

                Conv1DParams cp = {
                    line_in, (float *)kernel, NULL, line_out,
                    L, D, K
                };
                conv1d_depthwise_avx2(&cp);

                /* Scatter : remettre en place */
                for (long t = 0; t < L; t++) {
                    long idx = (o * L * inner_count + t * inner_count + ic) * D;
                    memcpy(dst + idx, line_out + t * D, (size_t)D * sizeof(float));
                }
            }
        }

        free(line_in);
        free(line_out);
    }
}

/* ============================================================
 * conv1d_backward_c — Backward Conv1D scalaire
 *
 * Calcule dinput et dkernel a partir de dy.
 * Le backward d'une conv causale depthwise :
 *   dkernel[j,d] = sum_t dy[t,d] * input[t-K+1+j, d]
 *   dinput[s,d]  = sum_t dy[t,d] * kernel[t-s+K-1, d]   (correlation)
 *
 * Note : zero dinput, accumule dans dkernel (ne le zero pas).
 * ============================================================ */
static void conv1d_backward_c(const float *input, const float *kernel,
                               const float *dy, float *dinput, float *dkernel,
                               long L, long D, long K) {
    memset(dinput, 0, (size_t)(L * D) * sizeof(float));

    for (long t = 0; t < L; t++) {
        long j_start = K - 1 - t;
        if (j_start < 0) j_start = 0;

        for (long j = j_start; j < K; j++) {
            long src_t = t - K + 1 + j;
            for (long d = 0; d < D; d++) {
                float grad = dy[t * D + d];
                dkernel[j * D + d]    += grad * input[src_t * D + d];
                dinput[src_t * D + d] += grad * kernel[j * D + d];
            }
        }
    }
}

/* ============================================================
 * apply_conv1d_axis_backward — Backward Conv1D le long d'un axe
 *
 * Miroir de apply_conv1d_axis. Meme logique gather/scatter,
 * mais appelle conv1d_backward_c au lieu de conv1d_depthwise_avx2.
 *
 * Accumule dans dkernel_axis (ne le zero pas — l'appelant le fait).
 * ============================================================ */
static void apply_conv1d_axis_backward(
    const float *fwd_input,     /* entree a ce step du forward */
    const float *kernel,        /* kernel pour cet axe [K * D] */
    const float *dy,            /* gradient entrant [total_floats] */
    float *dinput,              /* gradient sortant [total_floats] */
    float *dkernel_axis,        /* gradient kernel [K * D], accumule */
    const long *dims, long ndims, long axis, long D, long K)
{
    long L = dims[axis];
    long inner_count = product(dims + axis + 1, ndims - 1 - axis);
    long outer_count = product(dims, axis);
    long KD = K * D;

    if (inner_count == 1) {
        /* Axe contigu : backward direct par batch */
        float *dk_tmp = calloc((size_t)KD, sizeof(float));

        for (long b = 0; b < outer_count; b++) {
            long off = b * L * D;
            memset(dk_tmp, 0, (size_t)KD * sizeof(float));
            conv1d_backward_c(fwd_input + off, kernel, dy + off,
                             dinput + off, dk_tmp, L, D, K);
            for (long i = 0; i < KD; i++)
                dkernel_axis[i] += dk_tmp[i];
        }

        free(dk_tmp);
    } else {
        /* Axe stride : gather dy + input, backward, scatter dinput */
        float *line_in  = malloc((size_t)(L * D) * sizeof(float));
        float *line_dy  = malloc((size_t)(L * D) * sizeof(float));
        float *line_din = malloc((size_t)(L * D) * sizeof(float));
        float *dk_tmp   = calloc((size_t)KD, sizeof(float));

        for (long o = 0; o < outer_count; o++) {
            for (long ic = 0; ic < inner_count; ic++) {
                /* Gather input et dy */
                for (long t = 0; t < L; t++) {
                    long idx = (o * L * inner_count + t * inner_count + ic) * D;
                    memcpy(line_in + t * D, fwd_input + idx, (size_t)D * sizeof(float));
                    memcpy(line_dy + t * D, dy + idx, (size_t)D * sizeof(float));
                }

                memset(dk_tmp, 0, (size_t)KD * sizeof(float));
                conv1d_backward_c(line_in, kernel, line_dy,
                                 line_din, dk_tmp, L, D, K);

                /* Scatter dinput */
                for (long t = 0; t < L; t++) {
                    long idx = (o * L * inner_count + t * inner_count + ic) * D;
                    memcpy(dinput + idx, line_din + t * D, (size_t)D * sizeof(float));
                }

                /* Accumulate dkernel */
                for (long i = 0; i < KD; i++)
                    dkernel_axis[i] += dk_tmp[i];
            }
        }

        free(line_in);
        free(line_dy);
        free(line_din);
        free(dk_tmp);
    }
}

/* ============================================================
 * Workspace : sauvegarde les intermediaires du forward
 * pour que le backward n'ait pas a les recomputer.
 * ============================================================ */

ConvNDWorkspace* convnd_workspace_create(const ConvNDParams *p) {
    if (!p || !p->dims || p->ndims <= 0) return NULL;

    ConvNDWorkspace *ws = malloc(sizeof(ConvNDWorkspace));
    ws->ndims = p->ndims;
    ws->total_floats = product(p->dims, p->ndims) * p->D;
    ws->inter = malloc((size_t)(p->ndims + 1) * sizeof(float *));

    for (long k = 0; k <= p->ndims; k++)
        ws->inter[k] = malloc((size_t)ws->total_floats * sizeof(float));

    return ws;
}

void convnd_workspace_free(ConvNDWorkspace *ws) {
    if (!ws) return;
    if (ws->inter) {
        for (long k = 0; k <= ws->ndims; k++)
            free(ws->inter[k]);
        free(ws->inter);
    }
    free(ws);
}

/* ============================================================
 * convnd_do_forward — Execute le forward, sauvegarde les
 * intermediaires dans ws si non NULL.
 *
 * Chaine :
 *   inter[0] = input
 *   inter[1] = apres conv sur axe ndims-1
 *   inter[2] = apres conv sur axe ndims-2
 *   ...
 *   inter[ndims] = output (avant biais)
 * ============================================================ */
static void convnd_do_forward(ConvNDParams *p, ConvNDWorkspace *ws) {
    long n = p->ndims;
    long tf = product(p->dims, n) * p->D;  /* total floats */
    size_t tf_bytes = (size_t)tf * sizeof(float);

    /* 1D : appel direct au noyau ASM */
    if (n == 1) {
        Conv1DParams cp = {
            p->input, p->kernel, p->bias, p->output,
            p->dims[0], p->D, p->K
        };
        conv1d_depthwise_avx2(&cp);

        /* Sauvegarder dans ws pour un eventuel backward */
        if (ws) {
            memcpy(ws->inter[0], p->input, tf_bytes);
            memcpy(ws->inter[1], p->output, tf_bytes);
        }
        return;
    }

    /* ND : chaine de Conv1D axe par axe */
    int own_bufs = (ws == NULL);
    float **inter;
    if (ws) {
        inter = ws->inter;
    } else {
        /* Allocation temporaire minimale : 2 ping-pong + pas d'intermediaires */
        inter = NULL;  /* pas utilise dans ce chemin */
    }

    if (own_bufs) {
        /* Mode forward pur, pas besoin d'intermediaires */
        float *buf_a = malloc(tf_bytes);
        float *buf_b = malloc(tf_bytes);
        memcpy(buf_a, p->input, tf_bytes);
        float *src = buf_a, *dst = buf_b;

        for (long axis = n - 1; axis >= 0; axis--) {
            float *kernel_axis = p->kernel + axis * p->K * p->D;
            apply_conv1d_axis(src, dst, kernel_axis, p->dims, n, axis, p->D, p->K);
            float *tmp = src; src = dst; dst = tmp;
        }

        memcpy(p->output, src, tf_bytes);
        free(buf_a);
        free(buf_b);
    } else {
        /* Mode avec workspace : sauvegarder chaque intermediaire */
        memcpy(inter[0], p->input, tf_bytes);

        for (long k = 0; k < n; k++) {
            long axis = n - 1 - k;
            float *kernel_axis = p->kernel + axis * p->K * p->D;
            apply_conv1d_axis(inter[k], inter[k + 1], kernel_axis,
                             p->dims, n, axis, p->D, p->K);
        }

        memcpy(p->output, inter[n], tf_bytes);
    }

    /* Biais applique une seule fois, apres toutes les convolutions */
    if (p->bias) {
        long total_spatial = product(p->dims, n);
        for (long i = 0; i < total_spatial; i++)
            for (long d = 0; d < p->D; d++)
                p->output[i * p->D + d] += p->bias[d];
    }
}

/* ============================================================
 * convnd_do_backward — Execute le backward.
 *
 * Si ws contient les intermediaires du forward, les utilise.
 * Sinon, les recompute en interne (plus lent mais autonome).
 *
 * Backward de la chaine :
 *   Forward : f_{n-1} → f_{n-2} → ... → f_0 → biais
 *   Backward : biais → f_0 → f_1 → ... → f_{n-1}
 *
 *   A chaque step k (k=0..n-1) :
 *     axis = k
 *     input du step forward = inter[n-1-k]
 *     backprop dy_courant a travers conv sur cet axe
 *     → produit dy_suivant et accumule dkernel[axis]
 * ============================================================ */
static void convnd_do_backward(ConvNDParams *p, ConvNDWorkspace *ws) {
    long n = p->ndims;
    long tf = product(p->dims, n) * p->D;
    size_t tf_bytes = (size_t)tf * sizeof(float);

    /* 1D : backward direct */
    if (n == 1) {
        memset(p->dkernel, 0, (size_t)(p->K * p->D) * sizeof(float));
        conv1d_backward_c(p->input, p->kernel, p->dy,
                          p->dinput, p->dkernel,
                          p->dims[0], p->D, p->K);
        if (p->dbias) {
            memset(p->dbias, 0, (size_t)p->D * sizeof(float));
            for (long t = 0; t < p->dims[0]; t++)
                for (long d = 0; d < p->D; d++)
                    p->dbias[d] += p->dy[t * p->D + d];
        }
        return;
    }

    /* ------ Obtenir les intermediaires ------ */
    int own_ws = 0;
    ConvNDWorkspace *w = ws;

    if (!w) {
        /* Pas de workspace fourni : recomputer le forward pour obtenir les inter */
        w = convnd_workspace_create(p);
        own_ws = 1;

        memcpy(w->inter[0], p->input, tf_bytes);
        for (long k = 0; k < n; k++) {
            long axis = n - 1 - k;
            float *kernel_axis = p->kernel + axis * p->K * p->D;
            apply_conv1d_axis(w->inter[k], w->inter[k + 1], kernel_axis,
                             p->dims, n, axis, p->D, p->K);
        }
    }

    /* ------ dbias : sum dy sur toutes les positions spatiales ------ */
    if (p->dbias) {
        memset(p->dbias, 0, (size_t)p->D * sizeof(float));
        long total_spatial = product(p->dims, n);
        for (long i = 0; i < total_spatial; i++)
            for (long d = 0; d < p->D; d++)
                p->dbias[d] += p->dy[i * p->D + d];
    }

    /* ------ Backprop a travers chaque axe ------ */
    memset(p->dkernel, 0, (size_t)(n * p->K * p->D) * sizeof(float));

    float *dy_cur = malloc(tf_bytes);
    float *dy_nxt = malloc(tf_bytes);
    memcpy(dy_cur, p->dy, tf_bytes);

    for (long k = 0; k < n; k++) {
        long axis = k;
        float *fwd_in = w->inter[n - 1 - k];
        float *dk_axis = p->dkernel + axis * p->K * p->D;

        apply_conv1d_axis_backward(fwd_in, p->kernel + axis * p->K * p->D,
                                   dy_cur, dy_nxt, dk_axis,
                                   p->dims, n, axis, p->D, p->K);

        /* Swap : dy_nxt devient le dy pour le prochain step */
        float *tmp = dy_cur; dy_cur = dy_nxt; dy_nxt = tmp;
    }

    /* dy_cur contient maintenant dinput */
    memcpy(p->dinput, dy_cur, tf_bytes);

    free(dy_cur);
    free(dy_nxt);
    if (own_ws) convnd_workspace_free(w);
}

/* ============================================================
 * convnd — Point d'entree unifie
 *
 * Modes :
 *   CONVND_FORWARD  : forward, sauvegarde inter dans ws si fourni
 *   CONVND_BACKWARD : backward, utilise ws si fourni sinon recompute
 *   CONVND_COMPLETE : forward + backward enchaines
 * ============================================================ */
void convnd(ConvNDParams *p, ConvNDMode mode, ConvNDWorkspace *ws) {
    if (!p || !p->dims || p->ndims <= 0) return;

    if (mode & CONVND_FORWARD) {
        if (!p->input || !p->output || !p->kernel) return;
        convnd_do_forward(p, ws);
    }

    if (mode & CONVND_BACKWARD) {
        if (!p->dy || !p->dinput || !p->dkernel || !p->kernel) return;
        convnd_do_backward(p, ws);
    }
}

/* ============================================================
 * Wrappers legacy — backward-compatible
 * ============================================================ */
void convnd_forward(ConvNDParams *p) {
    convnd(p, CONVND_FORWARD, NULL);
}

void convnd_backward(ConvNDParams *p) {
    convnd(p, CONVND_BACKWARD, NULL);
}
