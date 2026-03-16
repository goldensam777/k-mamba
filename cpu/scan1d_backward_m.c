#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "scan.h"

void scan1d_backward_m_generic(ScanBackwardMParams *p) {
    if (!p || !p->x || !p->A || !p->B || !p->C || !p->delta ||
        !p->h || !p->dy || !p->dx || !p->dA || !p->dB ||
        !p->dC || !p->ddelta || p->L <= 0 || p->D <= 0 || p->M <= 0)
        return;

    memset(p->dx,     0, (size_t)(p->L * p->D)        * sizeof(float));
    memset(p->dA,     0, (size_t)(p->D * p->M)        * sizeof(float));
    memset(p->dB,     0, (size_t)(p->L * p->D * p->M) * sizeof(float));
    memset(p->dC,     0, (size_t)(p->L * p->D * p->M) * sizeof(float));
    memset(p->ddelta, 0, (size_t)(p->L * p->D)        * sizeof(float));

    float *adj_h = (float *)calloc((size_t)(p->D * p->M), sizeof(float));
    if (!adj_h) return;

    for (long t = p->L - 1; t >= 0; t--) {
        for (long d = 0; d < p->D; d++) {
            long  td  = t * p->D + d;
            float dt  = p->delta[td];
            float xt  = p->x[td];
            float dy  = p->dy[td];
            float ddt = 0.0f;

            for (long m = 0; m < p->M; m++) {
                long  dm   = d * p->M + m;
                long  tdm  = (t * p->D + d) * p->M + m;
                float a    = p->A[dm];
                float b    = p->B[tdm];
                float c    = p->C[tdm];
                float dA_  = expf(dt * a);
                float h_prev = (t > 0) ? p->h[((t - 1) * p->D + d) * p->M + m]
                             : (p->h0  ? p->h0[dm] : 0.0f);

                float ah = adj_h[dm] + dy * c;

                p->dC[tdm]  += dy  * p->h[tdm];
                p->dB[tdm]  += ah  * dt * xt;
                p->dA[dm]   += ah  * dt * dA_ * h_prev;
                p->dx[td]   += ah  * dt * b;
                ddt         += ah  * (a * dA_ * h_prev + b * xt);
                adj_h[dm]    = ah  * dA_;
            }

            p->ddelta[td] += ddt;
        }
    }

    free(adj_h);
}
