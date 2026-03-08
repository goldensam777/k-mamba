#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "optimatrix.h"

static size_t scan_tdm_index(long t, long d, long m, long D, long M) {
    return (size_t)(((t * D) + d) * M + m);
}

static size_t scan_dm_index(long d, long m, long M) {
    return (size_t)(d * M + m);
}

static int scan1d_backward_shared_validate(const ScanBackwardSharedParams *p) {
    return p && p->x && p->A && p->B && p->C && p->delta &&
           p->h && p->dy && p->dx && p->dA && p->dB &&
           p->dC && p->ddelta && p->L > 0 && p->D > 0;
}

static void scan1d_backward_m1_shared_bc_scalar(ScanBackwardSharedParams *p) {
    float *adj_h;

    memset(p->dx, 0, (size_t)(p->L * p->D) * sizeof(float));
    memset(p->dA, 0, (size_t)p->D * sizeof(float));
    memset(p->dB, 0, (size_t)p->D * sizeof(float));
    memset(p->dC, 0, (size_t)p->D * sizeof(float));
    memset(p->ddelta, 0, (size_t)p->L * sizeof(float));

    adj_h = (float *)calloc((size_t)p->D, sizeof(float));
    if (!adj_h) return;

    for (long t = p->L - 1; t >= 0; t--) {
        float dt = p->delta[t];
        float ddt = 0.0f;

        for (long d = 0; d < p->D; d++) {
            long td = t * p->D + d;
            float xt = p->x[td];
            float dy = p->dy[td];
            float a = p->A[d];
            float b = p->B[d];
            float c = p->C[d];
            float dA = p->A_diag ? p->A_diag[td] : expf(dt * a);
            float h_prev = 0.0f;
            float ah;

            if (t > 0) {
                h_prev = p->h[(t - 1) * p->D + d];
            } else if (p->h0) {
                h_prev = p->h0[d];
            }

            ah = adj_h[d] + dy * c;

            p->dC[d] += dy * p->h[td];
            p->dB[d] += ah * dt * xt;
            p->dA[d] += ah * dt * dA * h_prev;
            p->dx[td] += ah * dt * b;
            ddt += ah * (a * dA * h_prev + b * xt);
            adj_h[d] = ah * dA;
        }

        p->ddelta[t] += ddt;
    }

    free(adj_h);
}

static void scan1d_backward_m1_shared_bc_avx2(ScanBackwardSharedParams *p) {
    float *adj_h;
    long D = p->D;

    memset(p->dx, 0, (size_t)(p->L * p->D) * sizeof(float));
    memset(p->dA, 0, (size_t)p->D * sizeof(float));
    memset(p->dB, 0, (size_t)p->D * sizeof(float));
    memset(p->dC, 0, (size_t)p->D * sizeof(float));
    memset(p->ddelta, 0, (size_t)p->L * sizeof(float));

    adj_h = (float *)calloc((size_t)p->D, sizeof(float));
    if (!adj_h) return;

    for (long t = p->L - 1; t >= 0; t--) {
        long base = t * D;
        const float *x_row = p->x + base;
        const float *dy_row = p->dy + base;
        const float *h_row = p->h + base;
        const float *h_prev_row = NULL;
        const float *a_diag_row = p->A_diag ? p->A_diag + base : NULL;
        __m256 dt_v = _mm256_set1_ps(p->delta[t]);
        __m256 ddt_v = _mm256_setzero_ps();
        float ddt = 0.0f;
        long d = 0;

        if (t > 0) h_prev_row = p->h + (t - 1) * D;
        else if (p->h0) h_prev_row = p->h0;

        for (; d + 8 <= D; d += 8) {
            __m256 xt_v = _mm256_loadu_ps(x_row + d);
            __m256 dy_v = _mm256_loadu_ps(dy_row + d);
            __m256 a_v = _mm256_loadu_ps(p->A + d);
            __m256 b_v = _mm256_loadu_ps(p->B + d);
            __m256 c_v = _mm256_loadu_ps(p->C + d);
            __m256 h_v = _mm256_loadu_ps(h_row + d);
            __m256 adj_v = _mm256_loadu_ps(adj_h + d);
            __m256 h_prev_v = h_prev_row ? _mm256_loadu_ps(h_prev_row + d)
                                         : _mm256_setzero_ps();
            __m256 dA_v;
            __m256 ah_v;
            __m256 tmp_v;
            __m256 acc_v;

            if (a_diag_row) {
                dA_v = _mm256_loadu_ps(a_diag_row + d);
            } else {
                float tmp_dA[8];
                for (int lane = 0; lane < 8; lane++) {
                    tmp_dA[lane] = expf(p->delta[t] * p->A[d + lane]);
                }
                dA_v = _mm256_loadu_ps(tmp_dA);
            }

            ah_v = _mm256_add_ps(adj_v, _mm256_mul_ps(dy_v, c_v));

            acc_v = _mm256_loadu_ps(p->dC + d);
            acc_v = _mm256_add_ps(acc_v, _mm256_mul_ps(dy_v, h_v));
            _mm256_storeu_ps(p->dC + d, acc_v);

            tmp_v = _mm256_mul_ps(_mm256_mul_ps(ah_v, dt_v), xt_v);
            acc_v = _mm256_loadu_ps(p->dB + d);
            acc_v = _mm256_add_ps(acc_v, tmp_v);
            _mm256_storeu_ps(p->dB + d, acc_v);

            tmp_v = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(ah_v, dt_v), dA_v), h_prev_v);
            acc_v = _mm256_loadu_ps(p->dA + d);
            acc_v = _mm256_add_ps(acc_v, tmp_v);
            _mm256_storeu_ps(p->dA + d, acc_v);

            _mm256_storeu_ps(p->dx + base + d,
                             _mm256_mul_ps(_mm256_mul_ps(ah_v, dt_v), b_v));

            tmp_v = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(a_v, dA_v), h_prev_v),
                                  _mm256_mul_ps(b_v, xt_v));
            ddt_v = _mm256_add_ps(ddt_v, _mm256_mul_ps(ah_v, tmp_v));

            _mm256_storeu_ps(adj_h + d, _mm256_mul_ps(ah_v, dA_v));
        }

        {
            float lanes[8];
            _mm256_storeu_ps(lanes, ddt_v);
            for (int lane = 0; lane < 8; lane++) ddt += lanes[lane];
        }

        for (; d < D; d++) {
            long td = base + d;
            float xt = x_row[d];
            float dy = dy_row[d];
            float a = p->A[d];
            float b = p->B[d];
            float c = p->C[d];
            float dA = a_diag_row ? a_diag_row[d] : expf(p->delta[t] * a);
            float h_prev = h_prev_row ? h_prev_row[d] : 0.0f;
            float ah = adj_h[d] + dy * c;

            p->dC[d] += dy * h_row[d];
            p->dB[d] += ah * p->delta[t] * xt;
            p->dA[d] += ah * p->delta[t] * dA * h_prev;
            p->dx[td] = ah * p->delta[t] * b;
            ddt += ah * (a * dA * h_prev + b * xt);
            adj_h[d] = ah * dA;
        }

        p->ddelta[t] += ddt;
    }

    free(adj_h);
}

void scan1d_backward_m1_shared_bc(ScanBackwardSharedParams *p) {
    if (!scan1d_backward_shared_validate(p)) return;

    if (p->A_diag && p->D >= 8) {
        scan1d_backward_m1_shared_bc_avx2(p);
        return;
    }

    scan1d_backward_m1_shared_bc_scalar(p);
}

static int scan1d_backward_validate(const ScanBackwardParams *p) {
    return p && p->x && p->A && p->B && p->C && p->delta &&
           p->h && p->dy && p->dx && p->dA && p->dB &&
           p->dC && p->ddelta && p->L > 0 && p->D > 0 && p->M > 0;
}

static void scan1d_backward_zero_outputs(ScanBackwardParams *p) {
    memset(p->dx, 0, (size_t)(p->L * p->D) * sizeof(float));
    memset(p->dA, 0, (size_t)(p->D * p->M) * sizeof(float));
    memset(p->dB, 0, (size_t)(p->L * p->D * p->M) * sizeof(float));
    memset(p->dC, 0, (size_t)(p->L * p->D * p->M) * sizeof(float));
    memset(p->ddelta, 0, (size_t)(p->L * p->D) * sizeof(float));
}

static void scan1d_backward_m1(ScanBackwardParams *p) {
    float *adj_h;

    scan1d_backward_zero_outputs(p);

    adj_h = (float *)calloc((size_t)p->D, sizeof(float));
    if (!adj_h) return;

    for (long t = p->L - 1; t >= 0; t--) {
        for (long d = 0; d < p->D; d++) {
            long td = t * p->D + d;
            float dt = p->delta[td];
            float xt = p->x[td];
            float dy = p->dy[td];
            float a = p->A[d];
            float b = p->B[td];
            float c = p->C[td];
            float dA = expf(dt * a);
            float h_prev = 0.0f;
            float ah;

            if (t > 0) {
                h_prev = p->h[(t - 1) * p->D + d];
            } else if (p->h0) {
                h_prev = p->h0[d];
            }

            ah = adj_h[d] + dy * c;

            p->dC[td] += dy * p->h[td];
            p->dB[td] += ah * dt * xt;
            p->dA[d] += ah * dt * dA * h_prev;
            p->dx[td] += ah * dt * b;
            p->ddelta[td] += ah * (a * dA * h_prev + b * xt);
            adj_h[d] = ah * dA;
        }
    }

    free(adj_h);
}

static void scan1d_backward_generic(ScanBackwardParams *p) {
    float *adj_h;

    scan1d_backward_zero_outputs(p);

    adj_h = (float *)calloc((size_t)(p->D * p->M), sizeof(float));
    if (!adj_h) return;

    for (long t = p->L - 1; t >= 0; t--) {
        for (long d = 0; d < p->D; d++) {
            long td = t * p->D + d;
            float dt = p->delta[td];
            float xt = p->x[td];
            float dy = p->dy[td];
            float ddt = 0.0f;

            for (long m = 0; m < p->M; m++) {
                size_t dm = scan_dm_index(d, m, p->M);
                size_t tdm = scan_tdm_index(t, d, m, p->D, p->M);
                float a = p->A[dm];
                float b = p->B[tdm];
                float c = p->C[tdm];
                float dA = expf(dt * a);
                float h_prev = 0.0f;
                float ah;

                if (t > 0) {
                    h_prev = p->h[scan_tdm_index(t - 1, d, m, p->D, p->M)];
                } else if (p->h0) {
                    h_prev = p->h0[dm];
                }

                ah = adj_h[dm] + dy * c;

                p->dC[tdm] += dy * p->h[tdm];
                p->dB[tdm] += ah * dt * xt;
                p->dA[dm] += ah * dt * dA * h_prev;
                p->dx[td] += ah * dt * b;
                ddt += ah * (a * dA * h_prev + b * xt);
                adj_h[dm] = ah * dA;
            }

            p->ddelta[td] += ddt;
        }
    }

    free(adj_h);
}

void scan1d_backward(ScanBackwardParams *p) {
    if (!scan1d_backward_validate(p)) return;

    if (p->M == 1) {
        scan1d_backward_m1(p);
        return;
    }

    scan1d_backward_generic(p);
}
