#ifndef MAMBA_SCAN_CUDA_H
#define MAMBA_SCAN_CUDA_H

#include <stddef.h>
#include "scan_nd.h"

/* ============================================================
 * Mamba-specific CUDA kernels
 * ============================================================ */

/* Scan 1D forward with Blelloch optimization */
void mamba_scan1d_cuda_forward(
    const float *d_x,  const float *d_A,
    const float *d_B,  const float *d_C,
    const float *d_dt,
    float *d_y, float *d_h,
    int L, int D, int M
);

/* Scan 1D backward */
void mamba_scan1d_cuda_backward(
    const float *d_dy,
    const float *d_x,  const float *d_A,
    const float *d_B,  const float *d_C,
    const float *d_dt, const float *d_h,
    float *d_dx, float *d_dA,
    float *d_dB, float *d_dC,
    float *d_ddt,
    int L, int D, int M
);

/* Scan ND forward générique sur GPU */
int mamba_scannd_cuda_forward(ScanNDParams *p);

/* MambaBlock forward */
void mamba_block_cuda_forward(
    const float *x, const float *A_diag, const float *B_bar, const float *C,
    const float *dt, float *h, float *y,
    int seq_len, int state_size, int dim
);

/* MambaBlock backward */
void mamba_block_cuda_backward(
    const float *x, const float *A_diag, const float *B_bar, const float *C,
    const float *dt, const float *h, const float *dy,
    float *dx, float *dA_diag, float *dB_bar, float *dC, float *ddt,
    int seq_len, int state_size, int dim
);

/* MUON optimizer */
void mamba_muon_cuda_step(
    float *params, const float *grads, int n,
    float lr, float mu, float beta2, float eps, float clip_norm
);

#endif
