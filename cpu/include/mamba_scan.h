#ifndef MAMBA_SCAN_H
#define MAMBA_SCAN_H

#include <stddef.h>

typedef struct {
    float *x;
    float *A;
    float *B;
    float *C;
    float *delta;
    float *h;
    float *y;
    long L;
    long D;
    long M;
} MambaScan1DParams;

void mamba_scan1d_forward(MambaScan1DParams *p);

#endif
