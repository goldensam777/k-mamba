/* convnd.h — Convolution ND native wavefront */

#ifndef CONVND_H
#define CONVND_H

#include <stddef.h>
#include "wavefront_plan.h"

/* Convolution ND params - version wavefront unifiée */
typedef struct {
    float *input;           /* Input tensor [prod(dims), D] */
    const float *kernel;    /* Full kernel [K^ndims, D] */
    const float *bias;      /* Bias [D] or NULL */
    float *output;          /* Output tensor [prod(dims), D] */
    float *dy;              /* Gradient w.r.t. output (for backward) */
    float *dinput;          /* Gradient w.r.t. input */
    float *dkernel;         /* Gradient w.r.t. kernel */
    float *dbias;           /* Gradient w.r.t. bias */
    const long *dims;       /* Spatial shape [ndims] */
    long ndims;             /* Number of spatial dimensions */
    long D;                 /* Depth/channels */
    long K;                 /* Kernel size along every axis */
} ConvNDParams;

typedef enum {
    CONVND_FORWARD = 1,     /* Forward pass only */
    CONVND_BACKWARD = 2,    /* Backward pass only */
    CONVND_COMPLETE = 3     /* Forward + Backward */
} ConvNDMode;

/* Kernel volume K^ndims */
long convnd_kernel_volume(long ndims, long K);

/* Forward pass with wavefront - parallélisé intra-niveau */
void convnd_forward_wavefront(ConvNDParams *p, const KMWavefrontPlan *plan);

/* Backward pass with wavefront */
void convnd_backward_wavefront(ConvNDParams *p, const KMWavefrontPlan *plan);

/* Unified entry point */
void convnd(ConvNDParams *p, ConvNDMode mode);

#endif
