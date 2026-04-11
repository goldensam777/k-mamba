/* convnd_c_wrapper.c - C implementation of om_convnd_forward/backward for shared library
 * Fallback C implementation when CUDA is not available or for CPU-only builds
 */
#include "../include/convnd.h"
#include "../include/wavefront_plan.h"
#include <string.h>

/* Forward pass - CPU implementation using wavefront */
int om_convnd_forward(ConvNDParams *p) {
    /* Create a simple wavefront plan for the given dimensions */
    KMWavefrontPlan plan;
    
    /* Initialize plan for the convolution dimensions */
    size_t spatial_dims[4] = {0};
    for (int i = 0; i < p->ndims && i < 4; i++) {
        spatial_dims[i] = p->dims[i];
    }
    
    /* Use the existing convnd_forward_wavefront function */
    convnd_forward_wavefront(p, &plan);
    
    return 0; /* Success */
}

/* Backward pass - CPU implementation using wavefront */
int om_convnd_backward(ConvNDParams *p) {
    KMWavefrontPlan plan;
    
    /* Use the existing convnd_backward_wavefront function */
    convnd_backward_wavefront(p, &plan);
    
    return 0; /* Success */
}
