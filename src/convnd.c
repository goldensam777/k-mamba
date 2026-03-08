#include <stdlib.h>
#include <string.h>
#include "optimatrix.h"

// Helper function to compute linear index in N-dimensional tensor
static long nd_index(long *indices, long *dims, long ndims) {
    long idx = 0;
    long stride = 1;
    for (long i = ndims - 1; i >= 0; i--) {
        idx += indices[i] * stride;
        stride *= dims[i];
    }
    return idx;
}

// Helper function to compute total size
static long nd_total_size(long *dims, long ndims) {
    long size = 1;
    for (long i = 0; i < ndims; i++) {
        size *= dims[i];
    }
    return size;
}

void convnd_forward(ConvNDParams *p) {
    if (!p || !p->input || !p->output || !p->dims || p->ndims <= 0) {
        return;
    }
    
    // For 1D case, use existing scan1d implementation
    if (p->ndims == 1) {
        long L = p->dims[0];
        long stride = L * p->D;
        
        // Create temporary buffers for each M dimension
        float *h = malloc(p->M * stride * sizeof(float));
        float *x = malloc(p->M * stride * sizeof(float));
        
        // Initialize with input data
        memcpy(x, p->input, p->M * stride * sizeof(float));
        
        // Process each M dimension
        for (long m = 0; m < p->M; m++) {
            ScanBackwardMParams scan_params = {
                .x = x + m * stride,
                .A = p->A + m * p->M * p->D,  // Simplified for now
                .B = p->B + m * p->M * p->D,
                .C = p->C + m * p->M * p->D,
                .delta = p->delta + m * stride,
                .h0 = p->h0 ? p->h0 + m * p->D : NULL,
                .h = h + m * stride,
                .dy = NULL,  // Forward pass
                .dx = p->output + m * stride,
                .dA = NULL,
                .dB = NULL,
                .dC = NULL,
                .ddelta = NULL,
                .L = L,
                .D = p->D,
                .M = 1  // Process one M at a time
            };
            
            // For forward pass, we need to implement scan1d_forward
            // For now, copy input to output (placeholder)
            memcpy(p->output + m * stride, x + m * stride, stride * sizeof(float));
        }
        
        free(h);
        free(x);
        return;
    }
    
    // For N>1 dimensions, implement wavefront pattern
    long total_size = nd_total_size(p->dims, p->ndims);
    
    // Simple implementation: process each spatial position independently
    #pragma omp parallel for
    for (long i = 0; i < total_size; i++) {
        // Copy input to output for now (placeholder)
        for (long d = 0; d < p->D; d++) {
            p->output[i * p->D + d] = p->input[i * p->D + d];
        }
    }
}

void convnd_backward(ConvNDParams *p) {
    if (!p || !p->input || !p->output || !p->dims || p->ndims <= 0) {
        return;
    }
    
    // For 1D case, use scan1d_backward_m_generic
    if (p->ndims == 1) {
        long L = p->dims[0];
        long stride = L * p->D;
        
        // Create temporary buffers
        float *dy = malloc(p->M * stride * sizeof(float));
        float *dx = malloc(p->M * stride * sizeof(float));
        float *dA = malloc(p->M * p->M * p->D * sizeof(float));
        float *dB = malloc(p->M * p->M * p->D * sizeof(float));
        float *dC = malloc(p->M * p->M * p->D * sizeof(float));
        float *ddelta = malloc(p->M * stride * sizeof(float));
        
        // Initialize dy (gradient from output)
        memset(dy, 0, p->M * stride * sizeof(float));
        
        // Process each M dimension
        for (long m = 0; m < p->M; m++) {
            ScanBackwardMParams scan_params = {
                .x = p->input + m * stride,
                .A = p->A + m * p->M * p->D,
                .B = p->B + m * p->M * p->D,
                .C = p->C + m * p->M * p->D,
                .delta = p->delta + m * stride,
                .h0 = p->h0 ? p->h0 + m * p->D : NULL,
                .h = NULL,  // Will be allocated internally
                .dy = dy + m * stride,
                .dx = dx + m * stride,
                .dA = dA + m * p->D,
                .dB = dB + m * p->D,
                .dC = dC + m * p->D,
                .ddelta = ddelta + m * stride,
                .L = L,
                .D = p->D,
                .M = 1
            };
            
            scan1d_backward_m_generic(&scan_params);
        }
        
        free(dy);
        free(dx);
        free(dA);
        free(dB);
        free(dC);
        free(ddelta);
        return;
    }
    
    // For N>1 dimensions, implement N-dimensional backward pass
    long total_size = nd_total_size(p->dims, p->ndims);
    
    // Simple implementation: zero gradients (placeholder)
    #pragma omp parallel for
    for (long i = 0; i < total_size; i++) {
        for (long d = 0; d < p->D; d++) {
            p->output[i * p->D + d] = 0.0f;
        }
    }
}
