/* kmamba_cuda_utils.c — CUDA runtime detection and automatic dispatch */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kmamba_cuda_utils.h"

/* Global backend preference */
KMambaBackend kmamba_backend_preference = KMAMBA_BACKEND_AUTO;

/* ═══════════════════════════════════════════════════════════════════════
 * Runtime GPU Detection
 * ═══════════════════════════════════════════════════════════════════════ */

#ifdef KMAMBA_BUILD_CUDA

#include <cuda_runtime.h>

int kmamba_cuda_available(void) {
    cudaError_t err;
    int device_count = 0;
    
    /* Try to initialize CUDA runtime */
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        return 0;
    }
    
    /* Check if we can set a device */
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return 0;
    }
    
    /* Verify device properties */
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return 0;
    }
    
    /* Minimum compute capability check (sm_50 = Maxwell) */
    if (prop.major < 5) {
        fprintf(stderr, "[k-mamba] Warning: GPU compute capability %d.%d too old (need >= 5.0)\n",
                prop.major, prop.minor);
        return 0;
    }
    
    return 1;
}

int kmamba_cuda_device_count(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count;
}

int kmamba_cuda_current_device(void) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return -1;
    }
    return device;
}

#else /* !KMAMBA_BUILD_CUDA */

int kmamba_cuda_available(void) { return 0; }
int kmamba_cuda_device_count(void) { return 0; }
int kmamba_cuda_current_device(void) { return -1; }

#endif /* KMAMBA_BUILD_CUDA */

/* ═══════════════════════════════════════════════════════════════════════
 * Backend Configuration
 * ═══════════════════════════════════════════════════════════════════════ */

void kmamba_backend_init(void) {
    const char *env = getenv("KMAMBA_BACKEND");
    
    if (!env) {
        kmamba_backend_preference = KMAMBA_BACKEND_AUTO;
        return;
    }
    
    if (strcasecmp(env, "cpu") == 0) {
        kmamba_backend_preference = KMAMBA_BACKEND_CPU;
        fprintf(stderr, "[k-mamba] Backend forced to CPU via environment\n");
    } else if (strcasecmp(env, "gpu") == 0 || strcasecmp(env, "cuda") == 0) {
        kmamba_backend_preference = KMAMBA_BACKEND_GPU;
        fprintf(stderr, "[k-mamba] Backend forced to GPU via environment\n");
    } else if (strcasecmp(env, "auto") == 0) {
        kmamba_backend_preference = KMAMBA_BACKEND_AUTO;
    } else {
        fprintf(stderr, "[k-mamba] Warning: Unknown KMAMBA_BACKEND='%s', using auto\n", env);
        kmamba_backend_preference = KMAMBA_BACKEND_AUTO;
    }
}

KMambaBackend kmamba_backend_select(void) {
    switch (kmamba_backend_preference) {
        case KMAMBA_BACKEND_CPU:
            return KMAMBA_BACKEND_CPU;
            
        case KMAMBA_BACKEND_GPU:
            if (!kmamba_cuda_available()) {
                fprintf(stderr, "[k-mamba] Error: GPU requested but not available\n");
                /* Fall back to CPU with warning */
                return KMAMBA_BACKEND_CPU;
            }
            return KMAMBA_BACKEND_GPU;
            
        case KMAMBA_BACKEND_AUTO:
        default:
            /* GPU if available, else CPU */
            if (kmamba_cuda_available()) {
                return KMAMBA_BACKEND_GPU;
            }
            return KMAMBA_BACKEND_CPU;
    }
}
