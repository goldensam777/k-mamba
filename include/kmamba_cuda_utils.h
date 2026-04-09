/* kmamba_cuda_utils.h — CUDA runtime utilities and automatic GPU dispatch */

#ifndef KMAMBA_CUDA_UTILS_H
#define KMAMBA_CUDA_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Runtime GPU Detection
 * ═══════════════════════════════════════════════════════════════════════ */

/* Returns 1 if CUDA GPU is available and usable, 0 otherwise.
 * Checks: CUDA runtime init, device count, device properties */
int kmamba_cuda_available(void);

/* Get number of CUDA devices. Returns 0 if CUDA not available. */
int kmamba_cuda_device_count(void);

/* Get current active device ID (-1 if none). */
int kmamba_cuda_current_device(void);

/* ═══════════════════════════════════════════════════════════════════════
 * Automatic Dispatch Configuration
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    KMAMBA_BACKEND_AUTO = 0,    /* Use GPU if available, else CPU */
    KMAMBA_BACKEND_CPU = 1,   /* Force CPU */
    KMAMBA_BACKEND_GPU = 2    /* Force GPU (error if unavailable) */
} KMambaBackend;

/* Global backend preference (default: AUTO).
 * Can be set via environment variable KMAMBA_BACKEND=auto|cpu|gpu */
extern KMambaBackend kmamba_backend_preference;

/* Initialize backend from environment. Call once at startup. */
void kmamba_backend_init(void);

/* Get effective backend for current operation.
 * Respects preference and runtime GPU availability. */
KMambaBackend kmamba_backend_select(void);

/* ═══════════════════════════════════════════════════════════════════════
 * Helper macros for dispatch
 * ═══════════════════════════════════════════════════════════════════════ */

#ifdef KMAMBA_BUILD_CUDA
#define KMAMBA_IF_CUDA(expr) expr
#else
#define KMAMBA_IF_CUDA(expr) ((void)0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* KMAMBA_CUDA_UTILS_H */
