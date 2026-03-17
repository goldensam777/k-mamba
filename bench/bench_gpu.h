/*
 * bench_gpu.h — GPU benchmark functions (CUDA event timing)
 *
 * Called from bench_paper.c when compiled with CUDA support.
 * All functions fill an array of `repeat` timing values (ms).
 */

#ifndef BENCH_GPU_H
#define BENCH_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

/* Scan1D forward — auto-selects Blelloch (L<=1024) or sequential */
void bench_gpu_scan1d(int L, int D, int M,
                      int warmup, int repeat, float *out_ms);

/* Scan1D forward — forced sequential kernel */
void bench_gpu_scan1d_seq(int L, int D, int M,
                          int warmup, int repeat, float *out_ms);

#ifdef __cplusplus
}
#endif

#endif /* BENCH_GPU_H */
