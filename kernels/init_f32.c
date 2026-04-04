/*
 * init_f32.c — Initialization utilities
 */

#include "kmamba_kernels.h"
#include <stdlib.h>

/* Simple LCG random number generator (no libc dependency beyond rand) */
static inline float random_uniform(float min, float max) {
    /* Using rand() for simplicity - could be replaced with PCG or xorshift */
    float r = (float)rand() / (float)RAND_MAX;
    return min + r * (max - min);
}

void init_xavier_uniform_f32(float *W, size_t fan_in, size_t fan_out, unsigned int seed) {
    srand(seed);
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    size_t n = fan_in * fan_out;
    for (size_t i = 0; i < n; i++) {
        W[i] = random_uniform(-limit, limit);
    }
}

void init_kaiming_uniform_f32(float *W, size_t fan_in, unsigned int seed) {
    srand(seed);
    float limit = sqrtf(6.0f / (float)fan_in);
    size_t n = fan_in;
    for (size_t i = 0; i < n; i++) {
        W[i] = random_uniform(-limit, limit);
    }
}
