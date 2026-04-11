/* scan1d_c_wrapper.c - C implementation of scan1d for shared library
 * Fallback C implementation when ASM is not available (PIC issues)
 */
#include <math.h>
#include <string.h>
#include "../include/scan.h"

/* Simple C implementation of selective scan 1D */
void scan1d(ScanParams *p);
void scan2d(Scan2DParams *p);

void scan1d(ScanParams *p) {
    int L = p->L;
    int D = p->D;
    int M = p->M;
    
    float *h = p->h;  /* [D, M] state */
    
    for (int t = 0; t < L; t++) {
        for (int d = 0; d < D; d++) {
            float x = p->x[t * D + d];
            float dt = p->delta[t * D + d];
            
            for (int m = 0; m < M; m++) {
                float A = p->A[d * M + m];
                float B = p->B[t * D * M + d * M + m];
                float C = p->C[t * D * M + d * M + m];
                
                /* Selective scan recurrence */
                float A_bar = expf(A * dt);
                float B_bar = B * dt;
                
                /* Update state */
                h[d * M + m] = A_bar * h[d * M + m] + B_bar * x;
                
                /* Output */
                p->y[t * D * M + d * M + m] = C * h[d * M + m];
            }
        }
    }
}

/* Simple C implementation of selective scan 2D (wavefront) */
void scan2d(Scan2DParams *p) {
    int d1 = p->d1;  /* First dimension (e.g., variables) */
    int d2 = p->d2;  /* Second dimension (e.g., time) */
    int D = p->D;    /* Feature dimension */
    int M = p->M;    /* State size */
    
    float *h = p->h;  /* [D, M] state */
    
    /* 2D wavefront scan: iterate along anti-diagonals */
    for (int s = 0; s < d1 + d2 - 1; s++) {
        for (int i = 0; i <= s && i < d1; i++) {
            int j = s - i;
            if (j >= d2) continue;
            
            /* Index for this position */
            int idx = i * d2 + j;
            
            for (int d = 0; d < D; d++) {
                float x = p->x[idx * D + d];
                float dt1 = p->delta1[idx * D + d];
                float dt2 = p->delta2[idx * D + d];
                
                for (int m = 0; m < M; m++) {
                    float A1 = p->A1[d * M + m];
                    float A2 = p->A2[d * M + m];
                    float B = p->B[idx * D * M + d * M + m];
                    float C = p->C[idx * D * M + d * M + m];
                    
                    /* Selective scan recurrence - 2D with both dimensions */
                    float A1_bar = expf(A1 * dt1);
                    float A2_bar = expf(A2 * dt2);
                    float B_bar = B * (dt1 + dt2) * 0.5f;
                    
                    /* Update state with combined effect */
                    h[d * M + m] = A1_bar * A2_bar * h[d * M + m] + B_bar * x;
                    
                    /* Output */
                    p->y[idx * D * M + d * M + m] = C * h[d * M + m];
                }
            }
        }
    }
}
