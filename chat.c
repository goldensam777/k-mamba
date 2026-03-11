#include "bissimamba.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int sample_from_probs(const float *probs, size_t n, float r) {
    float c = 0.0f;
    for (size_t i = 0; i < n; i++) {
        c += probs[i];
        if (r <= c) return (int)i;
    }
    return (int)(n - 1);
}

static void softmax_temp(float *probs, const float *logits, size_t n, float temp) {
    float maxv = logits[0] / temp;
    for (size_t i = 1; i < n; i++) {
        float v = logits[i] / temp;
        if (v > maxv) maxv = v;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = expf(logits[i] / temp - maxv);
        probs[i] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++) probs[i] *= inv;
}

int main(int argc, char **argv) {
    const char *ckpt_path = (argc > 1) ? argv[1] : "checkpoint.bin";
    int max_new = (argc > 2) ? atoi(argv[2]) : 512;
    float temp = (argc > 3) ? (float)atof(argv[3]) : 1.0f;

    MBOptimConfig dummy = {0};
    BissiMamba *m = bissimamba_load(ckpt_path, 0, &dummy, 0.0f, 0.0f);
    if (!m) {
        fprintf(stderr, "Failed to load checkpoint: %s\n", ckpt_path);
        return 1;
    }

    size_t V = m->cfg.vocab_size;
    size_t L = m->cfg.seq_len;
    float *logits = (float *)calloc(L * V, sizeof(float));
    float *probs  = (float *)calloc(V, sizeof(float));
    uint8_t *ctx  = (uint8_t *)calloc(L, 1);
    if (!logits || !probs || !ctx) {
        fprintf(stderr, "OOM\n");
        free(logits); free(probs); free(ctx);
        bissimamba_free(m);
        return 1;
    }

    printf("BissiMamba — %s (seq=%zu, dim=%zu, layers=%zu)\n",
           ckpt_path, m->cfg.seq_len, m->cfg.dim, m->cfg.n_layers);
    printf("Ctrl-D to quit.\n");

    char line[4096];
    while (1) {
        printf("\n> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;

        size_t n = strlen(line);
        if (n && line[n - 1] == '\n') line[--n] = 0;

        memset(ctx, 0, L);
        size_t start = (n > L) ? (n - L) : 0;
        size_t clen = n - start;
        for (size_t i = 0; i < clen; i++) ctx[L - clen + i] = (uint8_t)line[start + i];

        fwrite(line, 1, n, stdout);
        fflush(stdout);

        for (int step = 0; step < max_new; step++) {
            if (bissimamba_forward(m, ctx, logits) != 0) break;
            const float *last = &logits[(L - 1) * V];
            softmax_temp(probs, last, V, temp);
            float r = (float)rand() / (float)RAND_MAX;
            int tok = sample_from_probs(probs, V, r);

            memmove(ctx, ctx + 1, L - 1);
            ctx[L - 1] = (uint8_t)tok;
            putchar(tok);
            fflush(stdout);

            if (tok == '\n') break;
        }
        printf("\n");
    }

    free(logits);
    free(probs);
    free(ctx);
    bissimamba_free(m);
    return 0;
}
