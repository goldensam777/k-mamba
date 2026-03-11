#include "bissimamba.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *read_entire_file(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }

    unsigned char *buf = (unsigned char *)malloc((size_t)n);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *n_out = (size_t)n;
    return buf;
}

int main(int argc, char **argv) {
    const char *corpus_path = (argc > 1) ? argv[1] : "data/conversations.txt";
    const char *ckpt_path   = (argc > 2) ? argv[2] : "checkpoint.bin";
    int epochs              = (argc > 3) ? atoi(argv[3]) : 200;
    long steps_override     = (argc > 4) ? atol(argv[4]) : 0;

    printf("corpus: %s\n", corpus_path);
    printf("ckpt:   %s\n", ckpt_path);
    printf("epochs: %d\n", epochs);

    size_t nbytes = 0;
    unsigned char *data = read_entire_file(corpus_path, &nbytes);
    if (!data || nbytes < 1024) {
        fprintf(stderr, "Failed to read corpus or corpus too small\n");
        free(data);
        return 1;
    }

    BissiMambaConfig cfg = {
        .vocab_size  = 256,
        .dim         = 384,
        .state_size  = 1024,
        .seq_len     = 128,
        .n_layers    = 1,
        .dt_scale    = 1.0f,
        .dt_min      = 0.001f,
        .dt_max      = 0.1f
    };

    MBOptimConfig opt = {
        .lr           = 1e-3f,
        .mu           = 0.9f,
        .beta2        = 0.999f,
        .eps          = 1e-8f,
        .clip_norm    = 1.0f,
        .weight_decay = 1e-5f
    };

    BissiMamba *m = NULL;
    {
        FILE *probe = fopen(ckpt_path, "rb");
        if (probe) {
            fclose(probe);
            m = bissimamba_load(ckpt_path, 1, &opt, 1e-3f, 1e-5f);
            if (m) printf("Loaded checkpoint\n");
        }
    }
    if (!m) {
        m = bissimamba_create(&cfg);
        if (!m) { fprintf(stderr, "Failed to create model\n"); free(data); return 1; }
        bissimamba_init(m, 1234);
        bissimamba_enable_training(m, &opt, 1e-3f, 1e-5f);
        printf("Initialized new model\n");
    }

    size_t batch_size = 8;
    size_t Lp1 = cfg.seq_len + 1;
    uint8_t *batch = (uint8_t *)malloc(batch_size * Lp1);
    if (!batch) { fprintf(stderr, "OOM\n"); bissimamba_free(m); free(data); return 1; }

    size_t steps_per_epoch = 200;
    if (nbytes > (cfg.seq_len + 2) * 1000) steps_per_epoch = 1000;
    if (steps_override > 0) steps_per_epoch = (size_t)steps_override;

    printf("batch=%zu, steps/epoch=%zu\n", batch_size, steps_per_epoch);

    for (int e = 0; e < epochs; e++) {
        double avg = 0.0;
        for (size_t s = 0; s < steps_per_epoch; s++) {
            for (size_t b = 0; b < batch_size; b++) {
                size_t start = (size_t)rand() % (nbytes - Lp1);
                memcpy(&batch[b * Lp1], &data[start], Lp1);
            }
            float loss = bissimamba_train_batch(m, batch, batch_size);
            avg += (double)loss;
        }
        avg /= (double)steps_per_epoch;
        printf("epoch %d/%d loss=%.4f\n", e + 1, epochs, (float)avg);

        if ((e + 1) % 10 == 0) {
            if (bissimamba_save(m, ckpt_path) == 0) printf("saved\n");
        }
    }

    bissimamba_save(m, ckpt_path);
    bissimamba_free(m);
    free(batch);
    free(data);
    return 0;
}
