/* train_lm.c — Character-level language model training on a text corpus.
 *
 * Usage: ./mamba_lm_train [dataset] [model_out] [epochs]
 *   dataset   : path to text file (default: data/conversations.txt)
 *   model_out : path to checkpoint file (default: lm_checkpoint.bin)
 *   epochs    : number of training epochs (default: 200)
 *
 * The entire corpus is read into memory and trained using a sliding
 * window of seq_len characters (stride = seq_len, non-overlapping).
 * Loss and perplexity are printed each epoch.
 */

#include "mamba.h"
#include "lm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DEFAULT_DATASET  "data/conversations.txt"
#define DEFAULT_MODEL    "lm_checkpoint.bin"
#define DEFAULT_EPOCHS   200

static char *load_corpus_bytes(const char *path, size_t *out_len) {
    FILE *fin = fopen(path, "rb");
    char *buf;
    long file_size;
    size_t nread;

    if (!fin) return NULL;
    if (fseek(fin, 0, SEEK_END) != 0) {
        fclose(fin);
        return NULL;
    }
    file_size = ftell(fin);
    if (file_size < 0) {
        fclose(fin);
        return NULL;
    }
    rewind(fin);

    buf = (char *)malloc((size_t)file_size + 1);
    if (!buf) {
        fclose(fin);
        return NULL;
    }

    nread = fread(buf, 1, (size_t)file_size, fin);
    if (nread != (size_t)file_size && ferror(fin)) {
        free(buf);
        fclose(fin);
        return NULL;
    }
    fclose(fin);

    buf[nread] = '\0';
    if (out_len) *out_len = nread;
    return buf;
}

int main(int argc, char *argv[]) {
    const char *dataset_path = (argc > 1) ? argv[1] : DEFAULT_DATASET;
    const char *model_path   = (argc > 2) ? argv[2] : DEFAULT_MODEL;
    int         num_epochs   = (argc > 3) ? atoi(argv[3]) : DEFAULT_EPOCHS;

    srand((unsigned)time(NULL));

    /* ---- 1. Load corpus ---- */
    size_t corpus_len = 0;
    char *corpus = load_corpus_bytes(dataset_path, &corpus_len);
    if (!corpus) {
        fprintf(stderr, "ERROR: cannot open dataset '%s'\n", dataset_path);
        return 1;
    }
    printf("Corpus loaded: %zu bytes from '%s'\n", corpus_len, dataset_path);

    /* ---- 2. Create LM ---- */
    LMConfig cfg = lm_default_config();
    LM *lm = lm_create(&cfg);
    if (!lm) {
        fprintf(stderr, "ERROR: failed to create LM\n");
        free(corpus);
        return 1;
    }
    printf("Model config: vocab=%zu dim=%zu state=%zu seq=%zu gen=%zu\n",
           cfg.vocab_size, cfg.dim, cfg.state_size, cfg.seq_len, cfg.max_gen_len);
    printf("Trainable params: %zu (~%.3fM)\n",
           lm_num_parameters(&cfg), (double)lm_num_parameters(&cfg) / 1e6);

    /* Try to load existing checkpoint to resume training */
    if (lm_load(lm, model_path) == 0) {
        printf("Resumed from checkpoint '%s'\n", model_path);
    } else {
        FILE *ckpt = fopen(model_path, "rb");
        if (ckpt) {
            fclose(ckpt);
            printf("Checkpoint '%s' ignored (missing or incompatible config)\n",
                   model_path);
        }
    }

    /* ---- 3. Optimizer config ---- */
    OptimConfig opt = {
        .lr          = 1e-3f,
        .mu          = 0.9f,
        .beta2       = 0.999f,
        .eps         = 1e-8f,
        .clip_norm   = 1.0f,
        .weight_decay= 1e-5f
    };

    size_t seq_len = cfg.seq_len;
    size_t n_windows = 0;
    if (corpus_len > seq_len + 1)
        n_windows = (corpus_len - 1) / seq_len;

    if (n_windows == 0) {
        fprintf(stderr, "ERROR: corpus too short (need > %zu chars)\n", seq_len + 1);
        lm_free(lm);
        free(corpus);
        return 1;
    }
    printf("Windows per epoch: %zu  (seq_len=%zu)\n", n_windows, seq_len);
    printf("Training for %d epochs...\n\n", num_epochs);

    int *in_seq  = (int *)malloc(seq_len * sizeof(int));
    int *tgt_seq = (int *)malloc(seq_len * sizeof(int));
    if (!in_seq || !tgt_seq) {
        lm_free(lm); free(corpus); free(in_seq); free(tgt_seq);
        return 1;
    }

    /* ---- 4. Training loop ---- */
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double total_loss = 0.0;
        size_t count      = 0;

        for (size_t pos = 0; pos + seq_len < corpus_len; pos += seq_len) {
            for (size_t t = 0; t < seq_len; t++) {
                in_seq[t]  = (unsigned char)corpus[pos + t];
                tgt_seq[t] = (unsigned char)corpus[pos + t + 1];
            }
            real_t loss = lm_train_step(lm, in_seq, tgt_seq, &opt);
            total_loss += (double)loss;
            count++;
        }

        double avg_loss = (count > 0) ? total_loss / (double)count : 0.0;
        double ppl      = exp(avg_loss);
        printf("Epoch %3d  loss=%.4f  ppl=%.2f\n", epoch, avg_loss, ppl);
        fflush(stdout);

        /* Checkpoint every 10 epochs */
        if ((epoch + 1) % 10 == 0) {
            if (lm_save(lm, model_path) == 0)
                printf("  -> checkpoint saved to '%s'\n", model_path);
        }
    }

    /* Final save */
    lm_save(lm, model_path);
    printf("\nTraining complete. Model saved to '%s'\n", model_path);

    free(in_seq);
    free(tgt_seq);
    free(corpus);
    lm_free(lm);
    return 0;
}
