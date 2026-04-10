#define _POSIX_C_SOURCE 200809L

/*
 * kmamba_cuda — Instance k-mamba CUDA
 *
 * BPE Language Model basé sur k-mamba.
 * Config CUDA : ~500M paramètres, mixed precision, logs CSV et corpus borné.
 *
 * Usage:
 *   ./kmamba_cuda                          # entraîne sur texte intégré, puis génère
 *   ./kmamba_cuda train <data.txt> [ckpt] [log-prefix]
 *   ./kmamba_cuda gen   <ckpt> [prompt]    # génère depuis un checkpoint
 *   ./kmamba_cuda chat  <ckpt>             # REPL interactif (chatbot)
 */

extern "C" {
#include "kmamba.h"
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================
 * CONFIG — instance CUDA (~500M params)
 * ============================================================ */
#define VOCAB_SIZE    32768
#define DIM           1024
#define STATE_SIZE    2048
#define N_LAYERS      24
#define SEQ_LEN       1024
#define BATCH_SIZE    4
#define N_EPOCHS      100
#define SAVE_EVERY    1
#define LR            5e-4f
#define LR_EMBED_HEAD 1e-4f
#define WEIGHT_DECAY  1e-5f
#define CLIP_NORM     1.0f
#define MOMENTUM      0.9f
#define BETA2         0.999f
#define EPS           1e-8f
#define TEMPERATURE   0.8f
#define GEN_LEN       512
#define CHAT_MAX_RESP 256
#define SEED          42
#define CHINCHILLA_TOKENS_PER_PARAM 20u
#define VAL_PERCENT   5u

/* Format conversation pour corpus */
#define CHAT_USER     "[speaker001:] "
#define CHAT_BOT      "[speaker002:] "
#define REPL_YOU      "speaker001"
#define REPL_CP       "speaker002"

/* ============================================================
 * Texte intégré — fallback
 * ============================================================ */
static const char *BUILTIN_TEXT =
    "Les systemes doivent operer par intentions qui convergent vers un equilibre, "
    "pas par instructions sequentielles. Chaque MambaBlock est une Volonte qui "
    "transforme la sequence. L'optimiseur arbitre les tensions entre gradients. "
    "Un bug n'est pas une erreur d'instruction, c'est un conflit de Volontes non resolu. "
    "On est assez grand pour voir des unites, il faut voir des structures. "
    "Ego Sum Optimus Optimus. "
    "State Space Models offrent une alternative lineaire aux Transformers. "
    "Le scan selectif choisit quoi retenir a chaque pas de temps. "
    "k-mamba est une bibliotheque C pure — zero Python, zero PyTorch. "
    "La separation Volontes-Puissance guide l'architecture. "
    "IFRI-UAC Benin — Systemes Embarques et IoT. ";

static KMambaConfig make_cuda_config(void) {
    KMambaConfig cfg = {
        .vocab_size   = VOCAB_SIZE,
        .dim          = DIM,
        .state_size   = STATE_SIZE,
        .seq_len      = SEQ_LEN,
        .n_layers     = N_LAYERS,
        .mimo_rank    = 1,
        .dt_scale     = 1.0f,
        .dt_min       = 0.001f,
        .dt_max       = 0.1f,
        .spatial_ndims = 0,
        .use_convnd   = 0,
        .convnd_K     = 1,
        .convnd_ndims = 1,
        .weight_tying = 1,
    };
    return cfg;
}

/* Note: CUDA build requires -DKMAMBA_BUILD_CUDA */

/* ============================================================
 * Utilitaires
 * ============================================================ */
typedef struct {
    char *data;
    size_t len;
} Dataset;

static Dataset load_file_prefix(const char *path, size_t max_bytes, size_t *source_len_out) {
    Dataset ds = {NULL, 0};
    size_t file_len = 0;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[erreur] impossible d'ouvrir %s\n", path); return ds; }
    fseek(f, 0, SEEK_END);
    file_len = (size_t)ftell(f);
    rewind(f);
    if (source_len_out) *source_len_out = file_len;
    ds.len = (max_bytes > 0 && file_len > max_bytes) ? max_bytes : file_len;
    ds.data = (char *)malloc(ds.len + 1);
    if (!ds.data) { fclose(f); return ds; }
    if (fread(ds.data, 1, ds.len, f) != ds.len) { free(ds.data); ds.data = NULL; ds.len = 0; }
    ds.data[ds.len] = '\0';
    fclose(f);
    return ds;
}

static Dataset from_string(const char *s) {
    Dataset ds;
    ds.len  = strlen(s);
    ds.data = (char *)malloc(ds.len + 1);
    memcpy(ds.data, s, ds.len);
    ds.data[ds.len] = '\0';
    return ds;
}

static char *xstrdup_local(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *copy = (char *)malloc(n);
    if (!copy) return NULL;
    memcpy(copy, s, n);
    return copy;
}

static char *make_log_path(const char *prefix, const char *kind) {
    if (!prefix || !kind) return NULL;
    size_t n = strlen(prefix) + 1 + strlen(kind) + 4 + 1;
    char *path = (char *)malloc(n);
    if (!path) return NULL;
    snprintf(path, n, "%s.%s.csv", prefix, kind);
    return path;
}

static FILE *open_csv_log(const char *path, const char *header) {
    if (!path || !header) return NULL;
    FILE *f = fopen(path, "a+");
    if (!f) { fprintf(stderr, "[warning] impossible d'ouvrir %s\n", path); return NULL; }
    long pos = 0;
    if (fseek(f, 0, SEEK_END) == 0) pos = ftell(f);
    if (pos == 0) { fprintf(f, "%s\n", header); fflush(f); }
    return f;
}

static double elapsed_ms(struct timespec *t0) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (double)(t1.tv_sec - t0->tv_sec) * 1000.0
         + (double)(t1.tv_nsec - t0->tv_nsec) / 1e6;
}

static float safe_perplexity(float loss) {
    if (!isfinite(loss)) return NAN;
    if (loss > 20.0f) loss = 20.0f;
    return expf(loss);
}

/* Norme L2 d'un vecteur float */
static float l2_norm_f32(const float *x, size_t n) {
    double acc = 0.0;
    if (!x) return 0.0f;
    for (size_t i = 0; i < n; i++) { double v = (double)x[i]; acc += v * v; }
    return sqrtf((float)acc);
}

/* Norme L2 de la différence entre deux vecteurs */
static float l2_diff_norm_f32(const float *a, const float *b, size_t n) {
    double acc = 0.0;
    if (!a || !b) return 0.0f;
    for (size_t i = 0; i < n; i++) { double d = (double)a[i] - (double)b[i]; acc += d * d; }
    return sqrtf((float)acc);
}

/* Calcul loss depuis logits (pour évaluation) */
static float compute_loss_from_logits(const float *logits, const uint32_t *targets,
                                      size_t seq_len, size_t vocab_size) {
    float loss = 0.0f;
    for (size_t t = 0; t < seq_len; t++) {
        const float *row = &logits[t * vocab_size];
        float maxv = row[0];
        for (size_t v = 1; v < vocab_size; v++) if (row[v] > maxv) maxv = row[v];
        float sum = 0.0f;
        for (size_t v = 0; v < vocab_size; v++) sum += expf(row[v] - maxv);
        float p = expf(row[targets[t]] - maxv) / sum;
        if (p < 1e-20f) p = 1e-20f;
        loss += -logf(p);
    }
    return loss / (float)seq_len;
}

/* Évaluation sur dataset (validation) */
static float evaluate_dataset(KMamba *m, const Dataset *ds, size_t max_windows) {
    if (!m || !ds || !ds->data) return NAN;
    size_t seq_tokens = m->cfg.seq_len + 1;
    if (ds->len < seq_tokens) return NAN;
    size_t available = ds->len - seq_tokens + 1;
    size_t windows = available < max_windows ? available : max_windows;
    if (windows == 0) windows = 1;
    size_t stride = windows > 1 ? (available - 1) / (windows - 1) : 0;
    float *logits = (float *)malloc(m->cfg.seq_len * m->cfg.vocab_size * sizeof(float));
    if (!logits) return NAN;
    float loss_sum = 0.0f;
    for (size_t i = 0; i < windows; i++) {
        size_t start = i * stride;
        size_t chunk_len = m->cfg.seq_len * 8;
        if (start + chunk_len > ds->len) chunk_len = ds->len - start;
        char chunk[4096];
        size_t copy = chunk_len < sizeof(chunk)-1 ? chunk_len : sizeof(chunk)-2;
        memcpy(chunk, ds->data + start, copy);
        chunk[copy] = '\0';
        size_t tok_len = 0;
        uint32_t *enc = kmamba_encode(chunk, &tok_len);
        uint32_t *window = (uint32_t*)calloc(seq_tokens, sizeof(uint32_t));
        for (size_t j = 0; j < seq_tokens; j++) window[j] = (j < tok_len) ? enc[j] : 0;
        kmamba_free_tokens(enc, tok_len);
        kmamba_forward(m, window, logits);
        loss_sum += compute_loss_from_logits(logits, window + 1, m->cfg.seq_len, m->cfg.vocab_size);
        free(window);
    }
    free(logits);
    return loss_sum / (float)windows;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static float lr_schedule(float lr_max, size_t step, size_t warmup_steps, size_t total_steps) {
    if (step == 0) return 0.0f;
    if (step < warmup_steps) return lr_max * (float)step / (float)warmup_steps;
    float progress = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
    float cosine = 0.5f * (1.0f + cosf(M_PI * progress));
    float lr_min = lr_max * 0.1f;
    return lr_min + (lr_max - lr_min) * cosine;
}

/* ============================================================
 * Sampling et génération
 * ============================================================ */
static uint32_t sample_token(const float *logits, size_t vocab_size, float temp) {
    float probs[32768];
    float maxv = logits[0];
    for (size_t i = 1; i < vocab_size; i++) if (logits[i] > maxv) maxv = logits[i];
    if (temp < 1e-4f) {
        size_t best = 0;
        for (size_t i = 1; i < vocab_size; i++) if (logits[i] > logits[best]) best = i;
        return (uint32_t)best;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - maxv) / temp);
        sum += probs[i];
    }
    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float acc = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        acc += probs[i];
        if (acc >= r) return (uint32_t)i;
    }
    return (uint32_t)(vocab_size - 1);
}

static void generate(KMamba *m, const char *prompt, size_t gen_len) {
    size_t L = m->cfg.seq_len;
    size_t V = m->cfg.vocab_size;
    uint32_t *ctx = (uint32_t*)calloc(L, sizeof(uint32_t));
    float *logits = (float*)calloc(L * V, sizeof(float));
    
    if (prompt && strlen(prompt) > 0) {
        size_t tok_len = 0;
        uint32_t *prompt_tok = kmamba_encode(prompt, &tok_len);
        size_t copy = tok_len < L ? tok_len : L;
        memcpy(ctx + L - copy, prompt_tok, copy * sizeof(uint32_t));
        kmamba_free_tokens(prompt_tok, tok_len);
    }
    
    printf("\n[génération");
    if (prompt && strlen(prompt) > 0) printf(" — prompt: \"%s\"", prompt);
    printf("]\n");
    if (prompt && strlen(prompt) > 0) printf("%s", prompt);
    
    for (size_t step = 0; step < gen_len; step++) {
        kmamba_forward(m, ctx, logits);
        uint32_t next = sample_token(&logits[(L - 1) * V], V, TEMPERATURE);
        char *tok_str = kmamba_decode(&next, 1);
        if (tok_str) {
            printf("%s", tok_str);
            fflush(stdout);
            kmamba_free_string(tok_str);
        }
        memmove(ctx, ctx + 1, (L - 1) * sizeof(uint32_t));
        ctx[L - 1] = next;
    }
    printf("\n[fin génération]\n");
    free(ctx); free(logits);
}

/* ============================================================
 * Chat REPL
 * ============================================================ */
static void chat_repl(KMamba *m) {
    size_t L = m->cfg.seq_len;
    size_t V = m->cfg.vocab_size;
    float *logits = (float*)calloc(L * V, sizeof(float));
    if (!logits) { fprintf(stderr, "[erreur] malloc logits\n"); return; }
    uint32_t *ctx = (uint32_t*)calloc(L, sizeof(uint32_t));
    if (!ctx) { free(logits); return; }
    char line[1024];

    printf("\n  k-mamba — session interactive\n");
    printf("  Ctrl+D ou 'quit' pour quitter\n");
    printf("  ─────────────────────────────\n\n");

    while (1) {
        printf("\033[1;32m" REPL_YOU "\033[0m> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) { printf("\n"); break; }
        size_t llen = strlen(line);
        if (llen > 0 && line[llen - 1] == '\n') line[--llen] = '\0';
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) break;
        if (llen == 0) continue;

        char inject_buf[sizeof(line) + 32];
        int inject_len = snprintf(inject_buf, sizeof(inject_buf), "%s%s\n%s", CHAT_USER, line, CHAT_BOT);
        size_t inj = (size_t)inject_len;
        if (inj >= L) {
            memcpy(ctx, (uint8_t *)inject_buf + inj - L, L);
        } else {
            memmove(ctx, ctx + inj, (L - inj) * sizeof(uint32_t));
            memcpy(ctx + (L - inj), inject_buf, inj);
        }

        printf("\033[1;34m" REPL_CP "\033[0m> ");
        fflush(stdout);

        for (size_t step = 0; step < CHAT_MAX_RESP; step++) {
            kmamba_forward(m, ctx, logits);
            uint32_t next = sample_token(&logits[(L - 1) * V], V, TEMPERATURE);
            if (next == '\n' || next == '\r') {
                memmove(ctx, ctx + 1, (L - 1) * sizeof(uint32_t));
                ctx[L - 1] = '\n';
                break;
            }
            char *tok_str = kmamba_decode(&next, 1);
            if (tok_str) {
                printf("%s", tok_str);
                fflush(stdout);
                kmamba_free_string(tok_str);
            }
            memmove(ctx, ctx + 1, (L - 1) * sizeof(uint32_t));
            ctx[L - 1] = next;
        }
        printf("\n\n");
    }
    printf("  [session terminée]\n");
    free(ctx); free(logits);
}

/* ============================================================
 * Affichage config
 * ============================================================ */
static void print_config(void) {
    size_t total_params = (size_t)(2.0f * VOCAB_SIZE * DIM + N_LAYERS * (4.0f * DIM * DIM + 2.0f * STATE_SIZE * DIM));
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║         k-mamba — Instance CPU (AVX2)            ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  vocab_size  : %-5d                            ║\n", VOCAB_SIZE);
    printf("║  dim         : %-5d                            ║\n", DIM);
    printf("║  state_size  : %-5d                            ║\n", STATE_SIZE);
    printf("║  n_layers    : %-5d                            ║\n", N_LAYERS);
    printf("║  seq_len     : %-5d                            ║\n", SEQ_LEN);
    printf("║  batch_size  : %-5d                            ║\n", BATCH_SIZE);
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  params      : %7.0fK                          ║\n", (float)total_params / 1000.0f);
    printf("╚══════════════════════════════════════════════════╝\n\n");
}

/* ============================================================
 * Sampling batch
 * ============================================================ */
/* Sampling avec offset pour boucler sur dataset (Chinchilla) */
static void sample_batch_offset(const Dataset *ds, uint32_t *batch, size_t batch_size, 
                                 size_t seq_len, size_t *offset) {
    size_t seq_tokens = seq_len + 1;
    for (size_t b = 0; b < batch_size; b++) {
        /* Offset avance dans le dataset, boucle si nécessaire */
        size_t start = *offset % (ds->len > 100 ? ds->len - 100 : 1);
        *offset += seq_len * 4; /* ~4.5 chars/token, avance d'environ 1 seq */
        size_t chunk_len = seq_len * 8;
        if (start + chunk_len > ds->len) chunk_len = ds->len - start;
        char chunk[4096];
        size_t copy = chunk_len < sizeof(chunk)-1 ? chunk_len : sizeof(chunk)-2;
        memcpy(chunk, ds->data + start, copy);
        chunk[copy] = '\0';
        size_t tok_len = 0;
        uint32_t *enc = kmamba_encode(chunk, &tok_len);
        for (size_t i = 0; i <= seq_len; i++) {
            batch[b * seq_tokens + i] = (i < tok_len) ? enc[i] : 0;
        }
        kmamba_free_tokens(enc, tok_len);
    }
}

/* ============================================================
 * main
 * ============================================================ */
int main(int argc, char *argv[]) {
    srand(SEED);
    print_config();

    /* Mode et arguments */
    int mode_gen = 0, mode_chat = 0;
    const char *data_path = NULL, *ckpt_path = NULL, *prompt_str = NULL, *log_prefix_arg = NULL;
    size_t total_tokens_target = 0; /* 0 = auto (1 epoch), sinon Chinchilla */

    if (argc >= 2 && strcmp(argv[1], "chat") == 0) {
        mode_chat = 1;
        ckpt_path = argc >= 3 ? argv[2] : NULL;
        if (!ckpt_path) { fprintf(stderr, "usage: %s chat <ckpt>\n", argv[0]); return 1; }
    } else if (argc >= 2 && strcmp(argv[1], "gen") == 0) {
        mode_gen = 1;
        ckpt_path = argc >= 3 ? argv[2] : NULL;
        prompt_str = argc >= 4 ? argv[3] : NULL;
        if (!ckpt_path) { fprintf(stderr, "usage: %s gen <ckpt> [prompt]\n", argv[0]); return 1; }
    } else if (argc >= 2 && strcmp(argv[1], "train") == 0) {
        data_path = argc >= 3 ? argv[2] : NULL;
        ckpt_path = argc >= 4 ? argv[3] : NULL;
        log_prefix_arg = argc >= 5 ? argv[4] : NULL;
        /* Parse --total-tokens XXXXXX */
        for (int i = 5; i < argc - 1; i++) {
            if (strcmp(argv[i], "--total-tokens") == 0) {
                total_tokens_target = (size_t)atoll(argv[i+1]);
            }
        }
    } else if (argc == 2) {
        data_path = argv[1];
    }

    /* Mode chat */
    if (mode_chat) {
        KMamba *m = kmamba_load(ckpt_path, 0, NULL, 0.0f, 0.0f);
        if (!m) { fprintf(stderr, "[erreur] impossible de charger %s\n", ckpt_path); return 1; }
        printf("[checkpoint : %s]\n", ckpt_path);
        chat_repl(m);
        kmamba_free(m);
        return 0;
    }

    /* Mode génération */
    if (mode_gen) {
        KMamba *m = kmamba_load(ckpt_path, 0, NULL, 0.0f, 0.0f);
        if (!m) { fprintf(stderr, "[erreur] impossible de charger %s\n", ckpt_path); return 1; }
        printf("[checkpoint chargé : %s]\n", ckpt_path);
        generate(m, prompt_str, GEN_LEN);
        kmamba_free(m);
        return 0;
    }

    /* Mode entraînement */
    Dataset ds;
    size_t source_len = 0;
    size_t max_bytes = (size_t)(2.0f * VOCAB_SIZE * DIM * CHINCHILLA_TOKENS_PER_PARAM);
    if (data_path) {
        ds = load_file_prefix(data_path, max_bytes, &source_len);
        if (!ds.data) return 1;
        printf("[données] %s — %zu / %zu bytes utilisés\n", data_path, ds.len, source_len);
    } else {
        ds = from_string(BUILTIN_TEXT);
        printf("[données] texte intégré — %zu bytes\n", ds.len);
    }

    if (ds.len < 100) {
        fprintf(stderr, "[erreur] données trop courtes\n");
        free(ds.data);
        return 1;
    }

    /* Création modèle */
    KMambaConfig cfg = make_cuda_config();
    KMamba *m = NULL;
    MBOptimConfig opt = {LR, MOMENTUM, BETA2, EPS, CLIP_NORM, WEIGHT_DECAY};

    if (ckpt_path) {
        m = kmamba_load(ckpt_path, 1, &opt, LR_EMBED_HEAD, WEIGHT_DECAY);
        if (m) printf("[checkpoint repris : %s]\n\n", ckpt_path);
    }
    if (!m) {
        m = kmamba_create(&cfg);
        kmamba_init(m, SEED);
        kmamba_enable_training(m, &opt, LR_EMBED_HEAD, WEIGHT_DECAY);
        printf("[modèle initialisé (Xavier, seed=%d)]\n\n", SEED);
    }

    /* Split train/val */
    Dataset train_ds = ds;
    Dataset val_ds = {NULL, 0};
    size_t val_len = 0;
    if (ds.len >= 20 * (SEQ_LEN + 1)) {
        val_len = (ds.len * VAL_PERCENT) / 100u;
        if (val_len < (SEQ_LEN + 1)) val_len = (SEQ_LEN + 1);
        if (val_len >= ds.len) val_len = 0;
    }
    if (val_len > 0 && ds.len > val_len + (SEQ_LEN + 1)) {
        train_ds.len = ds.len - val_len;
        val_ds.data = ds.data + train_ds.len;
        val_ds.len = val_len;
        printf("[split] train=%zu bytes | val=%zu bytes (%u%%)\n\n", train_ds.len, val_ds.len, VAL_PERCENT);
    } else {
        printf("[split] train seul (validation désactivée, corpus trop court)\n\n");
    }

    /* Setup logs */
    unsigned long long run_id = (unsigned long long)time(NULL);
    char *owned_log_prefix = xstrdup_local(ckpt_path ? ckpt_path : "kmamba_cuda");
    const char *log_prefix = log_prefix_arg ? log_prefix_arg : owned_log_prefix;
    char *step_log_path = make_log_path(log_prefix, "step");
    char *epoch_log_path = make_log_path(log_prefix, "epoch");
    FILE *step_log = open_csv_log(step_log_path, "run_id,epoch,step_in_epoch,global_step,train_loss,train_ppl,grad_norm,grad_over_clip,would_clip,step_ms,tokens_per_sec,bad_loss,lr");
    FILE *epoch_log = open_csv_log(epoch_log_path, "run_id,epoch,steps_in_epoch,total_tokens,train_loss,train_ppl,train_eval_loss,train_eval_ppl,val_loss,val_ppl,epoch_ms,step_ms_avg,tokens_per_sec,param_norm,update_norm,train_bytes,val_bytes");
    if (step_log || epoch_log) {
        printf("[logs] step=%s | epoch=%s\n\n", step_log_path ? step_log_path : "(none)", epoch_log_path ? epoch_log_path : "(none)");
    }

    /* Boucle entraînement - Chinchilla ou Auto */
    size_t seq_tokens = SEQ_LEN + 1;
    uint32_t *batch = (uint32_t*)calloc(BATCH_SIZE * seq_tokens, sizeof(uint32_t));
    
    /* 4.5 chars/token pour estimation Chinchilla */
    const float chars_per_token = 4.5f;
    size_t dataset_tokens = (size_t)(train_ds.len / chars_per_token);
    size_t steps_per_epoch, total_steps;
    
    if (total_tokens_target > 0) {
        /* Mode Chinchilla: forcer le nombre total de tokens à voir */
        total_steps = total_tokens_target / (BATCH_SIZE * SEQ_LEN);
        if (total_steps < 1) total_steps = 1;
        steps_per_epoch = total_steps / N_EPOCHS;
        if (steps_per_epoch < 1) steps_per_epoch = 1;
        printf("[Chinchilla] Target: %zu tokens, steps: %zu, epochs: %d\n", 
               total_tokens_target, total_steps, N_EPOCHS);
    } else {
        /* Mode auto: 1 epoch sur le dataset */
        steps_per_epoch = dataset_tokens / (BATCH_SIZE * SEQ_LEN);
        if (steps_per_epoch < 1) steps_per_epoch = 1;
        total_steps = (size_t)N_EPOCHS * steps_per_epoch;
    }
    
    size_t warmup_steps = total_steps / 40;
    double step_ms_sum = 0.0;
    size_t data_offset = 0; /* Offset pour boucler sur le dataset */

    printf("entraînement : %d epochs × %zu steps (~%zu tokens/step)\n\n", 
           N_EPOCHS, steps_per_epoch, (size_t)(BATCH_SIZE * SEQ_LEN));
    printf(" epoch | train_bt | train_ev |   val    |  tok/s   |  ms/epoch |    lr\n");
    printf("-------+----------+----------+----------+----------+-----------+-----------\n");

    for (int epoch = 1; epoch <= N_EPOCHS; epoch++) {
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        float loss_sum = 0.0f;
        int bad_loss = 0;

        for (size_t s = 0; s < steps_per_epoch; s++) {
            struct timespec step_t0;
            clock_gettime(CLOCK_MONOTONIC, &step_t0);
            size_t global_step = (size_t)(epoch - 1) * steps_per_epoch + s;
            float current_lr = lr_schedule(LR, global_step, warmup_steps, total_steps);
            kmamba_update_lr(m, current_lr, current_lr);
            
            sample_batch_offset(&train_ds, batch, BATCH_SIZE, SEQ_LEN, &data_offset);
            float loss = kmamba_train_batch(m, batch, BATCH_SIZE);
            loss_sum += loss;
            if (!isfinite(loss)) bad_loss = 1;
            
            double step_ms = elapsed_ms(&step_t0);
            step_ms_sum += step_ms;
            double step_tokens_s = step_ms > 0.0 ? ((double)BATCH_SIZE * (double)SEQ_LEN * 1000.0) / step_ms : 0.0;

            if (step_log) {
                fprintf(step_log, "%llu,%d,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%d,%.3f,%.3f,%d,%.6f\n", 
                        run_id, epoch, s, global_step, loss, safe_perplexity(loss), 
                        kmamba_last_grad_norm(m), kmamba_last_grad_over_clip(m), 
                        kmamba_last_grad_would_clip(m), step_ms, step_tokens_s, bad_loss, current_lr);
            }
        }

        double ms = elapsed_ms(&t0);
        double step_ms_avg = steps_per_epoch > 0 ? step_ms_sum / (double)steps_per_epoch : 0.0;
        double tokens_total = (double)steps_per_epoch * (double)BATCH_SIZE * (double)SEQ_LEN;
        double tokens_s = ms > 0.0 ? (tokens_total * 1000.0) / ms : 0.0;
        float avg_loss = loss_sum / (float)steps_per_epoch;
        float train_eval_loss = evaluate_dataset(m, &train_ds, 32);
        float val_loss = (val_ds.len > 0) ? evaluate_dataset(m, &val_ds, 32) : NAN;
        float current_lr = lr_schedule(LR, (size_t)epoch * steps_per_epoch, warmup_steps, total_steps);

        printf("  %4d | %8.4f | %8.4f | %8.4f | %8.0f | %8.1f | %.2e\n",
               epoch, avg_loss, train_eval_loss, val_loss, tokens_s, ms, current_lr);
        fflush(stdout);

        if (epoch_log) {
            fprintf(epoch_log, "%llu,%d,%zu,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%.6f,%.6f,%zu,%zu\n",
                    run_id, epoch, steps_per_epoch, tokens_total, avg_loss, safe_perplexity(avg_loss),
                    train_eval_loss, safe_perplexity(train_eval_loss), val_loss, safe_perplexity(val_loss),
                    ms, step_ms_avg, tokens_s, 0.0f, 0.0f, train_ds.len, val_ds.len);
        }

        if (ckpt_path && epoch % SAVE_EVERY == 0) {
            kmamba_save(m, ckpt_path);
            printf("         [checkpoint sauvegardé : %s]\n", ckpt_path);
        }
    }

    /* Checkpoint final */
    const char *final_ckpt = ckpt_path ? ckpt_path : "kmamba_cpu.bin";
    kmamba_save(m, final_ckpt);
    printf("\n[checkpoint final : %s]\n", final_ckpt);

    /* Génération démonstrative */
    generate(m, "Les systemes", GEN_LEN);

    if (step_log) fclose(step_log);
    if (epoch_log) fclose(epoch_log);
    kmamba_free(m);
    free(ds.data);
    free(batch);
    free(owned_log_prefix);
    free(step_log_path);
    free(epoch_log_path);
    return 0;
}
