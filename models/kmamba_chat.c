/* chat.c — Mode inférence/chat avec k-mamba
 *
 * Usage: ./chat <checkpoint.bin> [max_tokens] [temperature]
 * 
 * Exemples:
 *   ./chat kmamba_500k.bin 256 0.8
 *   echo "Bonjour, comment vas-tu?" | ./chat kmamba_500k.bin 100 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "kmamba.h"

/* Configuration par défaut (doit matcher le checkpoint) */
static KMambaConfig make_config(void) {
    KMambaConfig cfg = {
        .vocab_size   = 32768,
        .dim          = 48,
        .state_size   = 96,
        .seq_len      = 128,
        .n_layers     = 2,
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

/* Sampling aléatoire avec temperature */
static uint32_t sample_token(const float *logits, size_t vocab_size, float temperature) {
    float probs[32768];  /* MAX_VOCAB_SIZE */
    
    /* Softmax avec temperature */
    float max_val = logits[0];
    for (size_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_val) / temperature);
        sum += probs[i];
    }
    
    /* CDF sampling */
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        cumsum += probs[i] / sum;
        if (r <= cumsum) return (uint32_t)i;
    }
    return 0;
}

/* Génération de texte */
static void generate(KMamba *m, const char *prompt, size_t max_tokens, float temperature) {
    size_t V = m->cfg.vocab_size;
    size_t L = m->cfg.seq_len;
    
    /* Encode le prompt */
    size_t prompt_len = 0;
    uint32_t *prompt_tokens = kmamba_encode(prompt, &prompt_len);
    
    if (!prompt_tokens || prompt_len == 0) {
        fprintf(stderr, "Erreur: prompt vide ou tokenization échouée\n");
        return;
    }
    
    printf("Prompt (%zu tokens): ", prompt_len);
    char *decoded = kmamba_decode(prompt_tokens, prompt_len);
    if (decoded) {
        printf("%s\n\n", decoded);
        kmamba_free_string(decoded);
    }
    
    /* Buffer pour contexte (sliding window) */
    uint32_t context[512];  /* Max seq_len */
    size_t ctx_len = prompt_len < L ? prompt_len : L;
    memcpy(context, prompt_tokens, ctx_len * sizeof(uint32_t));
    
    kmamba_free_tokens(prompt_tokens, prompt_len);
    
    printf("=== Réponse ===\n");
    fflush(stdout);
    
    /* Génération token par token */
    float *logits = calloc(V, sizeof(float));
    
    for (size_t i = 0; i < max_tokens; i++) {
        /* Forward sur les derniers L tokens */
        uint32_t input[128];
        
        /* Padding à gauche si nécessaire */
        if (ctx_len < L) {
            memset(input, 0, (L - ctx_len) * sizeof(uint32_t));
            memcpy(input + (L - ctx_len), context, ctx_len * sizeof(uint32_t));
        } else {
            memcpy(input, context + (ctx_len - L), L * sizeof(uint32_t));
        }
        
        /* Forward - on prend seulement le dernier logit */
        float *all_logits = calloc(L * V, sizeof(float));
        kmamba_forward(m, input, all_logits);
        
        /* Dernier token = prédiction du prochain */
        memcpy(logits, &all_logits[(L - 1) * V], V * sizeof(float));
        free(all_logits);
        
        /* Sample */
        uint32_t next_token = sample_token(logits, V, temperature);
        
        /* Decode et affiche */
        char *tok_str = kmamba_decode(&next_token, 1);
        if (tok_str) {
            printf("%s", tok_str);
            fflush(stdout);
            kmamba_free_string(tok_str);
        }
        
        /* Ajoute au contexte */
        if (ctx_len < 512) {
            context[ctx_len++] = next_token;
        } else {
            /* Slide window */
            memmove(context, context + 1, (512 - 1) * sizeof(uint32_t));
            context[511] = next_token;
            ctx_len = 512;
        }
        
        /* Stop condition: token 0 (PAD) répété ou fin de phrase */
        if (next_token == 0 && i > 10) break;
    }
    
    printf("\n\n=== Fin (tokens générés: %zu) ===\n", max_tokens);
    free(logits);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <checkpoint.bin> [max_tokens] [temperature]\n", argv[0]);
        fprintf(stderr, "\nExemples:\n");
        fprintf(stderr, "  %s kmamba_500k.bin 256 0.8\n", argv[0]);
        fprintf(stderr, "  echo \"Bonjour\" | %s kmamba_500k.bin 100 1.0\n", argv[0]);
        return 1;
    }
    
    const char *checkpoint_path = argv[1];
    size_t max_tokens = (argc > 2) ? atoi(argv[2]) : 256;
    float temperature = (argc > 3) ? atof(argv[3]) : 0.8f;
    
    /* Clamp temperature */
    if (temperature < 0.1f) temperature = 0.1f;
    if (temperature > 2.0f) temperature = 2.0f;
    
    srand((unsigned)time(NULL));
    
    /* Crée modèle avec config */
    KMambaConfig cfg = make_config();
    printf("=== k-mamba Chat ===\n");
    printf("Config: vocab=%zu, dim=%zu, layers=%zu\n", cfg.vocab_size, cfg.dim, cfg.n_layers);
    printf("Température: %.2f, Max tokens: %zu\n\n", temperature, max_tokens);
    
    KMamba *m = kmamba_create(&cfg);
    if (!m) {
        fprintf(stderr, "Erreur création modèle\n");
        return 1;
    }
    
    /* Charge checkpoint */
    KMamba *loaded = kmamba_load(checkpoint_path, 0, NULL, 0.0f, 0.0f);
    if (!loaded) {
        fprintf(stderr, "Note: Pas de checkpoint trouvé, utilisation poids aléatoires\n");
        printf("(Le modèle n'est pas entraîné - output sera du bruit)\n\n");
        kmamba_init(m, 42);
    } else {
        kmamba_free(m);
        m = loaded;
        printf("✓ Checkpoint chargé: %s\n\n", checkpoint_path);
    }
    
    /* Lire prompt */
    char prompt[4096];
    printf("Entrez votre message (Ctrl+D pour fini):\n");
    
    if (fgets(prompt, sizeof(prompt), stdin) == NULL) {
        strcpy(prompt, "Bonjour, comment ça va?");
    }
    
    /* Enlève newline */
    size_t len = strlen(prompt);
    if (len > 0 && prompt[len-1] == '\n') prompt[len-1] = '\0';
    
    /* Génère réponse */
    generate(m, prompt, max_tokens, temperature);
    
    kmamba_free(m);
    return 0;
}
