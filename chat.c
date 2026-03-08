/* chat.c — Interactive REPL for BissiMamba.
 *
 * Usage: ./mamba_chat [model_path]
 *   model_path : path to lm_checkpoint.bin (default: lm_checkpoint.bin)
 *
 * Loads the model once, then loops:
 *   You> <prompt>
 *   Bot> <generation>
 * Type "quit" or "exit", or send EOF (Ctrl+D) to exit.
 */

#include "mamba.h"
#include "lm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_MODEL    "lm_checkpoint.bin"
#define MAX_PROMPT_LEN   1024
#define GENERATION_TEMP  0.8f

int main(int argc, char *argv[]) {
    const char *model_path = (argc > 1) ? argv[1] : DEFAULT_MODEL;

    srand((unsigned)time(NULL));

    /* ---- 1. Create LM ---- */
    LMConfig cfg = lm_default_config();
    LM *lm = lm_create(&cfg);
    if (!lm) {
        fprintf(stderr, "ERROR: failed to create LM\n");
        return 1;
    }

    /* ---- 2. Load checkpoint ---- */
    if (lm_load(lm, model_path) != 0) {
        fprintf(stderr, "ERROR: cannot load model from '%s'\n", model_path);
        fprintf(stderr, "       Train first: make mamba_lm_train && ./mamba_lm_train\n");
        lm_free(lm);
        return 1;
    }

    fprintf(stderr, "BissiMamba REPL — model: %s\n", model_path);
    fprintf(stderr, "Config: vocab=%zu dim=%zu state=%zu seq=%zu (~%.3fM params)\n",
            cfg.vocab_size, cfg.dim, cfg.state_size, cfg.seq_len,
            (double)lm_num_parameters(&cfg) / 1e6);
    fprintf(stderr, "Type your message and press Enter. Ctrl+D or 'quit' to exit.\n\n");

    /* ---- 3. REPL loop ---- */
    char prompt[MAX_PROMPT_LEN];
    char context_buf[MAX_PROMPT_LEN + 64];

    for (;;) {
        /* Print user prompt */
        fprintf(stdout, "You> ");
        fflush(stdout);

        /* Read one line */
        if (!fgets(prompt, sizeof(prompt), stdin)) {
            /* EOF (Ctrl+D) */
            fprintf(stdout, "\n");
            break;
        }

        /* Strip trailing newline/CR */
        size_t plen = strlen(prompt);
        while (plen > 0 && (prompt[plen-1] == '\n' || prompt[plen-1] == '\r'))
            prompt[--plen] = '\0';

        /* Skip empty lines */
        if (plen == 0) continue;

        /* Exit commands */
        if (strcmp(prompt, "quit") == 0 || strcmp(prompt, "exit") == 0)
            break;

        /* Build context and generate */
        snprintf(context_buf, sizeof(context_buf), "Human: %s\nBot: ", prompt);

        fprintf(stdout, "Bot> ");
        fflush(stdout);
        lm_generate(lm, context_buf, cfg.max_gen_len, GENERATION_TEMP);
        fprintf(stdout, "\n");
        fflush(stdout);
    }

    /* ---- 4. Cleanup ---- */
    lm_free(lm);
    fprintf(stderr, "Bye.\n");
    return 0;
}
