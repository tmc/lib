/* Independent oracle driver. Link against the pinned upstream engine, not
 * ds4-min or its scalar helper bodies. The complete seeded GGUF materializer
 * and bounded compiler/link/native admission remain unqualified.
 *
 * Before each eval the session logits are poisoned with NaN through the
 * upstream setter, because ds4_session_copy_logits copies s->logits whether
 * or not eval refreshed them. A position whose logits equal the previous
 * position's is also refused. */
#include "ds4.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: seed-oracle model.gguf binary-sha256 physical-bytes output.json\n");
        return 2;
    }
    if (strlen(argv[2]) != 64 || strspn(argv[2], "0123456789abcdef") != 64) return 2;
    char *end = NULL;
    errno = 0;
    uint64_t physical = strtoull(argv[3], &end, 10);
    if (errno || !end || *end || physical == 0 || physical > (UINT64_C(16) << 30)) return 2;
    FILE *output = fopen(argv[4], "wx");
    if (!output) return 2;
    int result = 1;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    float *logits = NULL, *previous = NULL, *poison = NULL;
    static const int tokens[5] = {0, 1, 2, 3, 4};
    ds4_engine_options options = {0};
    options.model_path = argv[1];
    options.backend = DS4_BACKEND_CUDA;
    options.n_threads = 1;
    options.context_size = 5;
    options.prefill_chunk = 1;
    if (ds4_engine_create_with_gpu_config(&engine, &options, NULL) != 0 || !engine) goto done;
    const int vocab = ds4_engine_vocab_size(engine);
    if (vocab != 129280 || ds4_session_create(&session, engine, 5) != 0 || !session) goto done;
    logits = malloc((size_t)vocab * sizeof(*logits));
    previous = malloc((size_t)vocab * sizeof(*previous));
    poison = malloc((size_t)vocab * sizeof(*poison));
    if (!logits || !previous || !poison) goto done;
    for (int i = 0; i < vocab; i++) poison[i] = NAN;
    fprintf(output, "{\"SeedVersion\":\"pro-role-alias-v3-header1894\",\"UpstreamCommit\":\"11e8cad4473716a41baf4d62e92ed9764a00fce6\",\"OracleBinarySHA256\":\"%s\",\"PhysicalBytes\":%" PRIu64 ",\"Tokens\":[0,1,2,3,4],\"Logits\":[", argv[2], physical);
    for (int position = 0; position < 5; position++) {
        char error[512] = {0};
        if (ds4_session_set_logits(session, poison, vocab) != 0) goto done;
        if (ds4_session_eval(session, tokens[position], error, sizeof(error)) != 0) {
            fprintf(stderr, "upstream position%d token%d: %s\n", position, tokens[position], error);
            goto done;
        }
        if (ds4_session_copy_logits(session, logits, vocab) != vocab) goto done;
        int nonzero = 0;
        for (int i = 0; i < vocab; i++) {
            if (!isfinite(logits[i])) goto done;
            nonzero |= logits[i] != 0.0f;
        }
        if (!nonzero) goto done;
        if (position && !memcmp(logits, previous, (size_t)vocab * sizeof(*logits))) {
            fprintf(stderr, "upstream position%d logits repeat position%d\n", position, position - 1);
            goto done;
        }
        memcpy(previous, logits, (size_t)vocab * sizeof(*logits));
        fprintf(output, "%s[", position ? "," : "");
        for (int i = 0; i < vocab; i++) fprintf(output, "%s%.9g", i ? "," : "", (double)logits[i]);
        fprintf(output, "]");
        fprintf(stderr, "PINNED_C_SEEDED_STEP position=%d token=%d logits=%d\n", position, tokens[position], vocab);
    }
    fprintf(output, "]}\n");
    if (fflush(output) != 0 || ferror(output)) goto done;
    result = 0;
done:
    free(logits);
    free(previous);
    free(poison);
    if (session) ds4_session_free(session);
    if (engine) ds4_engine_close(engine);
    if (fclose(output) != 0) result = 1;
    /* Any partial output is retained and is never treated as qualified. */
    return result;
}
