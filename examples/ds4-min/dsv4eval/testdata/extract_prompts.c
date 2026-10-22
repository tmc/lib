/* Build on Linux with the pinned oracle include directory, cases source,
 * -ffunction-sections -fdata-sections -Wl,--gc-sections -lm -pthread -ldl.
 * Calls the oracle's actual question builder; stdout is TSV of source/id,
 * generation budget and the exact rendered-question bytes as hex. */
#define main upstream_eval_main
#include "ds4_eval.c"
#undef main
int main(void) {
    unsigned n=0;
    for (size_t i=0;i<eval_hard_case_count;i++) {
        const eval_case *c=&eval_hard_cases[i];
        if (!(c->suites&EVAL_SUITE_HARD_SMOKE)) continue;
        char *question=build_question_prompt(c);
        if (!question) return 1;
        printf("%s/%s\t%u\t",c->source,c->id,c->max_tokens);
        for (const unsigned char *p=(const unsigned char *)question;*p;p++) printf("%02x",*p);
        putchar('\n'); free(question); n++;
    }
    return n==12 ? 0 : 1;
}
