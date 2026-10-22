/* Compile with the pinned upstream ds4_eval_cases.c and include directory.
 * stdout is a JSON corpus; no prompt or answer is reconstructed by hand. */
#include <stdio.h>
#include "ds4_eval_cases.h"

static void string(const char *s) {
    putchar('"');
    if (s) for (const unsigned char *p=(const unsigned char *)s; *p; p++) {
        if (*p=='"' || *p=='\\') { putchar('\\'); putchar(*p); }
        else if (*p<32) printf("\\u%04x",*p);
        else putchar(*p);
    }
    putchar('"');
}
static void field(const char *key,const char *value) { string(key); putchar(':'); string(value); }
int main(void) {
    unsigned n=0;
    puts("[");
    for (size_t i=0;i<eval_hard_case_count;i++) {
        const eval_case *c=&eval_hard_cases[i];
        if (!(c->suites&EVAL_SUITE_HARD_SMOKE)) continue;
        if (n++) puts(",");
        putchar('{'); field("Source",c->source); putchar(','); field("ID",c->id);
        putchar(','); field("Domain",c->domain); putchar(','); field("Title",c->title);
        putchar(','); field("Question",c->question); putchar(','); field("Answer",c->answer);
        printf(",\"Kind\":%u,\"Suites\":%u,\"MaxTokens\":%u,\"Choices\":[",c->answer_kind,c->suites,c->max_tokens);
        unsigned count=0;
        for (unsigned j=0;j<EVAL_MAX_CHOICES;j++) if (c->choice[j]) { if(count++) putchar(','); string(c->choice[j]); }
        printf("],\"Aliases\":["); count=0;
        for (unsigned j=0;j<EVAL_MAX_ALIASES;j++) if(c->alias[j]) { if(count++) putchar(','); string(c->alias[j]); }
        printf("]}");
    }
    puts("\n]");
    return n==12 ? 0 : 1;
}
