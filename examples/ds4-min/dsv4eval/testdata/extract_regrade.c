/* Calls the hash-qualified upstream parser. No question data is copied here. */
#define main upstream_eval_main
#include "ds4_eval.c"
#undef main
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    if (!strcmp(argv[1], "--write")) {
        eval_config cfg = {0};
        cfg.model_path = "host-format-fixture";
        cfg.ctx_size = 32768;
        cfg.suite_mask = EVAL_SUITE_HARD_SMOKE;
        cfg.think_mode = DS4_THINK_HIGH;
        int count = 0;
        for (size_t i = 0; i < eval_hard_case_count; i++)
            if (eval_hard_cases[i].suites & EVAL_SUITE_HARD_SMOKE) count++;
        trace_write_header(stdout, &cfg, "host-format-fixture", count, 0, 0);
        int index = 0;
        for (size_t i = 0; i < eval_hard_case_count; i++) {
            const eval_case *tc = &eval_hard_cases[i];
            if (!(tc->suites & EVAL_SUITE_HARD_SMOKE)) continue;
            trace_write_case(stdout, &cfg, tc, index++, count,
                "NOT_GRADED", NULL, "", tc->question, "",
                DS4_THINK_HIGH, 0, 0, tc->max_tokens, 0, NULL, NULL);
        }
        fprintf(stdout, "===== SUMMARY =====\ntotal: %d\n", count);
        return 0;
    }
    return regrade_trace_file(argv[1]);
}
