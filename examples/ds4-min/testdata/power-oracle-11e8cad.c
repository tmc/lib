/* Extracted verbatim from ds4.c at 11e8cad4473716a41baf4d62e92ed9764a00fce6.
 * Complete source SHA256 62e458f2c416148c0882f7247c7b16f9820054b0c2ae41374a9209541fe7a4c7.
 * These arithmetic helpers require the caller's explicit fixture graph/sleeper.
 */
static bool graph_power_throttle_enabled(const ds4_gpu_graph *g) {
    return g && g->power_percent > 0 && g->power_percent < 100;
}

static double graph_power_update_avg(double avg, double sample) {
    if (sample <= 0.0 || !isfinite(sample)) return avg;
    if (avg <= 0.0 || !isfinite(avg)) return sample;
    return avg * 0.875 + sample * 0.125;
}

static void graph_power_sleep(double work_sec, uint32_t power_percent) {
    if (power_percent == 0 || power_percent >= 100) return;
    /* Target duty cycle: work / (work + sleep) = power / 100.
     * At --power 50 this sleeps for one measured work interval; at 25 it
     * sleeps for three. */
    const double sleep = work_sec * (100.0 - (double)power_percent) /
                         (double)power_percent;
    sleep_sec(sleep);
}

static void graph_power_note_prefill_layer(ds4_gpu_graph *g,
                                           uint32_t il,
                                           double elapsed_sec) {
    if (!graph_power_throttle_enabled(g)) return;
    if (il >= DS4_N_LAYER) return;
    g->prefill_layer_avg_sec[il] =
        graph_power_update_avg(g->prefill_layer_avg_sec[il], elapsed_sec);
    graph_power_sleep(g->prefill_layer_avg_sec[il], g->power_percent);
}

static void graph_power_note_decode_token(ds4_gpu_graph *g, double elapsed_sec) {
    if (!graph_power_throttle_enabled(g)) return;
    g->decode_token_avg_sec =
        graph_power_update_avg(g->decode_token_avg_sec, elapsed_sec);
    graph_power_sleep(g->decode_token_avg_sec, g->power_percent);
}
