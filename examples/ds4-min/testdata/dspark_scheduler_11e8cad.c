// Actual scheduler bodies copied from antirez/ds4 11e8cad4473716a41baf4d62e92ed9764a00fce6.
// Host fixture controls only; no model, CUDA, scheduling speed or native gate.
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#undef __APPLE__
#define DS4_N_LEADING_DENSE 0
#define DS4_BACKEND_CUDA 1
#define DS4_BACKEND_METAL 2
#define DS4_TENSOR_IQ2_XXS 16
#define DS4_TENSOR_Q2_K 10
#define DS4_TENSOR_MXFP4 39
typedef struct { int type; } ds4_tensor;
typedef struct { ds4_tensor *ffn_gate_exps,*ffn_down_exps; } ds4_layer_weights;
typedef struct {
 int backend; bool ssd_streaming,dspark_exact_sampling;
 struct { ds4_layer_weights layer[1]; } weights;
 struct { bool active; } tp;
} ds4_engine;
typedef struct {
 ds4_engine *engine;
 uint32_t dspark_sched_cycles,dspark_sched_accepted,dspark_sched_no_draft;
 uint32_t dspark_sched_skip,dspark_sched_lifetime_accepted;
 double dspark_sched_extra_ms,dspark_sched_saved_ms;
 double dspark_sched_life_extra_ms,dspark_sched_life_saved_ms;
 double dspark_last_target_eval_ms;
 bool dspark_sched_skipped_cycle,dspark_sched_long_accept_seen,dspark_sched_bypass;
 bool dspark_last_confidence0_valid;
 float dspark_last_confidence0;
 struct { uint32_t scheduler_skips; double saved_ms; } dspark_stats;
} ds4_session;
static bool spark;
static bool ds4_dspark_rocm_gfx1151_fast_path(void) {return false;}
static bool ds4_dspark_rocm_gfx1151_reference_alignment(void) {return false;}
static bool ds4_gpu_device_is_spark(void) {return spark;}
static bool ds4_gpu_device_is_m5_apple_silicon(void) {return false;}
static bool ds4_dspark_stats_enabled(void) {return false;}
static uint32_t ds4_dspark_env_u32(const char *name, uint32_t fallback) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || errno != 0 || v > UINT32_MAX) return fallback;
    return (uint32_t)v;
}

static bool ds4_session_dspark_rocm_gfx1151_fast_path(
        const ds4_session *s) {
    return s && s->engine &&
           s->engine->backend == DS4_BACKEND_CUDA &&
           ds4_dspark_rocm_gfx1151_fast_path();
}

static bool ds4_session_dspark_rocm_gfx1151_reference_alignment(
        const ds4_session *s) {
    return s && s->engine &&
           s->engine->backend == DS4_BACKEND_CUDA &&
           ds4_dspark_rocm_gfx1151_reference_alignment();
}

static bool ds4_session_dspark_seed_batch_enabled(
        const ds4_session *s) {
    const char *env = getenv("DS4_DSPARK_SEED_BATCH");
    if (env && env[0]) return env[0] != '0';
    if (ds4_session_dspark_rocm_gfx1151_fast_path(s)) return true;
    if (!s || !s->engine) return false;
    const ds4_engine *e = s->engine;
    const ds4_layer_weights *layer = &e->weights.layer[DS4_N_LEADING_DENSE];
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    if (e->backend == DS4_BACKEND_CUDA && ds4_gpu_device_is_spark()) {
        return !e->ssd_streaming && !e->dspark_exact_sampling &&
               layer->ffn_gate_exps && layer->ffn_down_exps &&
               layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
               layer->ffn_down_exps->type == DS4_TENSOR_Q2_K;
    }
#endif
    /* Keep streaming and unmeasured weight/device combinations on their
     * existing schedule. No sampling decision is changed by this dispatch. */
    return e->backend == DS4_BACKEND_METAL && !e->ssd_streaming &&
           !e->dspark_exact_sampling &&
           ds4_gpu_device_is_m5_apple_silicon() &&
           layer->ffn_gate_exps && layer->ffn_down_exps &&
           ((layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
             layer->ffn_down_exps->type == DS4_TENSOR_Q2_K) ||
            (e->tp.active && layer->ffn_gate_exps->type == DS4_TENSOR_MXFP4 &&
             layer->ffn_down_exps->type == DS4_TENSOR_MXFP4));
}

static bool ds4_session_dspark_seed_batch_short_fallback(const ds4_session *s) {
    if (s->engine->backend == DS4_BACKEND_METAL) return true;
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    return s->engine->backend == DS4_BACKEND_CUDA &&
           ds4_gpu_device_is_spark();
#else
    return false;
#endif
}

static bool ds4_dspark_scheduler_enabled(const ds4_session *s) {
    const char *env = getenv("DS4_DSPARK_SCHEDULER");
    if (env && env[0]) return strcmp(env, "0") != 0;
    (void)s;
    return true;
}

static uint32_t ds4_dspark_scheduler_window(const ds4_session *s) {
    const uint32_t fallback =
        ds4_session_dspark_rocm_gfx1151_fast_path(s) ? 16u : 4u;
    uint32_t v = ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_WINDOW", fallback);
    return v ? v : 4;
}

static uint32_t ds4_dspark_scheduler_skip_cycles(const ds4_session *s) {
    uint32_t fallback =
        s && ds4_session_dspark_seed_batch_short_fallback(s) &&
        ds4_session_dspark_seed_batch_enabled(s) ? 32u :
        ds4_dspark_rocm_gfx1151_fast_path() ? 4u : 2u;
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    /* Recheck sooner after useful long drafts; low-acceptance prose keeps
     * the longer pause instead of paying for another unproductive proposal. */
    if (s && s->dspark_sched_long_accept_seen &&
        ds4_session_dspark_seed_batch_enabled(s) && ds4_gpu_device_is_spark())
        fallback = 8u;
#endif
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_SKIP", fallback);
}

static uint32_t ds4_dspark_scheduler_slow_skip_cycles(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_SLOW_SKIP", 4);
}

static uint32_t ds4_dspark_scheduler_min_avg_milli(void) {
    const uint32_t fallback =
        ds4_dspark_rocm_gfx1151_fast_path() ? 4000u : 1500u;
    return ds4_dspark_env_u32(
            "DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI", fallback);
}

static uint32_t ds4_dspark_scheduler_max_ms_per_accept_milli(void) {
    return ds4_dspark_env_u32(
            "DS4_DSPARK_SCHEDULER_MAX_MS_PER_ACCEPT_MILLI", 0);
}

static uint32_t ds4_dspark_scheduler_max_extra_saved_ratio_milli(void) {
    return ds4_dspark_env_u32(
            "DS4_DSPARK_SCHEDULER_MAX_EXTRA_SAVED_RATIO_MILLI", 0);
}

static uint32_t ds4_dspark_scheduler_break_even_window(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_BREAK_EVEN_WINDOW", 0);
}

static uint32_t ds4_dspark_scheduler_no_draft_skip_cycles(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP", 3);
}

static uint32_t ds4_dspark_scheduler_short_accept_no_draft_skip_cycles(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP", 4);
}

static uint32_t ds4_dspark_scheduler_cold_low_confidence_skip_cycles(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_SKIP", 7);
}

static uint32_t ds4_dspark_scheduler_tail_min_tokens(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS", 10);
}

static float ds4_dspark_scheduler_cold_low_confidence_threshold(void) {
    return (float)ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_MILLI", 500) / 1000.0f;
}

/* Timing-sensitive scheduling changes which arithmetic path advances a token.
 * Keep it opt-in so greedy DSpark output is reproducible across runs. */
static bool ds4_dspark_scheduler_timing_enabled(void) {
    return ds4_dspark_scheduler_max_ms_per_accept_milli() != 0 ||
           ds4_dspark_scheduler_max_extra_saved_ratio_milli() != 0;
}

static void ds4_session_dspark_scheduler_reset(ds4_session *s) {
    if (!s) return;
    s->dspark_sched_cycles = 0;
    s->dspark_sched_accepted = 0;
    s->dspark_sched_no_draft = 0;
    s->dspark_sched_extra_ms = 0.0;
    s->dspark_sched_saved_ms = 0.0;
}

static void ds4_session_dspark_scheduler_begin_request(ds4_session *s) {
    if (!s) return;
    ds4_session_dspark_scheduler_reset(s);
    s->dspark_sched_skip = 0;
    s->dspark_sched_lifetime_accepted = 0;
    s->dspark_sched_life_extra_ms = 0.0;
    s->dspark_sched_life_saved_ms = 0.0;
    s->dspark_sched_skipped_cycle = false;
    s->dspark_sched_long_accept_seen = false;
    s->dspark_sched_bypass = false;
}

static bool ds4_session_dspark_scheduler_should_skip(ds4_session *s) {
    if (!s || !ds4_dspark_scheduler_enabled(s)) return false;
    s->dspark_sched_skipped_cycle = false;
    if (s->dspark_sched_bypass) {
        s->dspark_sched_skipped_cycle = true;
        s->dspark_stats.scheduler_skips++;
        return true;
    }
    if (s->dspark_sched_skip == 0) return false;
    s->dspark_sched_skip--;
    s->dspark_sched_skipped_cycle = true;
    s->dspark_stats.scheduler_skips++;
    if (getenv("DS4_DSPARK_SPEC_LOG") != NULL) {
        fprintf(stderr,
                "ds4: DSpark scheduler skip remaining=%u\n",
                s->dspark_sched_skip);
    }
    return true;
}

static void ds4_session_dspark_scheduler_note(
        ds4_session *s,
        uint32_t     accepted_drafts,
        bool         no_draft,
        double       extra_ms) {
    if (!s || !ds4_dspark_scheduler_enabled(s)) return;
    if (s->dspark_sched_skipped_cycle) {
        s->dspark_sched_skipped_cycle = false;
        return;
    }

    s->dspark_sched_cycles++;
    s->dspark_sched_accepted += accepted_drafts;
    if (accepted_drafts != 0) {
        if (s->dspark_sched_lifetime_accepted <=
            UINT32_MAX - accepted_drafts) {
            s->dspark_sched_lifetime_accepted += accepted_drafts;
        } else {
            s->dspark_sched_lifetime_accepted = UINT32_MAX;
        }
        if (accepted_drafts > 2u) {
            s->dspark_sched_long_accept_seen = true;
        }
    }
    if (no_draft) s->dspark_sched_no_draft++;
    if (extra_ms > 0.0 && isfinite(extra_ms)) {
        s->dspark_sched_extra_ms += extra_ms;
    }
    if (accepted_drafts != 0 &&
        s->dspark_last_target_eval_ms > 0.0 &&
        isfinite(s->dspark_last_target_eval_ms)) {
        const double saved_ms =
            s->dspark_last_target_eval_ms * (double)accepted_drafts;
        s->dspark_sched_saved_ms += saved_ms;
        if (ds4_dspark_stats_enabled()) {
            s->dspark_stats.saved_ms += saved_ms;
        }
    }

    const uint32_t no_draft_skip =
        ds4_dspark_scheduler_no_draft_skip_cycles();
    if (no_draft && no_draft_skip != 0) {
        uint32_t skip = no_draft_skip;
        if (s->dspark_sched_lifetime_accepted != 0 &&
            !s->dspark_sched_long_accept_seen) {
            const uint32_t short_accept_skip =
                ds4_dspark_scheduler_short_accept_no_draft_skip_cycles();
            if (skip < short_accept_skip) skip = short_accept_skip;
        } else if (s->dspark_sched_lifetime_accepted == 0 &&
                   s->dspark_last_confidence0_valid &&
                   s->dspark_last_confidence0 <=
                       ds4_dspark_scheduler_cold_low_confidence_threshold()) {
            const uint32_t cold_low_conf_skip =
                ds4_dspark_scheduler_cold_low_confidence_skip_cycles();
            if (skip < cold_low_conf_skip) skip = cold_low_conf_skip;
        }
        if (s->dspark_sched_skip < skip) {
            s->dspark_sched_skip = skip;
        }
        if (getenv("DS4_DSPARK_SPEC_LOG") != NULL) {
            fprintf(stderr,
                    "ds4: DSpark scheduler no-draft pause skip=%u "
                    "accepted_total=%u long_accept=%d confidence0=%s%.3f\n",
                    s->dspark_sched_skip,
                    s->dspark_sched_lifetime_accepted,
                    s->dspark_sched_long_accept_seen ? 1 : 0,
                    s->dspark_last_confidence0_valid ? "" : "n/a:",
                    s->dspark_last_confidence0);
        }
    }

    const uint32_t window = ds4_dspark_scheduler_window(s);
    const uint32_t break_even_window =
        ds4_dspark_scheduler_break_even_window();

    const uint32_t max_extra_saved_ratio_milli =
        ds4_dspark_scheduler_max_extra_saved_ratio_milli();
    const bool measured_unprofitable =
        max_extra_saved_ratio_milli != 0 &&
        s->dspark_sched_accepted != 0 &&
        s->dspark_sched_saved_ms > 0.0 &&
        s->dspark_sched_extra_ms * 1000.0 >
            s->dspark_sched_saved_ms *
            (double)max_extra_saved_ratio_milli;

    if (break_even_window != 0 &&
        s->dspark_sched_cycles >= break_even_window &&
        measured_unprofitable) {
        s->dspark_sched_skip = ds4_dspark_scheduler_slow_skip_cycles();
        if (getenv("DS4_DSPARK_SPEC_LOG") != NULL) {
            fprintf(stderr,
                    "ds4: DSpark scheduler break-even pause cycles=%u "
                    "accepted=%u saved=%.3fms extra=%.3fms skip=%u\n",
                    s->dspark_sched_cycles,
                    s->dspark_sched_accepted,
                    s->dspark_sched_saved_ms,
                    s->dspark_sched_extra_ms,
                    s->dspark_sched_skip);
        }
        ds4_session_dspark_scheduler_reset(s);
        return;
    }

    if (s->dspark_sched_cycles < window) return;

    const uint64_t avg_milli =
        ((uint64_t)s->dspark_sched_accepted * 1000ull) /
        (uint64_t)s->dspark_sched_cycles;
    const uint32_t min_avg_milli =
        ds4_dspark_scheduler_min_avg_milli();
    const bool low_accept = avg_milli < min_avg_milli;
    const bool many_no_draft =
        s->dspark_sched_no_draft * 2u >= s->dspark_sched_cycles;
    const uint32_t max_ms_per_accept_milli =
        ds4_dspark_scheduler_max_ms_per_accept_milli();
    const double extra_per_accept_ms =
        s->dspark_sched_accepted != 0 ?
        s->dspark_sched_extra_ms / (double)s->dspark_sched_accepted : 0.0;
    const bool slow_accept =
        max_ms_per_accept_milli != 0 &&
        s->dspark_sched_accepted != 0 &&
        extra_per_accept_ms * 1000.0 > (double)max_ms_per_accept_milli;
    if (low_accept || many_no_draft || slow_accept || measured_unprofitable) {
        if (ds4_session_dspark_rocm_gfx1151_fast_path(s)) {
            s->dspark_sched_bypass = true;
            s->dspark_sched_skip = 0;
            if (getenv("DS4_DSPARK_SPEC_LOG") != NULL) {
                fprintf(stderr,
                        "ds4: DSpark scheduler bypass accepted=%u avg=%.3f "
                        "no_draft=%u\n",
                        s->dspark_sched_accepted,
                        (double)avg_milli / 1000.0,
                        s->dspark_sched_no_draft);
            }
            ds4_session_dspark_scheduler_reset(s);
            return;
        }
        s->dspark_sched_skip = ds4_dspark_scheduler_skip_cycles(s);
        if (many_no_draft || slow_accept || measured_unprofitable) {
            const uint32_t slow_skip = ds4_dspark_scheduler_slow_skip_cycles();
            if (s->dspark_sched_skip < slow_skip) {
                s->dspark_sched_skip = slow_skip;
            }
        }
        if (getenv("DS4_DSPARK_SPEC_LOG") != NULL) {
            fprintf(stderr,
                    "ds4: DSpark scheduler pause cycles=%u accepted=%u "
                    "avg=%.3f no_draft=%u extra_per_accept=%.3fms "
                    "saved=%.3fms extra=%.3fms skip=%u\n",
                    s->dspark_sched_cycles,
                    s->dspark_sched_accepted,
                    (double)avg_milli / 1000.0,
                    s->dspark_sched_no_draft,
                    extra_per_accept_ms,
                    s->dspark_sched_saved_ms,
                    s->dspark_sched_extra_ms,
                    s->dspark_sched_skip);
        }
    }
    ds4_session_dspark_scheduler_reset(s);
}
int main(int argc,char **argv) {
 spark=argc>1 && atoi(argv[1]);
 ds4_tensor gate={16},down={10};
 ds4_engine engine={.backend=1};
 engine.weights.layer[0].ffn_gate_exps=&gate;
 engine.weights.layer[0].ffn_down_exps=&down;
 ds4_session s={.engine=&engine};
 ds4_session_dspark_scheduler_begin_request(&s);
 unsigned int accepted[]={0,0,0,1,0,3,0,0,1,1,1,1,3,0,0,0,3,3,3,3};
 for(unsigned int i=0;i<70;i++) {
   bool skipped=ds4_session_dspark_scheduler_should_skip(&s);
   s.dspark_last_confidence0_valid=true;s.dspark_last_confidence0=0.5f;
   s.dspark_last_target_eval_ms=2.0;
   unsigned int n=accepted[i%20];
   ds4_session_dspark_scheduler_note(&s,n,n==0,4.0);
   printf("%u %u %u %u %u %u %u\n",skipped,s.dspark_sched_cycles,s.dspark_sched_accepted,s.dspark_sched_no_draft,s.dspark_sched_skip,s.dspark_sched_lifetime_accepted,s.dspark_sched_long_accept_seen);
 }
}
