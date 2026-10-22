// Pinned antirez/ds4 11e8cad4473716a41baf4d62e92ed9764a00fce6 probability bodies.
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define DS4_NEG_INF (-1.0e30f)
#define DS4_MAYBE_UNUSED
static void *xmalloc(size_t n) { void *p=malloc(n); if(!p) exit(2); return p; }
static inline void argmax_f32_unrolled8_range(
        const float *logits,
        uint32_t     begin,
        uint32_t     end,
        int         *best,
        float       *best_v) {
    uint32_t i = begin;
    int b0 = *best, b1 = *best, b2 = *best, b3 = *best;
    int b4 = *best, b5 = *best, b6 = *best, b7 = *best;
    float v0 = *best_v, v1 = *best_v, v2 = *best_v, v3 = *best_v;
    float v4 = *best_v, v5 = *best_v, v6 = *best_v, v7 = *best_v;

    while (end - i >= 8u) {
        const float x0 = logits[i + 0u];
        const float x1 = logits[i + 1u];
        const float x2 = logits[i + 2u];
        const float x3 = logits[i + 3u];
        const float x4 = logits[i + 4u];
        const float x5 = logits[i + 5u];
        const float x6 = logits[i + 6u];
        const float x7 = logits[i + 7u];
        if (x0 > v0) { v0 = x0; b0 = (int)(i + 0u); }
        if (x1 > v1) { v1 = x1; b1 = (int)(i + 1u); }
        if (x2 > v2) { v2 = x2; b2 = (int)(i + 2u); }
        if (x3 > v3) { v3 = x3; b3 = (int)(i + 3u); }
        if (x4 > v4) { v4 = x4; b4 = (int)(i + 4u); }
        if (x5 > v5) { v5 = x5; b5 = (int)(i + 5u); }
        if (x6 > v6) { v6 = x6; b6 = (int)(i + 6u); }
        if (x7 > v7) { v7 = x7; b7 = (int)(i + 7u); }
        i += 8u;
    }

#define DS4_ARGMAX_MERGE_LANE(b, v) \
    do { \
        if ((v) > *best_v || ((v) == *best_v && (b) < *best)) { \
            *best_v = (v); \
            *best = (b); \
        } \
    } while (0)
    DS4_ARGMAX_MERGE_LANE(b0, v0);
    DS4_ARGMAX_MERGE_LANE(b1, v1);
    DS4_ARGMAX_MERGE_LANE(b2, v2);
    DS4_ARGMAX_MERGE_LANE(b3, v3);
    DS4_ARGMAX_MERGE_LANE(b4, v4);
    DS4_ARGMAX_MERGE_LANE(b5, v5);
    DS4_ARGMAX_MERGE_LANE(b6, v6);
    DS4_ARGMAX_MERGE_LANE(b7, v7);
#undef DS4_ARGMAX_MERGE_LANE

    for (; i < end; i++) {
        const float v = logits[i];
        if (v > *best_v) {
            *best_v = v;
            *best = (int)i;
        }
    }
}

static int sample_argmax_unrolled8(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = DS4_NEG_INF;
    argmax_f32_unrolled8_range(logits, 0, n_vocab, &best, &best_v);
    return best;
}

static int argmax_f32_excluding_unrolled8(
        const float *logits,
        uint32_t     n,
        int          excluded_id) {
    const uint32_t first = excluded_id == 0 ? 1u : 0u;
    if (first >= n) return -1;

    int best = (int)first;
    float best_v = logits[first];
    const uint32_t begin = first + 1u;
    if (excluded_id >= 0 &&
        (uint32_t)excluded_id >= begin &&
        (uint32_t)excluded_id < n) {
        const uint32_t excluded = (uint32_t)excluded_id;
        argmax_f32_unrolled8_range(logits, begin, excluded, &best, &best_v);
        argmax_f32_unrolled8_range(logits, excluded + 1u, n, &best, &best_v);
    } else {
        argmax_f32_unrolled8_range(logits, begin, n, &best, &best_v);
    }
    return best;
}

static int sample_argmax(const float *logits, uint32_t n_vocab) {
    if (getenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == NULL) {
        return sample_argmax_unrolled8(logits, n_vocab);
    }
    int best = 0;
    float best_v = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

static DS4_MAYBE_UNUSED void logits_top2(const float *logits, uint32_t n_vocab,
                        int *top0, float *logit0,
                        int *top1, float *logit1) {
    int b0 = -1, b1 = -1;
    float v0 = DS4_NEG_INF, v1 = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > v0) {
            b1 = b0; v1 = v0;
            b0 = (int)i; v0 = v;
        } else if (v > v1) {
            b1 = (int)i; v1 = v;
        }
    }
    if (top0) *top0 = b0;
    if (logit0) *logit0 = v0;
    if (top1) *top1 = b1;
    if (logit1) *logit1 = v1;
}

static uint64_t sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}

static float sample_rng_f32(uint64_t *state) {
    const uint64_t x = sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

typedef struct {
    int id;
    float logit;
    float prob;
} sample_candidate;

static int sample_candidate_cmp_desc(const void *a, const void *b) {
    const sample_candidate *ca = a;
    const sample_candidate *cb = b;
    const int logit_order =
        (cb->logit > ca->logit) - (cb->logit < ca->logit);
    if (logit_order != 0) return logit_order;
    return (ca->id > cb->id) - (ca->id < cb->id);
}

static bool sample_candidate_gt(sample_candidate a, sample_candidate b) {
    if (a.logit != b.logit) return a.logit > b.logit;
    return a.id < b.id;
}

static void sample_heap_sift_up(sample_candidate *heap, uint32_t idx) {
    while (idx > 0) {
        const uint32_t parent = (idx - 1u) / 2u;
        if (!sample_candidate_gt(heap[parent], heap[idx])) break;
        sample_candidate tmp = heap[parent];
        heap[parent] = heap[idx];
        heap[idx] = tmp;
        idx = parent;
    }
}

static void sample_heap_sift_down(sample_candidate *heap, uint32_t n, uint32_t idx) {
    for (;;) {
        const uint32_t left = idx * 2u + 1u;
        const uint32_t right = left + 1u;
        uint32_t smallest = idx;
        if (left < n && sample_candidate_gt(heap[smallest], heap[left])) {
            smallest = left;
        }
        if (right < n && sample_candidate_gt(heap[smallest], heap[right])) {
            smallest = right;
        }
        if (smallest == idx) break;
        sample_candidate tmp = heap[idx];
        heap[idx] = heap[smallest];
        heap[smallest] = tmp;
        idx = smallest;
    }
}

/* Materialize the exact distribution represented by the public sampling
 * knobs. Speculative decoding needs probabilities, rather than only a sampled
 * token, for the p/q acceptance test and the corrected rejection sample. */
static bool sample_build_probabilities(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        float       *probs) {
    if (!logits || !probs || n_vocab == 0) return false;
    memset(probs, 0, (size_t)n_vocab * sizeof(probs[0]));

    if (temperature <= 0.0f) {
        probs[sample_argmax(logits, n_vocab)] = 1.0f;
        return true;
    }
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k > 1024) top_k = 1024;
    if (top_k < 0) top_k = 0;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    enum { SAMPLE_PROB_FAST_CAP = 512 };
    sample_candidate heap[1024];
    const uint32_t heap_cap = top_k > 0 ? (uint32_t)top_k :
        (uint32_t)SAMPLE_PROB_FAST_CAP;
    uint32_t finite = 0;
    uint32_t heap_n = 0;
    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        if (!isfinite(logits[i])) continue;
        finite++;
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    if (finite == 0) {
        probs[sample_argmax(logits, n_vocab)] = 1.0f;
        return true;
    }

    float full_sum = 0.0f;
    float heap_sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float raw = expf((v - max_logit) / temperature);
        full_sum += raw;
        sample_candidate cand = {.id = (int)i, .logit = v, .prob = raw};
        if (heap_n < heap_cap) {
            heap[heap_n] = cand;
            heap_sum += raw;
            sample_heap_sift_up(heap, heap_n);
            heap_n++;
        } else if (sample_candidate_gt(cand, heap[0])) {
            heap_sum -= heap[0].prob;
            heap[0] = cand;
            heap_sum += raw;
            sample_heap_sift_down(heap, heap_n, 0);
        }
    }
    if (full_sum <= 0.0f || !isfinite(full_sum) || heap_n == 0) {
        probs[sample_argmax(logits, n_vocab)] = 1.0f;
        return true;
    }

    qsort(heap, heap_n, sizeof(heap[0]), sample_candidate_cmp_desc);
    float base_sum = top_k > 0 ? heap_sum : full_sum;
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    bool stopped_by_min_p = false;
    bool reached_top_p = false;
    const float min_raw = heap[0].prob * min_p;
    for (uint32_t i = 0; i < heap_n; i++) {
        if (i > 0 && heap[i].prob < min_raw) {
            stopped_by_min_p = true;
            break;
        }
        filtered_sum += heap[i].prob;
        filtered++;
        if (filtered_sum / base_sum >= top_p) {
            reached_top_p = true;
            break;
        }
    }

    const bool heap_has_all = heap_n == finite;
    const bool min_p_proves_tail_filtered =
        stopped_by_min_p && heap[heap_n - 1u].prob < min_raw;
    const bool fast_complete = top_k > 0 || heap_has_all || reached_top_p ||
        min_p_proves_tail_filtered;

    sample_candidate *cand = heap;
    uint32_t cand_n = heap_n;
    bool owned = false;
    if (!fast_complete) {
        cand = xmalloc((size_t)finite * sizeof(cand[0]));
        cand_n = 0;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            cand[cand_n++] = (sample_candidate){
                .id = (int)i,
                .logit = v,
                .prob = expf((v - max_logit) / temperature),
            };
        }
        qsort(cand, cand_n, sizeof(cand[0]), sample_candidate_cmp_desc);
        base_sum = full_sum;
        filtered_sum = 0.0f;
        filtered = 0;
        const float full_min_raw = cand[0].prob * min_p;
        for (uint32_t i = 0; i < cand_n; i++) {
            if (i > 0 && cand[i].prob < full_min_raw) break;
            filtered_sum += cand[i].prob;
            filtered++;
            if (filtered_sum / base_sum >= top_p) break;
        }
        owned = true;
    }

    if (filtered == 0 || filtered_sum <= 0.0f || !isfinite(filtered_sum)) {
        probs[cand[0].id] = 1.0f;
    } else {
        const float inv_sum = 1.0f / filtered_sum;
        for (uint32_t i = 0; i < filtered; i++) {
            probs[cand[i].id] = cand[i].prob * inv_sum;
        }
    }
    if (owned) free(cand);
    return true;
}


int main(int argc,char **argv) {
 if(argc!=6)return 2;
 unsigned int n=atoi(argv[1]); float temperature=strtof(argv[2],0);
 int topk=atoi(argv[3]);float topp=strtof(argv[4],0),minp=strtof(argv[5],0);
 float *logits=malloc(n*sizeof(float)),*prob=malloc(n*sizeof(float));
 for(unsigned int i=0;i<n;i++) if(scanf("%f",&logits[i])!=1)return 3;
 if(!sample_build_probabilities(logits,n,temperature,topk,topp,minp,prob))return 4;
 for(unsigned int i=0;i<n;i++)printf("%.9g\n",prob[i]);
 free(logits);free(prob);
}
