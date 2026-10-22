/* Literal ds4.c bodies at 11e8cad4473716a41baf4d62e92ed9764a00fce6.
 * RNG 42785:42798, sample_probabilities 42983:43006,
 * point acceptance 74520:74526, replacement 74528:74538.
 * Only the surrounding bounded driver and draw hook are fixture code.
 */
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
/* BEGIN PINNED RNG */
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

/* END PINNED RNG */
static unsigned draw_count, scripted_count, scripted_at;
static float scripted[16];
static float reference_draw(uint64_t *state) {
    draw_count++;
    if (scripted_at < scripted_count) return scripted[scripted_at++];
    return sample_rng_f32(state);
}
#define sample_rng_f32 reference_draw
static uint32_t fixture_vocab;
#define DS4_N_VOCAB fixture_vocab
typedef struct { float *sample_probs; } ds4_session;
/* BEGIN PINNED SAMPLE */
static int sample_probabilities(const float *probs, uint32_t n_vocab,
                                uint64_t *rng) {
    if (!probs || !rng || n_vocab == 0) return -1;
    float sum = 0.0f;
    int best = 0;
    float best_p = probs[0];
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float p = probs[i];
        if (p > 0.0f && isfinite(p)) sum += p;
        if (p > best_p) {
            best_p = p;
            best = (int)i;
        }
    }
    if (sum <= 0.0f || !isfinite(sum)) return best;
    float r = sample_rng_f32(rng) * sum;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float p = probs[i];
        if (p <= 0.0f || !isfinite(p)) continue;
        r -= p;
        if (r <= 0.0f) return (int)i;
    }
    return best;
}

/* END PINNED SAMPLE */
/* BEGIN PINNED ACCEPT */
static bool speculative_point_accept(float target_p, float draft_p,
                                     uint64_t *rng) {
    if (!rng || !(draft_p > 0.0f) || !isfinite(draft_p)) return false;
    if (target_p >= draft_p) return true;
    if (!(target_p > 0.0f) || !isfinite(target_p)) return false;
    return sample_rng_f32(rng) * draft_p <= target_p;
}
/* END PINNED ACCEPT */
/* BEGIN PINNED REPLACEMENT */
static int speculative_point_replacement(
        ds4_session *s,
        int draft_token,
        uint64_t *rng) {
    if (!s || !s->sample_probs || !rng ||
        draft_token < 0 || draft_token >= (int)DS4_N_VOCAB) {
        return -1;
    }
    s->sample_probs[draft_token] = 0.0f;
    return sample_probabilities(s->sample_probs, DS4_N_VOCAB, rng);
}
/* END PINNED REPLACEMENT */
int main(void) {
    float probabilities[8192];
    unsigned n, draft;
    uint64_t state;
    while (scanf("%u %u %" SCNu64 " %u", &n, &draft, &state, &scripted_count) == 4) {
        if (n == 0 || n > 8192 || draft >= n || scripted_count > 16) return 2;
        scripted_at = draw_count = 0;
        for (unsigned i = 0; i < scripted_count; i++) {
            uint32_t bits;
            if (scanf("%" SCNx32, &bits) != 1) return 2;
            memcpy(&scripted[i], &bits, sizeof(bits));
        }
        for (unsigned i = 0; i < n; i++) {
            uint32_t bits;
            if (scanf("%" SCNx32, &bits) != 1) return 2;
            memcpy(&probabilities[i], &bits, sizeof(bits));
        }
        fixture_vocab = n;
        ds4_session session = { probabilities };
        const bool accepted = speculative_point_accept(probabilities[draft], 1.0f, &state);
        const int token = accepted ? (int)draft : speculative_point_replacement(&session, (int)draft, &state);
        printf("%d %d %" PRIu64 " %u\n", token, accepted, state, draw_count);
    }
    return ferror(stdin) ? 2 : 0;
}
