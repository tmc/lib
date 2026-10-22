// Standalone NVRTC source: no host system headers are available.
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
static_assert(sizeof(int32_t) == 4 && sizeof(uint32_t) == 4 && sizeof(uint64_t) == 8, "integer widths");

// Adapted from pinned CUDA scalar router with dynamic expert count and scale.
// Pro's CPU scale is 2.5; the pinned CUDA wrapper wrongly admits only 1.5.
__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

extern "C" __global__ void ds4_shape_router_select(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode,
        uint32_t n_expert,
        float expert_scale,
        uint32_t *invalid) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *log = logits + (uint64_t)t * n_expert;
    float *prob = probs + (uint64_t)t * n_expert;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;

    for (uint32_t i = 0; i < n_expert; i++) {
        if (!isfinite(log[i]) || (has_bias && !isfinite(bias[i]))) { *invalid = 1; return; }
        prob[i] = sqrtf(softplus_dev(log[i]));
    }

    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int i = 0; i < 6; i++) sel[i] = row[i];
    } else {
        for (int i = 0; i < 6; i++) sel[i] = -1;
        for (int i = 0; i < n_expert; i++) {
            float score = prob[i] + (has_bias ? bias[i] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > prob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = i;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < 6; i++) {
        int e = sel[i];
        if (e < 0 || (uint32_t)e >= n_expert) { *invalid = 1; return; }
        float v = prob[e];
        w[i] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int i = 0; i < 6; i++) w[i] = w[i] / sum * expert_scale;
}
