typedef unsigned int uint32_t;
static_assert(sizeof(uint32_t) == 4, "integer width");
#define INFINITY __int_as_float(0x7f800000)

// Correctness-first dynamic top-k consumer. The ordering contract is score
// descending, then cache index ascending. Unlike the Flash selector, k is not
// compiled as 512. Full backend admission must measure this serial route.
extern "C" __global__ void ds4_shape_indexer_select(const float *scores,
        float *mask, uint32_t *invalid, uint32_t rows, uint32_t k) {
    if (blockIdx.x || threadIdx.x) return;
    *invalid = 0;
    if (rows <= k) {
        for (uint32_t i = 0; i < rows; i++) mask[i] = 0.0f;
        return;
    }
    for (uint32_t i = 0; i < rows; i++) {
        mask[i] = -INFINITY;
        if (!isfinite(scores[i])) *invalid = 1;
    }
    if (*invalid) return;
    for (uint32_t n = 0; n < k && n < rows; n++) {
        uint32_t best = rows;
        for (uint32_t i = 0; i < rows; i++) {
            if (mask[i] == 0.0f) continue;
            if (best == rows || scores[i] > scores[best]) best = i;
        }
        if (best == rows) { *invalid = 1; return; }
        mask[best] = 0.0f;
    }
}
