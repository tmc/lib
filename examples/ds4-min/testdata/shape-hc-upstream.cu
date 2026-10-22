__global__ static void hc_weighted_sum_kernel(float *out, const float *x, const float *w, uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens, uint32_t weight_stride_f32) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_tokens;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint32_t t = gid / n_embd;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += x[(uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d] *
               w[(uint64_t)t * weight_stride_f32 + h];
    }
    out[(uint64_t)t * n_embd + d] = acc;
}

__global__ static void hc_expand_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *block_add2,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add,
        int has_add2) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) {
        float add_v = block_add[(uint64_t)t * n_embd + d];
        if (has_add2) add_v += block_add2[(uint64_t)t * n_embd + d];
        block_v += add_v;
    }
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}
