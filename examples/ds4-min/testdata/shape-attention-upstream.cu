__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t score_lanes_single) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    const uint32_t score_threads = blockDim.x > 256u ? 256u : blockDim.x;
    const bool score_thread = threadIdx.x < score_threads;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    if (score_thread) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += score_threads) {
            raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
        }
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    if (score_thread) {
        if (visible_comp == 0 || (n_tokens == 1u && score_lanes_single == 0u)) {
            for (uint32_t r = threadIdx.x; r < raw_count; r += score_threads) {
                const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                scores[r] = dot * scale;
                local_max = fmaxf(local_max, scores[r]);
            }
            for (uint32_t c = threadIdx.x; c < visible_comp; c += score_threads) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                float s = -INFINITY;
                if (add > -1.0e20f) {
                    const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                    s = dot * scale + add;
                }
                scores[raw_count + c] = s;
                local_max = fmaxf(local_max, s);
            }
        } else if (n_tokens == 1u && score_lanes_single == 4u) {
            uint32_t qlane = threadIdx.x & 3u;
            uint32_t qgroup = threadIdx.x >> 2u;
            for (uint32_t row0 = 0; row0 < n_score; row0 += 64u) {
                uint32_t row = row0 + qgroup;
                if (row < n_score) {
                    float add = 0.0f;
                    const float *kvrow = NULL;
                    if (row < raw_count) {
                        kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                    } else {
                        uint32_t c = row - raw_count;
                        add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                        if (add > -1.0e20f) kvrow = comp_kv + (uint64_t)c * head_dim;
                    }
                    float s = -INFINITY;
                    if (kvrow) {
                        float dot = 0.0f;
                        for (uint32_t d = qlane; d < head_dim; d += 4u) dot += qh[d] * kvrow[d];
                        const uint32_t mask = 0xfu << (threadIdx.x & 28u);
                        dot += __shfl_down_sync(mask, dot, 2, 4);
                        dot += __shfl_down_sync(mask, dot, 1, 4);
                        s = dot * scale + add;
                    }
                    if (qlane == 0) scores[row] = s;
                }
            }
            __syncthreads();
            for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
                local_max = fmaxf(local_max, scores[i]);
            }
        } else {
            uint32_t qlane = threadIdx.x & 7u;
            uint32_t qgroup = threadIdx.x >> 3u;
            for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
                uint32_t row = row0 + qgroup;
                if (row < n_score) {
                    float add = 0.0f;
                    const float *kvrow = NULL;
                    if (row < raw_count) {
                        kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                    } else {
                        uint32_t c = row - raw_count;
                        add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                        if (add > -1.0e20f) kvrow = comp_kv + (uint64_t)c * head_dim;
                    }
                    float s = -INFINITY;
                    if (kvrow) {
                        float dot = 0.0f;
                        for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                        const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                        for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                            dot += __shfl_down_sync(mask, dot, off, 8);
                        }
                        s = dot * scale + add;
                    }
                    if (qlane == 0) scores[row] = s;
                }
            }
            __syncthreads();
            for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
                local_max = fmaxf(local_max, scores[i]);
            }
        }
    }
    if (score_thread) partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            scores[i] = expf(scores[i] - max_s);
            den_local += scores[i];
        }
    }
    if (score_thread) partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x >= 512u) {
        uint32_t d = threadIdx.x;
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc += kv[d] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc += kv[d] * s;
        }
        oh[d] = acc / denom;
    } else if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
            oh[d] = acc / denom;
        }
    }
}
