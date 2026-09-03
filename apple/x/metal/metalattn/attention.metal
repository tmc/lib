#include <metal_stdlib>
using namespace metal;

#ifndef METALATTN_REDUCE_HELPERS
#define METALATTN_REDUCE_HELPERS

// These are duplicated verbatim in every metalattn shader string so each one
// compiles standalone; the guard keeps concatenating several of them legal.
inline float attn_simd_sum(float v) {
    for (ushort off = 16; off > 0; off >>= 1) {
        v += simd_shuffle_down(v, off);
    }
    return v;
}

inline float attn_simd_max(float v) {
    for (ushort off = 16; off > 0; off >>= 1) {
        v = max(v, simd_shuffle_down(v, off));
    }
    return v;
}

#endif

constant constexpr int ATTN_TILE = 256;
constant constexpr int ATTN_MAX_HEAD_DIM = 256;
constant constexpr float ATTN_NEG_INF = -1e30f;

// CausalFlashAttentionDecode computes single-query attention against the KV cache
// for tokens 0..cur_pos, one threadgroup per query head.
//
// Keys and values are streamed in tiles of ATTN_TILE with an online softmax: the
// running maximum and running denominator are carried across tiles in threadgroup
// memory and the accumulator is rescaled whenever the maximum grows. Both the
// per-tile maximum and the per-tile exponential sum are reduced across the whole
// threadgroup before use, so every thread normalizes by the same denominator.
kernel void CausalFlashAttentionDecode(
    device const float* q              [[buffer(0)]],
    device const float* k_cache        [[buffer(1)]],
    device const float* v_cache        [[buffer(2)]],
    device float* out                  [[buffer(3)]],
    constant int& cur_pos              [[buffer(4)]],
    constant int& max_seq_len          [[buffer(5)]],
    constant int& num_q_heads          [[buffer(6)]],
    constant int& num_kv_heads         [[buffer(7)]],
    constant int& head_dim             [[buffer(8)]],
    constant float& scale              [[buffer(9)]],
    uint head_idx                      [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (head_idx >= uint(num_q_heads)) return;
    if (head_dim > ATTN_MAX_HEAD_DIM) return;

    const int kv_group_size = num_q_heads / num_kv_heads;
    const int kv_head_idx = int(head_idx) / kv_group_size;
    const int seq_len = min(cur_pos + 1, max_seq_len);

    threadgroup float s_scores[ATTN_TILE];
    threadgroup float s_acc[ATTN_MAX_HEAD_DIM];
    threadgroup float s_red[4];
    threadgroup float s_max;
    threadgroup float s_denom;

    device const float* q_ptr = q + int(head_idx) * head_dim;
    device float* out_ptr = out + int(head_idx) * head_dim;

    for (int d = int(tid); d < head_dim; d += 128) {
        s_acc[d] = 0.0f;
    }
    if (tid == 0) {
        s_max = ATTN_NEG_INF;
        s_denom = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int base = 0; base < seq_len; base += ATTN_TILE) {
        const int tile = min(ATTN_TILE, seq_len - base);

        // Scores for this tile, plus a threadgroup-wide maximum.
        float local_max = ATTN_NEG_INF;
        for (int j = int(tid); j < tile; j += 128) {
            device const float* k_ptr = k_cache + ((base + j) * num_kv_heads + kv_head_idx) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_ptr[d] * k_ptr[d];
            }
            dot *= scale;
            s_scores[j] = dot;
            local_max = max(local_max, dot);
        }
        local_max = attn_simd_max(local_max);
        if (simd_lane == 0) s_red[simd_id] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float tile_max = max(max(s_red[0], s_red[1]), max(s_red[2], s_red[3]));

        // Rescale factor for everything accumulated from earlier tiles.
        const float new_max = max(s_max, tile_max);
        const float corr = (s_max <= ATTN_NEG_INF) ? 0.0f : exp(s_max - new_max);

        // Exponentiate against the new running maximum, and sum across the threadgroup.
        float local_sum = 0.0f;
        for (int j = int(tid); j < tile; j += 128) {
            const float e = exp(s_scores[j] - new_max);
            s_scores[j] = e;
            local_sum += e;
        }
        local_sum = attn_simd_sum(local_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) s_red[simd_id] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float tile_sum = s_red[0] + s_red[1] + s_red[2] + s_red[3];

        // Accumulate this tile's weighted values, rescaling the running total.
        for (int d = int(tid); d < head_dim; d += 128) {
            float acc = 0.0f;
            for (int j = 0; j < tile; j++) {
                device const float* v_ptr = v_cache + ((base + j) * num_kv_heads + kv_head_idx) * head_dim;
                acc += s_scores[j] * v_ptr[d];
            }
            s_acc[d] = s_acc[d] * corr + acc;
        }

        if (tid == 0) {
            s_max = new_max;
            s_denom = s_denom * corr + tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float denom = s_denom;
    for (int d = int(tid); d < head_dim; d += 128) {
        out_ptr[d] = (denom > 0.0f) ? (s_acc[d] / denom) : 0.0f;
    }
}

// BlockTableCausalAttentionDecode reads immutable prefixes from GPU-address
// descriptor tables, followed by independently writable contiguous suffixes.
// Prefix storage is never repacked. The encoder declares each indirectly
// referenced block with useResource before dispatch.
struct BlockAttentionDescriptor {
    uint start;
    uint end;
    ulong k_address;
    ulong v_address;
};

struct BlockAttentionRow {
    uint prefix_tokens;
    uint suffix_tokens;
    uint suffix_offset;
    uint block_offset;
    uint block_count;
};

kernel void BlockTableCausalAttentionDecode(
    device const float* q [[buffer(0)]],
    device const float* suffix_k [[buffer(1)]],
    device const float* suffix_v [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const BlockAttentionDescriptor* blocks [[buffer(4)]],
    device const BlockAttentionRow* rows [[buffer(5)]],
    constant uint& num_q_heads [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint head_idx = group % num_q_heads;
    const uint row_idx = group / num_q_heads;
    const uint simd_lane = tid & 31;
    const uint simd_id = tid >> 5;
    if (head_idx >= num_q_heads || head_dim > ATTN_MAX_HEAD_DIM) return;
    const BlockAttentionRow p = rows[row_idx];
    const uint kv_head = head_idx / (num_q_heads / num_kv_heads);
    const uint total = p.prefix_tokens + p.suffix_tokens;
    const uint kv_dim = num_kv_heads * head_dim;
    threadgroup float scores[ATTN_TILE];
    threadgroup float acc[ATTN_MAX_HEAD_DIM];
    threadgroup float red[4];
    threadgroup float run_max;
    threadgroup float run_sum;
    for (uint d = tid; d < head_dim; d += 128) acc[d] = 0.0f;
    if (tid == 0) { run_max = ATTN_NEG_INF; run_sum = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = 0; base < total; base += ATTN_TILE) {
        const uint tile = min(uint(ATTN_TILE), total - base);
        float local_max = ATTN_NEG_INF;
        for (uint j = tid; j < tile; j += 128) {
            const uint token = base + j;
            device const float* kp = suffix_k + p.suffix_offset * kv_dim;
            uint local = token - p.prefix_tokens;
            if (token < p.prefix_tokens) {
                for (uint bi = 0; bi < p.block_count; bi++) {
                    const BlockAttentionDescriptor block = blocks[p.block_offset + bi];
                    if (token < block.start || token >= block.end) continue;
                    local = token - block.start;
                    kp = reinterpret_cast<device const float*>(block.k_address);
                    break;
                }
            }
            kp += (local * num_kv_heads + kv_head) * head_dim;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d++) dot += q[(row_idx*num_q_heads+head_idx)*head_dim+d] * kp[d];
            scores[j] = dot * scale;
            local_max = max(local_max, scores[j]);
        }
        local_max = attn_simd_max(local_max);
        if (simd_lane == 0) red[simd_id] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float tile_max = max(max(red[0], red[1]), max(red[2], red[3]));
        const float next_max = max(run_max, tile_max);
        const float correction = run_max <= ATTN_NEG_INF ? 0.0f : exp(run_max-next_max);
        float local_sum = 0.0f;
        for (uint j = tid; j < tile; j += 128) { scores[j] = exp(scores[j]-next_max); local_sum += scores[j]; }
        local_sum = attn_simd_sum(local_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) red[simd_id] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float tile_sum = red[0]+red[1]+red[2]+red[3];
        for (uint d = tid; d < head_dim; d += 128) {
            float sum = 0.0f;
            for (uint j = 0; j < tile; j++) {
                const uint token = base+j;
                device const float* vp = suffix_v + p.suffix_offset * kv_dim;
                uint local = token-p.prefix_tokens;
                if (token < p.prefix_tokens) {
                    for (uint bi = 0; bi < p.block_count; bi++) {
                        const BlockAttentionDescriptor block = blocks[p.block_offset + bi];
                        if (token < block.start || token >= block.end) continue;
                        local = token-block.start;
                        vp = reinterpret_cast<device const float*>(block.v_address);
                        break;
                    }
                }
                vp += (local*num_kv_heads+kv_head)*head_dim;
                sum += scores[j]*vp[d];
            }
            acc[d] = acc[d]*correction+sum;
        }
        if (tid == 0) { run_max=next_max; run_sum=run_sum*correction+tile_sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d=tid; d<head_dim; d+=128) out[(row_idx*num_q_heads+head_idx)*head_dim+d] = run_sum > 0 ? acc[d]/run_sum : 0;
}
