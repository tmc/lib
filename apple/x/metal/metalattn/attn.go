package metalattn

import (
	"math"
)

// Config defines the dimensional configuration for Multi-Head or Grouped Query Attention.
type Config struct {
	NumQHeads  int
	NumKVHeads int
	HeadDim    int
	Scale      float32
}

// DefaultConfig creates an attention configuration with standard 1/sqrt(headDim) softmax scale.
func DefaultConfig(numQHeads, numKVHeads, headDim int) Config {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	return Config{
		NumQHeads:  numQHeads,
		NumKVHeads: numKVHeads,
		HeadDim:    headDim,
		Scale:      scale,
	}
}

// Dispatch invariants for CausalFlashAttentionDecode.
//
// The kernel reduces across exactly DecodeThreadgroupSize/32 SIMD groups, so it
// must be dispatched with DecodeThreadgroupSize threads per threadgroup and one
// threadgroup per query head. Sequence length is unbounded: keys and values are
// consumed in tiles of DecodeTileSize with an online-softmax rescale between
// tiles. Head dimension is bounded by DecodeMaxHeadDim, which is the size of the
// threadgroup output accumulator.
const (
	DecodeThreadgroupSize = 128
	DecodeTileSize        = 256
	DecodeMaxHeadDim      = 256
)

// CausalFlashAttentionMSL is the embedded Metal Shading Language compute shader for causal attention decode.
const CausalFlashAttentionMSL = `
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
`
