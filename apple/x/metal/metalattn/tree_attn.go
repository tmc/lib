// SPDX-License-Identifier: Apache-2.0

package metalattn

// TreeAttentionMaxNodes is the largest candidate tree a uint64 row bitmask can
// encode. Wider trees have no bitmask representation and are truncated to the
// first TreeAttentionMaxNodes nodes.
const TreeAttentionMaxNodes = 64

// TreeAttentionMSL is the Metal compute shader executing 2D Causal Tree Attention with 64-bit row bitmasks.
//
// Dispatch it with a threadgroup grid of (num_nodes, num_heads) and a
// power-of-two threadgroup width that is a multiple of the 32-lane SIMD group,
// up to 1024. The kernel reads its own width, so any such width is valid.
const TreeAttentionMSL = `
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

constant constexpr int TREE_ATTN_MAX_NODES = 64;
constant constexpr float TREE_ATTN_NEG_INF = -1e30f;

// TreeFlashAttention performs multi-path causal attention using bitmask-encoded ancestor trees.
// mask_bitmasks has 1 uint64 per tree node where bit j is 1 iff node j is an ancestor of node i.
//
// The row maximum and the softmax denominator are reduced across the whole
// threadgroup before either is used: each thread walks a strided slice of the
// row, and the output loop partitions by head_dim rather than by node, so a
// per-thread statistic would leave the divisor unrelated to the numerator.
kernel void TreeFlashAttention(
    device const float* q              [[buffer(0)]], // [num_nodes, num_heads, head_dim]
    device const float* k_cache        [[buffer(1)]], // [num_nodes, num_heads, head_dim]
    device const float* v_cache        [[buffer(2)]], // [num_nodes, num_heads, head_dim]
    device float* out                  [[buffer(3)]], // [num_nodes, num_heads, head_dim]
    device const ulong* bitmasks       [[buffer(4)]], // [num_nodes]
    constant int& num_nodes            [[buffer(5)]],
    constant int& num_heads            [[buffer(6)]],
    constant int& head_dim             [[buffer(7)]],
    constant float& scale              [[buffer(8)]],
    uint2 node_head_idx                [[threadgroup_position_in_grid]], // (node_id, head_id)
    uint2 tid_in_tg                    [[thread_position_in_threadgroup]],
    uint2 tg_dims                      [[threads_per_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    // MSL requires every grid-position input to share a dimensionality, so the
    // threadgroup-local coordinates arrive as vectors alongside the 2D grid
    // position even though only their x component is used.
    const uint tid = tid_in_tg.x;
    const uint tg_size = tg_dims.x;

    uint node_id = node_head_idx.x;
    uint head_id = node_head_idx.y;
    if (node_id >= uint(num_nodes) || head_id >= uint(num_heads)) return;

    // A row bitmask holds one bit per node, so wider trees are not addressable
    // and the score array cannot be indexed past its bound.
    const int n = min(num_nodes, TREE_ATTN_MAX_NODES);
    const uint num_simds = (tg_size + 31u) / 32u;

    const ulong ancestor_mask = bitmasks[node_id];
    device const float* q_ptr = q + (node_id * num_heads + head_id) * head_dim;

    threadgroup float scores[TREE_ATTN_MAX_NODES];
    threadgroup float s_red[32];

    // Scores for the ancestor set, then a threadgroup-wide maximum.
    float thread_max = TREE_ATTN_NEG_INF;
    for (int t = int(tid); t < n; t += int(tg_size)) {
        if ((ancestor_mask >> t) & 1UL) {
            device const float* k_ptr = k_cache + (t * num_heads + head_id) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_ptr[d] * k_ptr[d];
            }
            dot *= scale;
            scores[t] = dot;
            thread_max = max(thread_max, dot);
        } else {
            scores[t] = TREE_ATTN_NEG_INF;
        }
    }

    thread_max = attn_simd_max(thread_max);
    if (simd_lane == 0) s_red[simd_id] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_score = TREE_ATTN_NEG_INF;
    for (uint i = 0; i < num_simds; i++) {
        max_score = max(max_score, s_red[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exponentiate, then reduce the denominator across the threadgroup too.
    float thread_sum = 0.0f;
    for (int t = int(tid); t < n; t += int(tg_size)) {
        float e = 0.0f;
        if ((ancestor_mask >> t) & 1UL) {
            e = exp(scores[t] - max_score);
        }
        scores[t] = e;
        thread_sum += e;
    }

    thread_sum = attn_simd_sum(thread_sum);
    if (simd_lane == 0) s_red[simd_id] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_exp = 0.0f;
    for (uint i = 0; i < num_simds; i++) {
        sum_exp += s_red[i];
    }

    // An empty ancestor row emits zeros rather than an unnormalized sum.
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    device float* out_ptr = out + (node_id * num_heads + head_id) * head_dim;
    for (int d = int(tid); d < head_dim; d += int(tg_size)) {
        float acc = 0.0f;
        for (int t = 0; t < n; t++) {
            if ((ancestor_mask >> t) & 1UL) {
                device const float* v_ptr = v_cache + (t * num_heads + head_id) * head_dim;
                acc += scores[t] * v_ptr[d];
            }
        }
        out_ptr[d] = acc * inv_sum;
    }
}
`

// GenerateTreeAttentionMSL returns the bitmask-accelerated tree attention shader source.
func GenerateTreeAttentionMSL() string {
	return TreeAttentionMSL
}
