// SPDX-License-Identifier: Apache-2.0

package metalattn

// TreeAttentionMSL is the Metal compute shader executing 2D Causal Tree Attention with 64-bit row bitmasks.
const TreeAttentionMSL = `
#include <metal_stdlib>
using namespace metal;

// TreeFlashAttention performs multi-path causal attention using bitmask-encoded ancestor trees.
// mask_bitmasks has 1 uint64 per tree node where bit j is 1 iff node j is an ancestor of node i.
kernel void TreeFlashAttention(
    device const float* q              [[buffer(0)]], // [num_nodes, num_heads, head_dim]
    device const float* k_cache        [[buffer(1)]], // [num_nodes, num_heads, head_dim]
    device const float* v_cache        [[buffer(2)]], // [num_nodes, num_heads, head_dim]
    device float* out                  [[buffer(3)]], // [num_nodes, num_heads, head_dim]
    device const uint64_t* bitmasks    [[buffer(4)]], // [num_nodes]
    constant int& num_nodes            [[buffer(5)]],
    constant int& num_heads            [[buffer(6)]],
    constant int& head_dim             [[buffer(7)]],
    constant float& scale              [[buffer(8)]],
    uint2 node_head_idx                [[threadgroup_position_in_grid]], // (node_id, head_id)
    uint tid                           [[thread_position_in_threadgroup]]
) {
    uint node_id = node_head_idx.x;
    uint head_id = node_head_idx.y;
    if (node_id >= uint(num_nodes) || head_id >= uint(num_heads)) return;

    uint64_t ancestor_mask = bitmasks[node_id];
    device const float* q_ptr = q + (node_id * num_heads + head_id) * head_dim;

    threadgroup float scores[64];
    float max_score = -1e20f;

    // Step 1: Compute Q * K^T only for valid ancestor nodes via bitmask check
    for (int t = int(tid); t < num_nodes; t += 32) {
        bool is_ancestor = (ancestor_mask & (1ULL << t)) != 0;
        if (is_ancestor) {
            device const float* k_ptr = k_cache + (t * num_heads + head_id) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_ptr[d] * k_ptr[d];
            }
            dot *= scale;
            scores[t] = dot;
            if (dot > max_score) max_score = dot;
        } else {
            scores[t] = -1e20f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Softmax over ancestors
    float sum_exp = 0.0f;
    for (int t = int(tid); t < num_nodes; t += 32) {
        if ((ancestor_mask & (1ULL << t)) != 0) {
            float e = exp(scores[t] - max_score);
            scores[t] = e;
            sum_exp += e;
        } else {
            scores[t] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Weighted V accumulation
    device float* out_ptr = out + (node_id * num_heads + head_id) * head_dim;
    for (int d = int(tid); d < head_dim; d += 32) {
        float acc = 0.0f;
        for (int t = 0; t < num_nodes; t++) {
            if ((ancestor_mask & (1ULL << t)) != 0) {
                device const float* v_ptr = v_cache + (t * num_heads + head_id) * head_dim;
                acc += scores[t] * v_ptr[d];
            }
        }
        if (sum_exp > 0.0f) acc /= sum_exp;
        out_ptr[d] = acc;
    }
}
`

// GenerateTreeAttentionMSL returns the bitmask-accelerated tree attention shader source.
func GenerateTreeAttentionMSL() string {
	return TreeAttentionMSL
}
