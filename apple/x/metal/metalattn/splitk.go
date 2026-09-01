// SPDX-License-Identifier: Apache-2.0

package metalattn

import "fmt"

// SplitKConfig configures sequence partitioning for long-context FlashDecoding.
type SplitKConfig struct {
	NumSplits int // Number of K-sequence partitions along the sequence dimension
	BlockSize int // Number of tokens per partition chunk
}

// SplitKFlashDecodingMSL is the Metal compute shader implementing parallel Split-K FlashDecoding.
const SplitKFlashDecodingMSL = `
#include <metal_stdlib>
using namespace metal;

// Stage 1: Partial FlashAttention reduction per split partition
kernel void SplitKFlashAttentionStage1(
    device const float* q              [[buffer(0)]],
    device const float* k_cache        [[buffer(1)]],
    device const float* v_cache        [[buffer(2)]],
    device float* partial_out          [[buffer(3)]], // [num_splits, num_q_heads, head_dim]
    device float* partial_lse          [[buffer(4)]], // [num_splits, num_q_heads]
    constant int& cur_pos              [[buffer(5)]],
    constant int& num_q_heads          [[buffer(6)]],
    constant int& num_kv_heads         [[buffer(7)]],
    constant int& head_dim             [[buffer(8)]],
    constant float& scale              [[buffer(9)]],
    constant int& num_splits           [[buffer(10)]],
    uint2 split_head_idx               [[threadgroup_position_in_grid]], // (split_id, head_id)
    uint tid                           [[thread_position_in_threadgroup]]
) {
    uint split_id = split_head_idx.x;
    uint head_id = split_head_idx.y;
    if (head_id >= uint(num_q_heads) || split_id >= uint(num_splits)) return;

    int tokens_per_split = (cur_pos + int(num_splits)) / int(num_splits);
    int start_pos = int(split_id) * tokens_per_split;
    int end_pos = min(start_pos + tokens_per_split, cur_pos + 1);
    if (start_pos >= end_pos) return;

    int kv_group_size = num_q_heads / num_kv_heads;
    int kv_head_idx = head_id / kv_group_size;

    device const float* q_ptr = q + head_id * head_dim;
    float max_score = -1e20f;

    // Local score reduction
    threadgroup float local_scores[256];
    for (int t = start_pos + int(tid); t < end_pos; t += 128) {
        device const float* k_ptr = k_cache + (t * num_kv_heads + kv_head_idx) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_ptr[d] * k_ptr[d];
        }
        dot *= scale;
        local_scores[t - start_pos] = dot;
        if (dot > max_score) {
            max_score = dot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_exp = 0.0f;
    for (int t = start_pos + int(tid); t < end_pos; t += 128) {
        float e = exp(local_scores[t - start_pos] - max_score);
        local_scores[t - start_pos] = e;
        sum_exp += e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write partial outputs
    int split_offset = (int(split_id) * num_q_heads + int(head_id)) * head_dim;
    for (int d = int(tid); d < head_dim; d += 128) {
        float acc = 0.0f;
        for (int t = start_pos; t < end_pos; t++) {
            device const float* v_ptr = v_cache + (t * num_kv_heads + kv_head_idx) * head_dim;
            acc += local_scores[t - start_pos] * v_ptr[d];
        }
        partial_out[split_offset + d] = acc;
    }

    if (tid == 0) {
        partial_lse[split_id * num_q_heads + head_id] = max_score + log(sum_exp + 1e-6f);
    }
}

// Stage 2: Log-Sum-Exp Reduction across splits
kernel void SplitKFlashAttentionStage2(
    device const float* partial_out    [[buffer(0)]],
    device const float* partial_lse    [[buffer(1)]],
    device float* out                  [[buffer(2)]],
    constant int& num_q_heads          [[buffer(3)]],
    constant int& head_dim             [[buffer(4)]],
    constant int& num_splits           [[buffer(5)]],
    uint head_id                       [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]]
) {
    if (head_id >= uint(num_q_heads)) return;

    // Find global max LSE
    float max_lse = -1e20f;
    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[s * num_q_heads + head_id];
        if (lse > max_lse) max_lse = lse;
    }

    float total_weight = 0.0f;
    float weights[32];
    for (int s = 0; s < num_splits && s < 32; s++) {
        float lse = partial_lse[s * num_q_heads + head_id];
        float w = exp(lse - max_lse);
        weights[s] = w;
        total_weight += w;
    }

    // Weighted accumulation across splits
    device float* out_ptr = out + head_id * head_dim;
    for (int d = int(tid); d < head_dim; d += 128) {
        float final_val = 0.0f;
        for (int s = 0; s < num_splits && s < 32; s++) {
            int split_offset = (s * num_q_heads + int(head_id)) * head_dim;
            final_val += partial_out[split_offset + d] * weights[s];
        }
        if (total_weight > 0.0f) {
            final_val /= total_weight;
        }
        out_ptr[d] = final_val;
    }
}
`

// GenerateSplitKShader returns the Split-K shader source for Metal GPU compilation.
func GenerateSplitKShader(cfg SplitKConfig) string {
	return fmt.Sprintf("// Split-K FlashDecoding (Splits: %d, BlockSize: %d)\n%s",
		cfg.NumSplits, cfg.BlockSize, SplitKFlashDecodingMSL)
}
