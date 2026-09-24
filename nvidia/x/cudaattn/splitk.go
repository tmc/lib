// SPDX-License-Identifier: Apache-2.0

package cudaattn

import (
	"fmt"
)

// SplitKConfig configures sequence partitioning for long-context CUDA FlashDecoding.
type SplitKConfig struct {
	NumSplits int // Number of K partitions along sequence dimension
	BlockSize int // Tokens per partition
}

// SplitKCUDATemplate contains the 2-stage parallel Split-K FlashDecoding CUDA kernel.
const SplitKCUDATemplate = `
extern "C" {

// Stage 1: Partial FlashAttention reduction per split partition
__global__ void SplitKFlashAttentionStage1(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ partial_out,
    float* __restrict__ partial_lse,
    int cur_pos,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int num_splits
) {
    int split_id = blockIdx.x;
    int head_id = blockIdx.y;
    if (head_id >= num_q_heads || split_id >= num_splits) return;

    int tokens_per_split = (cur_pos + num_splits) / num_splits;
    int start_pos = split_id * tokens_per_split;
    int end_pos = start_pos + tokens_per_split;
    if (end_pos > cur_pos + 1) end_pos = cur_pos + 1;
    if (start_pos >= end_pos) return;

    int kv_group_size = num_q_heads / num_kv_heads;
    int kv_head_idx = head_id / kv_group_size;

    const float* q_ptr = q + head_id * head_dim;
    float max_score = -1e20f;

    extern __shared__ float local_scores[];

    for (int t = start_pos + threadIdx.x; t < end_pos; t += blockDim.x) {
        const float* k_ptr = k_cache + (t * num_kv_heads + kv_head_idx) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_ptr[d] * k_ptr[d];
        }
        dot *= scale;
        local_scores[t - start_pos] = dot;
        if (dot > max_score) max_score = dot;
    }
    __syncthreads();

    float sum_exp = 0.0f;
    for (int t = start_pos + threadIdx.x; t < end_pos; t += blockDim.x) {
        float e = expf(local_scores[t - start_pos] - max_score);
        local_scores[t - start_pos] = e;
        sum_exp += e;
    }
    __syncthreads();

    int split_offset = (split_id * num_q_heads + head_id) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = start_pos; t < end_pos; t++) {
            const float* v_ptr = v_cache + (t * num_kv_heads + kv_head_idx) * head_dim;
            acc += local_scores[t - start_pos] * v_ptr[d];
        }
        partial_out[split_offset + d] = acc;
    }

    if (threadIdx.x == 0) {
        partial_lse[split_id * num_q_heads + head_id] = max_score + logf(sum_exp + 1e-6f);
    }
}

// Stage 2: Log-Sum-Exp reduction across splits
__global__ void SplitKFlashAttentionStage2(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_lse,
    float* __restrict__ out,
    int num_q_heads,
    int head_dim,
    int num_splits
) {
    int head_id = blockIdx.x;
    if (head_id >= num_q_heads) return;

    float max_lse = -1e20f;
    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[s * num_q_heads + head_id];
        if (lse > max_lse) max_lse = lse;
    }

    float total_weight = 0.0f;
    float weights[32];
    for (int s = 0; s < num_splits && s < 32; s++) {
        float lse = partial_lse[s * num_q_heads + head_id];
        float w = expf(lse - max_lse);
        weights[s] = w;
        total_weight += w;
    }

    float* out_ptr = out + head_id * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float final_val = 0.0f;
        for (int s = 0; s < num_splits && s < 32; s++) {
            int split_offset = (s * num_q_heads + head_id) * head_dim;
            final_val += partial_out[split_offset + d] * weights[s];
        }
        if (total_weight > 0.0f) {
            final_val /= total_weight;
        }
        out_ptr[d] = final_val;
    }
}

}
`

// GenerateSplitKCUDAShader returns the Split-K CUDA C++ shader source.
func GenerateSplitKCUDAShader(cfg SplitKConfig) string {
	return fmt.Sprintf("// CUDA Split-K FlashDecoding (Splits: %d)\n%s", cfg.NumSplits, SplitKCUDATemplate)
}
