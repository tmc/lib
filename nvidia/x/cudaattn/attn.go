package cudaattn

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Config defines the dimensional configuration for Multi-Head or Grouped Query Attention.
type Config struct {
	NumQHeads  int
	NumKVHeads int
	HeadDim    int
	Scale      float32
}

// DefaultConfig creates an attention configuration with the standard 1/sqrt(headDim) softmax scale.
func DefaultConfig(numQHeads, numKVHeads, headDim int) Config {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	return Config{
		NumQHeads:  numQHeads,
		NumKVHeads: numKVHeads,
		HeadDim:    headDim,
		Scale:      scale,
	}
}

// CausalFlashAttentionTemplate is the embedded CUDA C++ kernel for causal single-sequence decode attention.
const CausalFlashAttentionTemplate = `
extern "C" __global__ void CausalFlashAttentionDecode(
    const float* __restrict__ q,       // [batch, num_q_heads, head_dim]
    const float* __restrict__ k_cache, // [batch, max_seq_len, num_kv_heads, head_dim]
    const float* __restrict__ v_cache, // [batch, max_seq_len, num_kv_heads, head_dim]
    float* __restrict__ out,           // [batch, num_q_heads, head_dim]
    int cur_pos,
    int max_seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    int head_idx = blockIdx.x; // query head index
    if (head_idx >= num_q_heads) return;

    int kv_group_size = num_q_heads / num_kv_heads;
    int kv_head_idx = head_idx / kv_group_size;

    extern __shared__ float smem[];
    float* s_scores = smem; // size: cur_pos + 1

    const float* q_ptr = q + head_idx * head_dim;

    // Step 1: Compute Q * K^T dot products for all tokens 0..cur_pos
    float max_score = -1e20f;
    for (int t = threadIdx.x; t <= cur_pos; t += blockDim.x) {
        const float* k_ptr = k_cache + (t * num_kv_heads + kv_head_idx) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_ptr[d] * k_ptr[d];
        }
        dot *= scale;
        s_scores[t] = dot;
        if (dot > max_score) {
            max_score = dot;
        }
    }
    __syncthreads();

    // Step 2: Online Softmax Normalization
    float sum_exp = 0.0f;
    for (int t = threadIdx.x; t <= cur_pos; t += blockDim.x) {
        float e = expf(s_scores[t] - max_score);
        s_scores[t] = e;
        sum_exp += e;
    }
    __syncthreads();

    // Step 3: Weighted sum over V
    float* out_ptr = out + head_idx * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t <= cur_pos; t++) {
            const float* v_ptr = v_cache + (t * num_kv_heads + kv_head_idx) * head_dim;
            acc += s_scores[t] * v_ptr[d];
        }
        if (sum_exp > 0.0f) {
            acc /= sum_exp;
        }
        out_ptr[d] = acc;
    }
}
`

// BatchedPagedAttentionTemplate is the embedded CUDA C++ kernel for multi-sequence paged KV-cache attention.
const BatchedPagedAttentionTemplate = `
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void BatchedPagedAttentionDecode(
    const float* __restrict__ q,              // [num_seqs, num_q_heads, head_dim]
    const float* __restrict__ k_blocks,       // [total_blocks, block_size, num_kv_heads, head_dim]
    const float* __restrict__ v_blocks,       // [total_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,     // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,         // [num_seqs]
    float* __restrict__ out,                  // [num_seqs, num_q_heads, head_dim]
    int max_blocks_per_seq,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    if (head_idx >= num_q_heads) return;

    int seq_len = seq_lens[seq_idx];
    if (seq_len <= 0) return;

    int kv_group_size = num_q_heads / num_kv_heads;
    int kv_head_idx = head_idx / kv_group_size;
    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    extern __shared__ float smem[];
    float* s_scores = smem; // size: seq_len

    const float* q_ptr = q + (seq_idx * num_q_heads + head_idx) * head_dim;

    // Step 1: Compute Q * K^T through paged block lookups
    float max_score = -1e20f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        int block_idx = seq_block_table[t / block_size];
        int block_offset = t % block_size;

        const float* k_ptr = k_blocks + ((block_idx * block_size + block_offset) * num_kv_heads + kv_head_idx) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_ptr[d] * k_ptr[d];
        }
        dot *= scale;
        s_scores[t] = dot;
        if (dot > max_score) {
            max_score = dot;
        }
    }
    __syncthreads();

    // Step 2: Online Softmax Normalization
    float sum_exp = 0.0f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float e = expf(s_scores[t] - max_score);
        s_scores[t] = e;
        sum_exp += e;
    }
    __syncthreads();

    // Step 3: Weighted sum over Paged V
    float* out_ptr = out + (seq_idx * num_q_heads + head_idx) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            int block_idx = seq_block_table[t / block_size];
            int block_offset = t % block_size;
            const float* v_ptr = v_blocks + ((block_idx * block_size + block_offset) * num_kv_heads + kv_head_idx) * head_dim;
            acc += s_scores[t] * v_ptr[d];
        }
        if (sum_exp > 0.0f) {
            acc /= sum_exp;
        }
        out_ptr[d] = acc;
    }
}
`

// Compile builds the JIT FlashAttention decode module.
func Compile(major, minor int) (*nvrtcx.CompiledModule, error) {
	opts := []nvrtcx.Option{
		nvrtcx.WithName("flash_attn.cu"),
		nvrtcx.WithArch(major, minor),
	}
	mod, err := nvrtcx.Compile(CausalFlashAttentionTemplate, opts...)
	if err != nil {
		return nil, fmt.Errorf("cudaattn: compile: %w", err)
	}
	return mod, nil
}

// CompilePaged builds the JIT Batched Paged Attention decode module.
func CompilePaged(major, minor int) (*nvrtcx.CompiledModule, error) {
	opts := []nvrtcx.Option{
		nvrtcx.WithName("paged_attn.cu"),
		nvrtcx.WithArch(major, minor),
	}
	mod, err := nvrtcx.Compile(BatchedPagedAttentionTemplate, opts...)
	if err != nil {
		return nil, fmt.Errorf("cudaattn: compile paged: %w", err)
	}
	return mod, nil
}
