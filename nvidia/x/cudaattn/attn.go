package cudaattn

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// ErrInvalidConfig reports an attention shape that the kernels cannot execute.
var ErrInvalidConfig = errors.New("cudaattn: invalid configuration")

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

// Validate checks the shape invariants shared by the dense and paged kernels.
// HeadDim must be even because PagedRoPEStoreKV applies split-half rotary
// embeddings. The kernels otherwise support arbitrary positive head sizes.
func (c Config) Validate() error {
	if c.NumQHeads <= 0 || c.NumKVHeads <= 0 || c.HeadDim <= 0 || c.HeadDim%2 != 0 ||
		c.NumQHeads%c.NumKVHeads != 0 || !(c.Scale > 0) || math.IsNaN(float64(c.Scale)) || math.IsInf(float64(c.Scale), 0) {
		return ErrInvalidConfig
	}
	return nil
}

// PagedSharedMemoryBytes returns the dynamic shared-memory requirement for a
// paged decode block. Threads must be a positive power of two.
func PagedSharedMemoryBytes(maxSeqLen, threads int) (int, error) {
	if maxSeqLen <= 0 || threads <= 0 || threads > 1024 || threads&(threads-1) != 0 {
		return 0, ErrInvalidConfig
	}
	return 4 * (maxSeqLen + threads), nil
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
// status is set to 1 when device metadata names an invalid sequence length or
// physical block. Callers must initialize it to zero and inspect it after the
// dispatch. Invalid rows are filled with NaNs, never plausible output.

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
    float scale,
    int num_seqs,
    int total_blocks,
    int max_seq_len,
    int* __restrict__ status
) {
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
	if (seq_idx >= num_seqs || head_idx >= num_q_heads) return;
	if (blockDim.x == 0 || (blockDim.x & (blockDim.x - 1)) != 0 || block_size <= 0 || max_blocks_per_seq <= 0 || total_blocks <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
		if (threadIdx.x == 0) atomicExch(status, 1);
		return;
	}

    int seq_len = seq_lens[seq_idx];
	float* out_ptr = out + (seq_idx * num_q_heads + head_idx) * head_dim;
	if (seq_len <= 0 || seq_len > max_seq_len || num_q_heads % num_kv_heads != 0) {
		if (threadIdx.x == 0) atomicExch(status, 1);
		for (int d = threadIdx.x; d < head_dim; d += blockDim.x) out_ptr[d] = nanf("");
		return;
	}

    int kv_group_size = num_q_heads / num_kv_heads;
    int kv_head_idx = head_idx / kv_group_size;
    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    extern __shared__ float smem[];
	float* s_scores = smem; // seq_len floats
	float* reduce = s_scores + seq_len; // blockDim.x floats

    const float* q_ptr = q + (seq_idx * num_q_heads + head_idx) * head_dim;

    // Step 1: Compute Q * K^T through paged block lookups
	for (int t = 0; t < seq_len; t++) {
        int block_idx = seq_block_table[t / block_size];
        int block_offset = t % block_size;
		bool valid = block_idx >= 0 && block_idx < total_blocks;
		float dot = 0.0f;
		if (valid) {
			const float* k_ptr = k_blocks + ((block_idx * block_size + block_offset) * num_kv_heads + kv_head_idx) * head_dim;
			for (int d = threadIdx.x; d < head_dim; d += blockDim.x) dot += q_ptr[d] * k_ptr[d];
		}
		reduce[threadIdx.x] = dot;
		__syncthreads();
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			if (!valid) atomicExch(status, 1);
			s_scores[t] = valid ? reduce[0] * scale : nanf("");
		}
		__syncthreads();
    }

	// One thread normalizes the scores. This is intentionally conservative:
	// unlike the old kernel it is mathematically correct for every block size.
	if (threadIdx.x == 0) {
		float max_score = -3.402823466e+38F;
		for (int t = 0; t < seq_len; t++) max_score = fmaxf(max_score, s_scores[t]);
		float sum_exp = 0.0f;
		for (int t = 0; t < seq_len; t++) {
			s_scores[t] = expf(s_scores[t] - max_score);
			sum_exp += s_scores[t];
		}
		for (int t = 0; t < seq_len; t++) s_scores[t] /= sum_exp;
	}
    __syncthreads();

    // Step 3: Weighted sum over Paged V
	for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
		float acc = 0.0f;
		for (int t = 0; t < seq_len; t++) {
			int block_idx = seq_block_table[t / block_size];
			int block_offset = t % block_size;
			if (block_idx < 0 || block_idx >= total_blocks) {
				acc = nanf("");
				break;
			}
			const float* v_ptr = v_blocks + ((block_idx * block_size + block_offset) * num_kv_heads + kv_head_idx) * head_dim;
			acc += s_scores[t] * v_ptr[d];
        }
        out_ptr[d] = acc;
    }
}

extern "C" __global__ void PagedRoPEStoreKV(
	const float* __restrict__ q,
	const float* __restrict__ k,
	const float* __restrict__ v,
	float* __restrict__ q_rope,
	float* __restrict__ k_blocks,
	float* __restrict__ v_blocks,
	const int* __restrict__ block_tables,
	const int* __restrict__ seq_indices,
	const int* __restrict__ positions,
	int num_tokens,
	int max_blocks_per_seq,
	int block_size,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int num_seqs,
	int total_blocks,
	float theta,
	int* __restrict__ status
) {
	int token = blockIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (token >= num_tokens) return;
	int seq = seq_indices[token];
	int pos = positions[token];
	int block_number = pos >= 0 && block_size > 0 ? pos / block_size : -1;
	if (seq < 0 || seq >= num_seqs || pos < 0 || block_size <= 0 || max_blocks_per_seq <= 0 ||
		head_dim <= 0 || (head_dim & 1) != 0 || num_q_heads <= 0 || num_kv_heads <= 0 || theta <= 0.0f ||
		block_number >= max_blocks_per_seq) {
		if (i == 0) atomicExch(status, 1);
		return;
	}
	int slot = block_tables[seq * max_blocks_per_seq + block_number];
	if (slot < 0 || slot >= total_blocks) {
		if (i == 0) atomicExch(status, 1);
		return;
	}
	int half = head_dim / 2;
	int q_dim = num_q_heads * head_dim;
	if (i < q_dim) {
		int d = i % head_dim;
		int pair_d = d < half ? d + half : d - half;
		int freq_d = d < half ? d : pair_d;
		float angle = (float)pos / powf(theta, (2.0f * freq_d) / (float)head_dim);
		float a = q[token * q_dim + i];
		float b = q[token * q_dim + (i - d) + pair_d];
		q_rope[token * q_dim + i] = d < half ? a * cosf(angle) - b * sinf(angle) : b * sinf(angle) + a * cosf(angle);
	}
	int kv_dim = num_kv_heads * head_dim;
	if (i < kv_dim) {
		int d = i % head_dim;
		int pair_d = d < half ? d + half : d - half;
		int freq_d = d < half ? d : pair_d;
		float angle = (float)pos / powf(theta, (2.0f * freq_d) / (float)head_dim);
		float a = k[token * kv_dim + i];
		float b = k[token * kv_dim + (i - d) + pair_d];
		float rotated = d < half ? a * cosf(angle) - b * sinf(angle) : b * sinf(angle) + a * cosf(angle);
		int dst = ((slot * block_size + pos % block_size) * num_kv_heads * head_dim) + i;
		k_blocks[dst] = rotated;
		v_blocks[dst] = v[token * kv_dim + i];
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
