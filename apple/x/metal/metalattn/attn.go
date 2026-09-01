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

// CausalFlashAttentionMSL is the embedded Metal Shading Language compute shader for causal attention decode.
const CausalFlashAttentionMSL = `
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
    uint tid                           [[thread_position_in_threadgroup]]
) {
    if (head_idx >= uint(num_q_heads)) return;

    int kv_group_size = num_q_heads / num_kv_heads;
    int kv_head_idx = head_idx / kv_group_size;

    threadgroup float s_scores[512];
    device const float* q_ptr = q + head_idx * head_dim;

    // Step 1: Compute Q * K^T dot products for tokens 0..cur_pos
    float max_score = -1e20f;
    for (int t = int(tid); t <= cur_pos; t += 128) {
        device const float* k_ptr = k_cache + (t * num_kv_heads + kv_head_idx) * head_dim;
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
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Online Softmax Normalization
    float sum_exp = 0.0f;
    for (int t = int(tid); t <= cur_pos; t += 128) {
        float e = exp(s_scores[t] - max_score);
        s_scores[t] = e;
        sum_exp += e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Weighted sum over V
    device float* out_ptr = out + head_idx * head_dim;
    for (int d = int(tid); d < head_dim; d += 128) {
        float acc = 0.0f;
        for (int t = 0; t <= cur_pos; t++) {
            device const float* v_ptr = v_cache + (t * num_kv_heads + kv_head_idx) * head_dim;
            acc += s_scores[t] * v_ptr[d];
        }
        if (sum_exp > 0.0f) {
            acc /= sum_exp;
        }
        out_ptr[d] = acc;
    }
}
`
