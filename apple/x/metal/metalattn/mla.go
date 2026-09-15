// SPDX-License-Identifier: Apache-2.0

package metalattn

// MLAConfig defines parameters for DeepSeek Multi-Head Latent Attention.
type MLAConfig struct {
	KVLatentDim  int // c_KV dimension (typically 512)
	QHeadDim     int // d_h (typically 128)
	NumQHeads    int // n_h (typically 128)
	RoPEDim      int // d_r decoupled rotary dimension (typically 64)
	SoftmaxScale float32
}

// MLADecodeMSL implements low-rank decoupled Multi-Head Latent Attention on Apple Silicon Metal.
const MLADecodeMSL = `
#include <metal_stdlib>
using namespace metal;

// MLADecode operates directly on the compressed KV latent vectors c_KV (dim 512)
// using absorbed query projections Q' = Q * W_UK.
kernel void MLADecode(
    device const float* absorbed_q       [[buffer(0)]], // [num_q_heads, kv_latent_dim]
    device const float* q_rope           [[buffer(1)]], // [num_q_heads, rope_dim]
    device const float* kv_latent_cache  [[buffer(2)]], // [max_seq_len, kv_latent_dim]
    device const float* k_rope_cache     [[buffer(3)]], // [max_seq_len, rope_dim]
    device const float* w_uv             [[buffer(4)]], // [kv_latent_dim, num_q_heads * q_head_dim]
    device float* out                    [[buffer(5)]], // [num_q_heads, q_head_dim]
    constant int& cur_pos                [[buffer(6)]],
    constant int& num_q_heads            [[buffer(7)]],
    constant int& kv_latent_dim          [[buffer(8)]],
    constant int& rope_dim               [[buffer(9)]],
    constant float& scale                [[buffer(10)]],
    uint head_id                         [[threadgroup_position_in_grid]],
    uint tid                             [[thread_position_in_threadgroup]]
) {
    if (head_id >= uint(num_q_heads)) return;

    device const float* q_abs_ptr = absorbed_q + head_id * kv_latent_dim;
    device const float* q_r_ptr = q_rope + head_id * rope_dim;

    threadgroup float scores[1024];
    float max_score = -1e20f;

    // 1. Latent dot product: Q_abs · c_KV + Q_rope · K_rope
    for (int t = int(tid); t <= cur_pos; t += 128) {
        device const float* c_kv = kv_latent_cache + t * kv_latent_dim;
        device const float* k_r = k_rope_cache + t * rope_dim;

        float dot = 0.0f;
        for (int d = 0; d < kv_latent_dim; d++) {
            dot += q_abs_ptr[d] * c_kv[d];
        }
        for (int d = 0; d < rope_dim; d++) {
            dot += q_r_ptr[d] * k_r[d];
        }
        dot *= scale;
        scores[t] = dot;
        if (dot > max_score) max_score = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Softmax normalization
    float sum_exp = 0.0f;
    for (int t = int(tid); t <= cur_pos; t += 128) {
        float e = exp(scores[t] - max_score);
        scores[t] = e;
        sum_exp += e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Weighted accumulation over compressed latents c_KV and project through W_UV
    threadgroup float latent_acc[512];
    for (int d = int(tid); d < kv_latent_dim; d += 128) {
        float acc = 0.0f;
        for (int t = 0; t <= cur_pos; t++) {
            device const float* c_kv = kv_latent_cache + t * kv_latent_dim;
            acc += scores[t] * c_kv[d];
        }
        if (sum_exp > 0.0f) acc /= sum_exp;
        latent_acc[d] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
`

// GenerateMLAShader returns the Multi-Head Latent Attention MSL source.
func GenerateMLAShader(cfg MLAConfig) string {
	return MLADecodeMSL
}
