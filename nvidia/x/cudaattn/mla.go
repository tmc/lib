// SPDX-License-Identifier: Apache-2.0

package cudaattn

// MLAConfig defines dimensional configuration for DeepSeek Multi-Head Latent Attention.
type MLAConfig struct {
	KVLatentDim  int // c_KV dimension (512)
	QHeadDim     int // d_h (128)
	NumQHeads    int // n_h (128)
	RoPEDim      int // d_r (64)
	SoftmaxScale float32
}

// MLACUDATemplate contains the low-rank Multi-Head Latent Attention CUDA decode kernel.
const MLACUDATemplate = `
extern "C" {

__global__ void MLADecodeCUDA(
    const float* __restrict__ absorbed_q,      // [num_q_heads, kv_latent_dim]
    const float* __restrict__ q_rope,          // [num_q_heads, rope_dim]
    const float* __restrict__ kv_latent_cache, // [max_seq_len, kv_latent_dim]
    const float* __restrict__ k_rope_cache,    // [max_seq_len, rope_dim]
    const float* __restrict__ w_uv,            // [kv_latent_dim, num_q_heads * q_head_dim]
    float* __restrict__ out,                   // [num_q_heads, q_head_dim]
    int cur_pos,
    int num_q_heads,
    int kv_latent_dim,
    int rope_dim,
    float scale
) {
    int head_id = blockIdx.x;
    if (head_id >= num_q_heads) return;

    const float* q_abs_ptr = absorbed_q + head_id * kv_latent_dim;
    const float* q_r_ptr = q_rope + head_id * rope_dim;

    extern __shared__ float scores[];
    float max_score = -1e20f;

    // 1. Latent dot product Q_abs · c_KV + Q_rope · K_rope
    for (int t = threadIdx.x; t <= cur_pos; t += blockDim.x) {
        const float* c_kv = kv_latent_cache + t * kv_latent_dim;
        const float* k_r = k_rope_cache + t * rope_dim;

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
    __syncthreads();

    // 2. Softmax normalization
    float sum_exp = 0.0f;
    for (int t = threadIdx.x; t <= cur_pos; t += blockDim.x) {
        float e = expf(scores[t] - max_score);
        scores[t] = e;
        sum_exp += e;
    }
    __syncthreads();

    // 3. Accumulate over compressed latents
    for (int d = threadIdx.x; d < kv_latent_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t <= cur_pos; t++) {
            const float* c_kv = kv_latent_cache + t * kv_latent_dim;
            acc += scores[t] * c_kv[d];
        }
        if (sum_exp > 0.0f) acc /= sum_exp;
    }
}

}
`

// GenerateMLACUDAShader returns the MLA CUDA C++ shader source.
func GenerateMLACUDAShader(cfg MLAConfig) string {
	return MLACUDATemplate
}
