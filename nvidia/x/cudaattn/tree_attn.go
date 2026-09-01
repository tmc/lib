// SPDX-License-Identifier: Apache-2.0

package cudaattn

// TreeAttentionCUDA is the CUDA C++ kernel executing 2D Causal Tree Attention with 64-bit row bitmasks.
const TreeAttentionCUDA = `
extern "C" {

__global__ void TreeFlashAttentionCUDA(
    const float* __restrict__ q,           // [num_nodes, num_heads, head_dim]
    const float* __restrict__ k_cache,     // [num_nodes, num_heads, head_dim]
    const float* __restrict__ v_cache,     // [num_nodes, num_heads, head_dim]
    float* __restrict__ out,               // [num_nodes, num_heads, head_dim]
    const unsigned long long* __restrict__ bitmasks, // [num_nodes]
    int num_nodes,
    int num_heads,
    int head_dim,
    float scale
) {
    int node_id = blockIdx.x;
    int head_id = blockIdx.y;
    if (node_id >= num_nodes || head_id >= num_heads) return;

    unsigned long long ancestor_mask = bitmasks[node_id];
    const float* q_ptr = q + (node_id * num_heads + head_id) * head_dim;

    extern __shared__ float scores[];
    float max_score = -1e20f;

    for (int t = threadIdx.x; t < num_nodes; t += blockDim.x) {
        bool is_ancestor = (ancestor_mask & (1ULL << t)) != 0;
        if (is_ancestor) {
            const float* k_ptr = k_cache + (t * num_heads + head_id) * head_dim;
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
    __syncthreads();

    float sum_exp = 0.0f;
    for (int t = threadIdx.x; t < num_nodes; t += blockDim.x) {
        if ((ancestor_mask & (1ULL << t)) != 0) {
            float e = expf(scores[t] - max_score);
            scores[t] = e;
            sum_exp += e;
        } else {
            scores[t] = 0.0f;
        }
    }
    __syncthreads();

    float* out_ptr = out + (node_id * num_heads + head_id) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < num_nodes; t++) {
            if ((ancestor_mask & (1ULL << t)) != 0) {
                const float* v_ptr = v_cache + (t * num_heads + head_id) * head_dim;
                acc += scores[t] * v_ptr[d];
            }
        }
        if (sum_exp > 0.0f) acc /= sum_exp;
        out_ptr[d] = acc;
    }
}

}
`

// GenerateTreeAttentionCUDAShader returns the TreeFlashAttention CUDA C++ shader source.
func GenerateTreeAttentionCUDAShader() string {
	return TreeAttentionCUDA
}
