// SPDX-License-Identifier: Apache-2.0

package cudaattn

// TreeAttentionMaxNodes is the largest candidate tree a uint64 row bitmask can encode.
const TreeAttentionMaxNodes = 64

// TreeAttentionSharedBytes returns the dynamic shared memory, in bytes, that
// TreeFlashAttentionCUDA needs for a tree of numNodes nodes launched with
// blockSize threads per block. blockSize must be a power of two.
func TreeAttentionSharedBytes(numNodes, blockSize int) int {
	if numNodes > TreeAttentionMaxNodes {
		numNodes = TreeAttentionMaxNodes
	}
	return 4 * (numNodes + blockSize)
}

// TreeAttentionCUDA is the CUDA C++ kernel executing 2D Causal Tree Attention with 64-bit row bitmasks.
//
// Launch it with grid (num_nodes, num_heads), a power-of-two block size, and
// TreeAttentionSharedBytes of dynamic shared memory. Trees wider than
// TreeAttentionMaxNodes have no bitmask representation and are truncated.
const TreeAttentionCUDA = `
extern "C" {

#define TREE_ATTN_MAX_NODES 64

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

    // A row bitmask holds one bit per node, so wider trees are not addressable.
    int n = num_nodes < TREE_ATTN_MAX_NODES ? num_nodes : TREE_ATTN_MAX_NODES;

    unsigned long long ancestor_mask = bitmasks[node_id];
    const float* q_ptr = q + (node_id * num_heads + head_id) * head_dim;

    // Dynamic shared memory holds n scores followed by blockDim.x reduction slots.
    extern __shared__ float smem[];
    float* scores = smem;
    float* reduce = smem + n;

    float thread_max = -INFINITY;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        if ((ancestor_mask >> t) & 1ULL) {
            const float* k_ptr = k_cache + (t * num_heads + head_id) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_ptr[d] * k_ptr[d];
            }
            dot *= scale;
            scores[t] = dot;
            thread_max = fmaxf(thread_max, dot);
        } else {
            scores[t] = -INFINITY;
        }
    }

    // Each thread saw only a strided slice of the row, so the running maximum
    // has to be reduced across the whole block before it can rescale anything.
    reduce[threadIdx.x] = thread_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float max_score = reduce[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        float e = 0.0f;
        if ((ancestor_mask >> t) & 1ULL) {
            e = expf(scores[t] - max_score);
        }
        scores[t] = e;
        thread_sum += e;
    }

    // The softmax denominator likewise spans the block, not one thread's slice.
    reduce[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            reduce[threadIdx.x] += reduce[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_exp = reduce[0];
    __syncthreads();

    // An empty ancestor row emits zeros rather than a division by zero.
    float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;

    float* out_ptr = out + (node_id * num_heads + head_id) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n; t++) {
            if ((ancestor_mask >> t) & 1ULL) {
                const float* v_ptr = v_cache + (t * num_heads + head_id) * head_dim;
                acc += scores[t] * v_ptr[d];
            }
        }
        out_ptr[d] = acc * inv_sum;
    }
}

}
`

// GenerateTreeAttentionCUDAShader returns the TreeFlashAttention CUDA C++ shader source.
func GenerateTreeAttentionCUDAShader() string {
	return TreeAttentionCUDA
}
