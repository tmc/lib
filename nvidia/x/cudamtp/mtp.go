package cudamtp

import (
	"fmt"
	"strings"
)

// MTPConfig configures Multi-Token Prediction (MTP) speculative draft heads for CUDA execution.
type MTPConfig struct {
	NumDraftHeads int // Number of speculative future tokens to predict in parallel (e.g. 2, 4)
	Dim           int
	VocabSize     int
}

// GenerateMTPCudaSource generates a fused CUDA C++ kernel that computes N speculative draft heads
// simultaneously within a single grid launch across cooperative thread blocks.
func GenerateMTPCudaSource(cfg MTPConfig) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf(`// Auto-generated CUDA MTP megakernel by nvidia/x/cudamtp
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C" __global__ void MTPMultiHeadForwardCUDA(
    const float* __restrict__ h_trunk,
    const float* __restrict__ W_draft_proj, // [NumDraftHeads, Dim, Dim]
    const float* __restrict__ W_lm_heads,   // [NumDraftHeads, Dim, VocabSize]
    float* __restrict__ out_draft_logits,   // [NumDraftHeads, VocabSize]
    int dim,
    int vocab_size
) {
    int head_idx = blockIdx.y;
    if (head_idx >= %d) return;

    const float* w_proj = W_draft_proj + head_idx * dim * dim;
    const float* w_lm   = W_lm_heads   + head_idx * dim * vocab_size;
    float* head_logits  = out_draft_logits + head_idx * vocab_size;

    extern __shared__ float s_hidden[];

    // 1. Parallel draft head projection
    for (int row = threadIdx.x; row < dim; row += blockDim.x) {
        float acc = 0.0f;
        const float* w_row = w_proj + row * dim;
        for (int k = 0; k < dim; k++) {
            acc += h_trunk[k] * w_row[k];
        }
        s_hidden[row] = acc;
    }
    __syncthreads();

    // 2. Parallel draft LM Head logits projection
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < vocab_size) {
        float acc = 0.0f;
        for (int k = 0; k < dim; k++) {
            acc += s_hidden[k] * w_lm[k * vocab_size + v];
        }
        head_logits[v] = acc;
    }
}
`, cfg.NumDraftHeads))

	return sb.String()
}
