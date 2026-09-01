//go:build darwin

package metalmtp

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/apple/x/metal/egraph"
)

// MTPConfig configures Multi-Token Prediction (MTP) draft head architectures.
type MTPConfig struct {
	NumDraftHeads int // Number of future tokens to predict simultaneously (e.g. 2, 4)
	Dim           int
	VocabSize     int
}

// GenerateMTPMegakernel emits a single fused MSL compute kernel that calculates
// forward passes for N speculative draft heads in parallel within one threadgroup grid.
func GenerateMTPMegakernel(cfg MTPConfig) string {
	var sb strings.Builder

	sb.WriteString(`// Auto-generated MTP megakernel by apple/x/metal/metalmtp
#include <metal_stdlib>
using namespace metal;

inline float simd_sum(float val) {
    #pragma unroll
    for (ushort offset = 16; offset > 0; offset /= 2) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

kernel void MTPMultiHeadForward(
    device const float* h_trunk         [[buffer(0)]],
    device const float* W_draft_proj    [[buffer(1)]], // [NumDraftHeads, Dim, Dim]
    device const float* W_lm_heads      [[buffer(2)]], // [NumDraftHeads, Dim, VocabSize]
    device float* out_draft_logits      [[buffer(3)]], // [NumDraftHeads, VocabSize]
    constant uint& dim                  [[buffer(4)]],
    constant uint& vocab_size           [[buffer(5)]],
    uint head_idx                       [[threadgroup_position_in_grid]],
    uint tid                            [[thread_position_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]]
) {
    if (head_idx >= uint(`)
	sb.WriteString(fmt.Sprintf("%d", cfg.NumDraftHeads))
	sb.WriteString(`)) return;

    // 1. Point to specific MTP draft head weights in unified memory
    device const float* w_proj = W_draft_proj + head_idx * dim * dim;
    device const float* w_lm   = W_lm_heads   + head_idx * dim * vocab_size;
    device float* head_logits  = out_draft_logits + head_idx * vocab_size;

    threadgroup float s_hidden[2048];

    // 2. Draft Head intermediate projection GEMV
    for (uint row = tid; row < dim; row += 128) {
        float acc = 0.0f;
        device const float* w_row = w_proj + row * dim;
        for (uint k = 0; k < dim; k++) {
            acc += h_trunk[k] * w_row[k];
        }
        s_hidden[row] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Draft Head LM Head projection
    for (uint v = tid; v < vocab_size; v += 128) {
        float acc = 0.0f;
        for (uint k = 0; k < dim; k++) {
            acc += s_hidden[k] * w_lm[k * vocab_size + v];
        }
        head_logits[v] = acc;
    }
}
`)

	return sb.String()
}

// BuildMTPEGraph constructs an equivalence graph representing N speculative draft heads.
func BuildMTPEGraph(numDraftHeads int) (*egraph.EGraph, egraph.Id) {
	eg := egraph.New()
	hTrunk := eg.Add(egraph.NewLeaf("h_trunk"))

	draftHeads := make([]egraph.Id, numDraftHeads)
	for i := 0; i < numDraftHeads; i++ {
		wProj := eg.Add(egraph.NewLeaf(fmt.Sprintf("w_proj_%d", i)))
		wHead := eg.Add(egraph.NewLeaf(fmt.Sprintf("w_head_%d", i)))

		proj := eg.Add(egraph.NewNode("gemv", hTrunk, wProj))
		draftLogits := eg.Add(egraph.NewNode("gemv", proj, wHead))
		draftHeads[i] = draftLogits
	}

	root := eg.Add(egraph.NewNode("fused_mtp_tree", draftHeads...))
	return eg, root
}
