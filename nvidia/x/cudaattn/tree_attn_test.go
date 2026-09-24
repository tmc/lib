// SPDX-License-Identifier: Apache-2.0

package cudaattn_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaattn"
)

func TestTreeAttentionCUDAShader(t *testing.T) {
	src := cudaattn.GenerateTreeAttentionCUDAShader()
	for _, want := range []string{
		"TreeFlashAttentionCUDA",
		"ancestor_mask",
		"TREE_ATTN_MAX_NODES",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("CUDA tree attention shader missing %q", want)
		}
	}
}

// TestTreeAttentionCUDAReduces guards the softmax statistics against the
// per-thread reduction bug: a thread walks a strided slice of the row, so both
// the maximum and the exponent sum must be combined across the block.
func TestTreeAttentionCUDAReduces(t *testing.T) {
	src := cudaattn.GenerateTreeAttentionCUDAShader()
	for _, want := range []string{
		"reduce[threadIdx.x] = thread_max",
		"reduce[threadIdx.x] = thread_sum",
		"float max_score = reduce[0]",
		"float sum_exp = reduce[0]",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("CUDA tree attention shader missing block reduction step %q", want)
		}
	}
	if strings.Contains(src, "1ULL << t") {
		t.Error("CUDA tree attention shader shifts by an unbounded node index")
	}
}

func TestTreeAttentionSharedBytes(t *testing.T) {
	tests := []struct {
		name      string
		numNodes  int
		blockSize int
		want      int
	}{
		{"small tree", 8, 64, 4 * (8 + 64)},
		{"full tree", 64, 128, 4 * (64 + 128)},
		{"clamped to the bitmask width", 512, 32, 4 * (64 + 32)},
	}
	for _, tt := range tests {
		if got := cudaattn.TreeAttentionSharedBytes(tt.numNodes, tt.blockSize); got != tt.want {
			t.Errorf("%s: TreeAttentionSharedBytes(%d, %d) = %d, want %d",
				tt.name, tt.numNodes, tt.blockSize, got, tt.want)
		}
	}
}
