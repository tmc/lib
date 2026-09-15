// SPDX-License-Identifier: Apache-2.0

package cudaattn

import (
	"strings"
	"testing"
)

func TestSplitKCUDAShader(t *testing.T) {
	cfg := SplitKConfig{NumSplits: 8, BlockSize: 256}
	src := GenerateSplitKCUDAShader(cfg)
	if !strings.Contains(src, "SplitKFlashAttentionStage1") || !strings.Contains(src, "SplitKFlashAttentionStage2") {
		t.Errorf("CUDA Split-K shader missing expected stage kernels")
	}
}

func TestMLACUDAShader(t *testing.T) {
	cfg := MLAConfig{
		KVLatentDim: 512,
		QHeadDim:    128,
		NumQHeads:   128,
		RoPEDim:     64,
	}
	src := GenerateMLACUDAShader(cfg)
	if !strings.Contains(src, "MLADecodeCUDA") || !strings.Contains(src, "absorbed_q") {
		t.Errorf("CUDA MLA shader missing expected kernel definitions")
	}
}
