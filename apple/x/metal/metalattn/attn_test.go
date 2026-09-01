// SPDX-License-Identifier: Apache-2.0

package metalattn

import (
	"strings"
	"testing"
)

func TestSplitKShader(t *testing.T) {
	cfg := SplitKConfig{NumSplits: 4, BlockSize: 128}
	src := GenerateSplitKShader(cfg)
	if !strings.Contains(src, "SplitKFlashAttentionStage1") || !strings.Contains(src, "SplitKFlashAttentionStage2") {
		t.Errorf("Split-K shader missing expected stage functions")
	}
}

func TestMLAShader(t *testing.T) {
	cfg := MLAConfig{
		KVLatentDim: 512,
		QHeadDim:    128,
		NumQHeads:   128,
		RoPEDim:     64,
	}
	src := GenerateMLAShader(cfg)
	if !strings.Contains(src, "MLADecode") || !strings.Contains(src, "kv_latent_cache") {
		t.Errorf("MLA shader missing expected kernel definitions")
	}
}
