package cudamtp

import (
	"strings"
	"testing"
)

func TestGenerateMTPCudaSource(t *testing.T) {
	cfg := MTPConfig{
		NumDraftHeads: 4,
		Dim:           256,
		VocabSize:     1024,
	}

	src := GenerateMTPCudaSource(cfg)
	if !strings.Contains(src, "MTPMultiHeadForwardCUDA") {
		t.Errorf("expected MTPMultiHeadForwardCUDA in generated CUDA source, got:\n%s", src)
	}

	if !strings.Contains(src, "head_idx = blockIdx.y") || !strings.Contains(src, "extern __shared__ float s_hidden[]") {
		t.Errorf("expected shared memory and multi-head block scheduling in generated CUDA source, got:\n%s", src)
	}

	t.Logf("Generated CUDA MTP Megakernel source (%d bytes)", len(src))
}
