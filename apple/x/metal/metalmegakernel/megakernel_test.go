package metalmegakernel

import (
	"strings"
	"testing"
)

func TestGenerateLayerMegakernelMSL(t *testing.T) {
	cfg := LayerConfig{
		HiddenDim:       2048,
		IntermediateDim: 5632,
		NumQHeads:       32,
		NumKVHeads:      4,
		HeadDim:         64,
		RmsNormEps:      1e-5,
		RopeTheta:       10000.0,
		MaxSeqLen:       2048,
	}

	msl := GenerateLayerMegakernelMSL(cfg)
	if !strings.Contains(msl, "kernel void MetalLayerMegakernel") {
		t.Errorf("MSL missing kernel declaration")
	}
	if !strings.Contains(msl, "constexpr uint hidden_dim = 2048;") {
		t.Errorf("MSL missing hidden_dim constant")
	}
	if !strings.Contains(msl, "constexpr uint inter_dim  = 5632;") {
		t.Errorf("MSL missing inter_dim constant")
	}
}
