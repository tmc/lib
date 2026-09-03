package metalmegakernel

import (
	"strings"
	"testing"
)

func TestGenerateFusedFFNMSL(t *testing.T) {
	cfg := FusedFFNConfig{
		HiddenDim:       2048,
		IntermediateDim: 5632,
		RMSNormEpsilon:  1e-5,
	}

	msl, err := GenerateFusedFFNMSL(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(msl, "kernel void MetalFusedFFN") {
		t.Errorf("MSL missing kernel declaration")
	}
	if !strings.Contains(msl, "constexpr uint hidden_dim = 2048;") {
		t.Errorf("MSL missing hidden_dim constant")
	}
	if !strings.Contains(msl, "constexpr uint inter_dim  = 5632;") {
		t.Errorf("MSL missing inter_dim constant")
	}
	if strings.Contains(msl, "w_q") || strings.Contains(msl, "k_cache") {
		t.Error("fused FFN declares unused attention inputs")
	}
	if got, want := cfg.ScratchBytes(), 4*(2048+5632); got != want {
		t.Errorf("ScratchBytes() = %d, want %d", got, want)
	}
}

func TestGenerateFusedFFNMSLRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  FusedFFNConfig
	}{
		{"hidden", FusedFFNConfig{IntermediateDim: 1, RMSNormEpsilon: 1e-5}},
		{"intermediate", FusedFFNConfig{HiddenDim: 1, RMSNormEpsilon: 1e-5}},
		{"epsilon", FusedFFNConfig{HiddenDim: 1, IntermediateDim: 1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := GenerateFusedFFNMSL(test.cfg); err == nil {
				t.Fatal("GenerateFusedFFNMSL succeeded with invalid config")
			}
		})
	}
}
