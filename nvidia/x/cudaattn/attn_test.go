package cudaattn_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaattn"
)

func TestCausalFlashAttentionTemplate(t *testing.T) {
	tmpl := cudaattn.CausalFlashAttentionTemplate
	if !strings.Contains(tmpl, "CausalFlashAttentionDecode") {
		t.Error("expected CausalFlashAttentionDecode in template")
	}
	if !strings.Contains(tmpl, "Online Softmax") {
		t.Error("expected Online Softmax in template")
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := cudaattn.DefaultConfig(32, 8, 128)
	if cfg.NumQHeads != 32 || cfg.NumKVHeads != 8 || cfg.HeadDim != 128 {
		t.Errorf("unexpected config: %+v", cfg)
	}

	expectedScale := float32(1.0 / math.Sqrt(128.0))
	if math.Abs(float64(cfg.Scale-expectedScale)) > 1e-6 {
		t.Errorf("expected scale %g, got %g", expectedScale, cfg.Scale)
	}
}
