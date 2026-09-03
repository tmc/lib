package cudaattn_test

import (
	"errors"
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

func TestBatchedPagedAttentionTemplate(t *testing.T) {
	tmpl := cudaattn.BatchedPagedAttentionTemplate
	if !strings.Contains(tmpl, "BatchedPagedAttentionDecode") {
		t.Error("expected BatchedPagedAttentionDecode in template")
	}
	if !strings.Contains(tmpl, "block_tables") {
		t.Error("expected block_tables in template")
	}
	for _, name := range []string{"atomicExch(status, 1)", "PagedRoPEStoreKV", "reduce[threadIdx.x]", "total_blocks"} {
		if !strings.Contains(tmpl, name) {
			t.Errorf("template does not contain %q", name)
		}
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

func TestConfigValidate(t *testing.T) {
	tests := []struct {
		name string
		cfg  cudaattn.Config
		ok   bool
	}{
		{"mha", cudaattn.DefaultConfig(8, 8, 64), true},
		{"gqa odd dimension", cudaattn.DefaultConfig(8, 2, 70), true},
		{"uneven groups", cudaattn.DefaultConfig(7, 2, 64), false},
		{"odd dimension", cudaattn.DefaultConfig(8, 2, 63), false},
		{"zero heads", cudaattn.DefaultConfig(0, 0, 64), false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.cfg.Validate()
			if test.ok && err != nil {
				t.Fatalf("Validate: %v", err)
			}
			if !test.ok && !errors.Is(err, cudaattn.ErrInvalidConfig) {
				t.Fatalf("Validate error = %v, want ErrInvalidConfig", err)
			}
		})
	}
}

func TestPagedSharedMemoryBytes(t *testing.T) {
	got, err := cudaattn.PagedSharedMemoryBytes(17, 128)
	if err != nil || got != 4*(17+128) {
		t.Fatalf("PagedSharedMemoryBytes = %d, %v", got, err)
	}
	for _, threads := range []int{0, 3, 2048} {
		if _, err := cudaattn.PagedSharedMemoryBytes(17, threads); !errors.Is(err, cudaattn.ErrInvalidConfig) {
			t.Errorf("threads %d: error = %v", threads, err)
		}
	}
}
