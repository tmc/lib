// SPDX-License-Identifier: Apache-2.0

package cudamtp_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamtp"
)

func TestGenerateMTPCudaSource(t *testing.T) {
	cfg := cudamtp.MTPConfig{
		NumDraftHeads: 4,
		Dim:           256,
		VocabSize:     1024,
	}

	src := cudamtp.GenerateMTPCudaSource(cfg)
	if !strings.Contains(src, "MTPMultiHeadForwardCUDA") {
		t.Errorf("expected MTPMultiHeadForwardCUDA in generated CUDA source, got:\n%s", src)
	}

	if !strings.Contains(src, "head_idx = blockIdx.y") || !strings.Contains(src, "extern __shared__ float s_hidden[]") {
		t.Errorf("expected shared memory and multi-head block scheduling in generated CUDA source, got:\n%s", src)
	}
}

func TestMTPConfigValidation(t *testing.T) {
	tests := []struct {
		name string
		cfg  cudamtp.MTPConfig
		ok   bool
	}{
		{"valid", cudamtp.MTPConfig{NumDraftHeads: 2, Dim: 128, VocabSize: 256}, true},
		{"zero heads", cudamtp.MTPConfig{NumDraftHeads: 0, Dim: 128, VocabSize: 256}, false},
		{"zero dim", cudamtp.MTPConfig{NumDraftHeads: 2, Dim: 0, VocabSize: 256}, false},
		{"zero vocab", cudamtp.MTPConfig{NumDraftHeads: 2, Dim: 128, VocabSize: 0}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
			if tt.ok && err != nil {
				t.Fatalf("expected valid, got: %v", err)
			}
			if !tt.ok && !errors.Is(err, cudamtp.ErrMTPConfig) {
				t.Fatalf("expected ErrMTPConfig, got: %v", err)
			}
		})
	}
}

func TestMTPReferenceExecution(t *testing.T) {
	cfg := cudamtp.MTPConfig{
		NumDraftHeads: 2,
		Dim:           4,
		VocabSize:     3,
	}

	hTrunk := []float32{1.0, 0.5, -0.5, 2.0}

	// Draft projection: 2 heads * 4 * 4
	// Head 0: Identity projection
	// Head 1: Doubled identity projection
	wDraftProj := make([]float32, 2*4*4)
	for d := 0; d < 4; d++ {
		wDraftProj[0*16+d*4+d] = 1.0 // Head 0 identity
		wDraftProj[1*16+d*4+d] = 2.0 // Head 1 2x identity
	}

	// LM heads: 2 heads * 4 * 3
	wLMHeads := make([]float32, 2*4*3)
	// Fill Head 0 and Head 1 with fixed weights
	for h := 0; h < 2; h++ {
		for d := 0; d < 4; d++ {
			for v := 0; v < 3; v++ {
				wLMHeads[h*12+d*3+v] = float32(d+1) * 0.1 * float32(v+1)
			}
		}
	}

	logits, err := cudamtp.MTPReference(hTrunk, wDraftProj, wLMHeads, cfg)
	if err != nil {
		t.Fatalf("MTPReference failed: %v", err)
	}

	if len(logits) != 2*3 {
		t.Fatalf("logits len = %d, want 6", len(logits))
	}

	// For Head 1, hidden representation is exactly 2x Head 0 hidden representation,
	// so Head 1 logits should be exactly 2x Head 0 logits!
	for v := 0; v < 3; v++ {
		l0 := logits[0*3+v]
		l1 := logits[1*3+v]
		if math.Abs(float64(l1-2.0*l0)) > 1e-5 {
			t.Errorf("vocab %d: Head 1 logit %f != 2x Head 0 logit %f", v, l1, 2.0*l0)
		}
	}
}
