package cudacapture_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestComparisonIdentityValidation(t *testing.T) {
	tests := []struct {
		name    string
		id      cudacapture.ComparisonIdentity
		wantErr bool
	}{
		{
			name: "valid complete v1",
			id: cudacapture.ComparisonIdentity{
				Version:      "v1",
				Phase:        "prefill",
				WorkloadName: "llama3-8b",
				WorkloadHash: "sha256:model123",
				TokensHash:   "sha256:tokens123",
				Revision:     "git:abc",
				Device:       "NVIDIA H100",
				ComputeCap:   "sm_90a",
				Complete:     true,
			},
			wantErr: false,
		},
		{
			name: "missing version",
			id: cudacapture.ComparisonIdentity{
				Phase:    "prefill",
				Complete: true,
			},
			wantErr: true,
		},
		{
			name: "incomplete run",
			id: cudacapture.ComparisonIdentity{
				Version:  "v1",
				Phase:    "prefill",
				Complete: false,
			},
			wantErr: true,
		},
		{
			name: "missing phase",
			id: cudacapture.ComparisonIdentity{
				Version:  "v1",
				Complete: true,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.id.Valid()
			if (err != nil) != tt.wantErr {
				t.Errorf("Valid() error = %v, wantErr = %v", err, tt.wantErr)
			}
			if tt.wantErr && !errors.Is(err, cudacapture.ErrIneligibleComparison) {
				t.Errorf("expected ErrIneligibleComparison, got %v", err)
			}
		})
	}
}

func TestComparisonIdentityCompatibility(t *testing.T) {
	base := cudacapture.ComparisonIdentity{
		Version:      "v1",
		Phase:        "prefill",
		WorkloadName: "llama3-8b",
		WorkloadHash: "sha256:model123",
		TokensHash:   "sha256:tokens123",
		Revision:     "git:revA",
		Device:       "NVIDIA H100",
		ComputeCap:   "sm_90a",
		Toolchain:    "cuda 12.6, gcc 13",
		CompileOpts:  "-O3",
		SamplingCfg:  "freq:1000",
		Complete:     true,
	}

	t.Run("exact match", func(t *testing.T) {
		if err := base.Compatible(base); err != nil {
			t.Errorf("expected compatibility with self, got %v", err)
		}
	})

	t.Run("intended revision difference accepted", func(t *testing.T) {
		cand := base
		cand.Revision = "git:revB"
		cand.Toolchain = "cuda 12.6, clang 18"
		cand.CompileOpts = "-O3 -use_fast_math"
		if err := cand.Compatible(base); err != nil {
			t.Errorf("expected compatibility across revision difference, got %v", err)
		}
	})

	t.Run("phase mismatch rejected", func(t *testing.T) {
		cand := base
		cand.Phase = "decode"
		if err := cand.Compatible(base); err == nil {
			t.Error("expected error for phase mismatch, got nil")
		} else if !errors.Is(err, cudacapture.ErrIneligibleComparison) {
			t.Errorf("expected ErrIneligibleComparison, got %v", err)
		}
	})

	t.Run("workload hash mismatch rejected", func(t *testing.T) {
		cand := base
		cand.WorkloadHash = "sha256:model456"
		if err := cand.Compatible(base); err == nil {
			t.Error("expected error for workload hash mismatch, got nil")
		}
	})

	t.Run("tokens hash mismatch rejected", func(t *testing.T) {
		cand := base
		cand.TokensHash = "sha256:different_tokens"
		if err := cand.Compatible(base); err == nil {
			t.Error("expected error for tokens hash mismatch, got nil")
		}
	})

	t.Run("device mismatch rejected", func(t *testing.T) {
		cand := base
		cand.Device = "NVIDIA A100"
		if err := cand.Compatible(base); err == nil {
			t.Error("expected error for device mismatch, got nil")
		}
	})

	t.Run("compute capability mismatch rejected", func(t *testing.T) {
		cand := base
		cand.ComputeCap = "sm_80"
		if err := cand.Compatible(base); err == nil {
			t.Error("expected error for compute capability mismatch, got nil")
		}
	})

	t.Run("sampling config mismatch rejected", func(t *testing.T) {
		cand := base
		cand.SamplingCfg = "freq:500"
		if err := cand.Compatible(base); err == nil {
			t.Error("expected error for sampling config mismatch, got nil")
		}
	})

	t.Run("incomplete candidate rejected", func(t *testing.T) {
		cand := base
		cand.Complete = false
		if err := cand.Compatible(base); err == nil {
			t.Error("expected error for incomplete candidate, got nil")
		}
	})
}

func TestCaptureIdentityRoundTrip(t *testing.T) {
	id := &cudacapture.ComparisonIdentity{
		Version:      "v1",
		Phase:        "decode",
		WorkloadName: "llama3-8b",
		WorkloadHash: "sha256:abc",
		TokensHash:   "sha256:def",
		Revision:     "git:12345",
		Device:       "NVIDIA H100",
		ComputeCap:   "sm_90a",
		Complete:     true,
	}

	cap := &cudacapture.Capture{
		Profile: &profile.Profile{},
		Files: map[string][]byte{
			"kernel.cu":    []byte("// code"),
			"kernel.ptx":   []byte("// ptx"),
			"kernel.cubin": []byte{0x7f, 'E', 'L', 'F'},
			"kernel.sass":  []byte("// sass"),
		},
		Code: cudacapture.CodeFiles{
			Source: "kernel.cu",
			PTX:    "kernel.ptx",
			CUBIN:  "kernel.cubin",
			SASS:   "kernel.sass",
		},
		Identity: id,
	}

	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, cap); err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	decoded, err := cudacapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	if decoded.Identity == nil {
		t.Fatal("expected decoded.Identity to be non-nil")
	}
	if *decoded.Identity != *id {
		t.Errorf("decoded identity = %+v, want %+v", *decoded.Identity, *id)
	}
}
