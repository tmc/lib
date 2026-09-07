//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

// This reference was observed in the instrumented clone. A production pass
// establishes transfer to Eval; a mismatch does not by itself prove a regression.
const cudaProductionReferenceHash = "e1500b2e69ab4a58fc16c2a37c29d736e7ac1c9075cccee8bb3257c6428ae761"

func TestCUDAProductionHash(t *testing.T) {
	gate := os.Getenv("DS4_CUDA_PRODUCTION_HASH_GATE")
	if gate == "" {
		t.Skip("DS4_CUDA_PRODUCTION_HASH_GATE is not set")
	}
	if gate != "1" {
		t.Fatal("DS4_CUDA_PRODUCTION_HASH_GATE must be 1")
	}
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Fatal("DS4_MODEL is required")
	}
	r, err := newCUDARuntime(path, 1024)
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			if err := r.Close(); err != nil {
				t.Errorf("close runtime: %v", err)
			}
		}
	}()
	for iteration := range 2 {
		if iteration != 0 {
			if err := r.Reset(); err != nil {
				t.Fatalf("reset: %v", err)
			}
		}
		if err := r.Eval(0, 0); err != nil {
			t.Fatalf("iteration %d eval: %v", iteration, err)
		}
		got, err := checkCUDAProductionHash(r.Logits(), cudaProductionReferenceHash)
		t.Logf("production Eval token=0 position=0 context=1024 iteration=%d sha256=%s", iteration, got)
		if err != nil {
			t.Fatalf("iteration %d: %v", iteration, err)
		}
	}
	err = r.Close()
	closed = true
	if err != nil {
		t.Fatalf("close runtime: %v", err)
	}
	t.Log("production Eval reference matched twice; runtime closed; no oracle or throughput claim")
}

func checkCUDAProductionHash(logits []float32, want string) (string, error) {
	if len(logits) != 129280 {
		return "", fmt.Errorf("logits length %d, want 129280", len(logits))
	}
	raw := make([]byte, 4*len(logits))
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return "", fmt.Errorf("non-finite logit %d", i)
		}
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	got := fmt.Sprintf("%x", sha256.Sum256(raw))
	if got != want {
		return got, fmt.Errorf("production hash %s, want %s; reference transfer is unverified", got, want)
	}
	return got, nil
}

func TestCheckCUDAProductionHash(t *testing.T) {
	// SHA-256 of 129280 little-endian positive float32 zeros, computed independently.
	const zeroHash = "6c9f0d98cf37000cade942eb7f01dfebfa50da0f5e078082655513cd940df091"
	for _, tt := range []struct {
		name      string
		length    int
		first     float32
		want      string
		wantError bool
	}{
		{"positive", 129280, 0, zeroHash, false},
		{"wrong hash", 129280, 0, cudaProductionReferenceHash, true},
		{"short", 129279, 0, zeroHash, true},
		{"long", 129281, 0, zeroHash, true},
		{"nan", 129280, float32(math.NaN()), zeroHash, true},
		{"positive infinity", 129280, float32(math.Inf(1)), zeroHash, true},
		{"negative infinity", 129280, float32(math.Inf(-1)), zeroHash, true},
		{"negative zero", 129280, math.Float32frombits(1 << 31), zeroHash, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			logits := make([]float32, tt.length)
			logits[0] = tt.first
			got, err := checkCUDAProductionHash(logits, tt.want)
			if (err != nil) != tt.wantError {
				t.Fatalf("hash = %s, error = %v, want error %t", got, err, tt.wantError)
			}
			if math.IsNaN(float64(tt.first)) || math.IsInf(float64(tt.first), 0) {
				if err == nil || !strings.Contains(err.Error(), "non-finite") {
					t.Fatalf("non-finite input error = %v", err)
				}
			}
			if !tt.wantError && got != tt.want {
				t.Fatalf("hash = %s, want %s", got, tt.want)
			}
		})
	}
}
