//go:build linux && arm64

package sampler

import (
	"math"
	"testing"
)

func TestExpPinnedLinux(t *testing.T) {
	for _, tc := range []struct {
		name        string
		input, want uint32
	}{
		{"first-owning-disagreement", 0xb8ca127e, 0x3f7ff9b0},
		{"positive-zero", 0, 0x3f800000},
		{"negative-zero", 0x80000000, 0x3f800000},
		{"negative-domain-end", 0xc2d00000, 0},
		{"negative-infinity", 0xff800000, 0},
		{"positive-infinity", 0x7f800000, 0x7f800000},
		// These are discrete edge controls, not a positive-domain census.
		{"positive-one", 0x3f800000, 0x402df854},
		{"positive-overflow", 0x42b20000, 0x7f800000},
		{"subnormal-one", 0xc2ce0000, 0x00000001},
		{"subnormal-four", 0xc2cc0000, 0x00000004},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := math.Float32bits(Exp(math.Float32frombits(tc.input))); got != tc.want {
				t.Fatalf("input=%08x got=%08x want=%08x", tc.input, got, tc.want)
			}
		})
	}
	x := math.Float32frombits(0xb8ca127e)
	if old := math.Float32bits(float32(math.Exp(float64(x)))); old != 0x3f7ff9af {
		t.Fatalf("old-expression control changed: %08x", old)
	}
	if !math.IsNaN(float64(Exp(float32(math.NaN())))) {
		t.Fatal("NaN input did not return NaN")
	}
}

func TestSamplingUsesPinnedExp(t *testing.T) {
	x := math.Float32frombits(0xb8ca127e)
	minimum := math.Float32frombits(0x3f7ff9b0)
	for _, topK := range []int{0, 2} {
		s := New(2)
		r := RNG{State: 1}
		s.Sample([]float32{x, 0}, Params{Temperature: 1, TopP: 1, MinP: minimum, TopK: topK}, &r)
		index := 0
		if topK != 0 {
			index = 1
		}
		if len(s.probs) != 2 || math.Float32bits(s.probs[index]) != 0x3f7ff9b0 {
			t.Fatalf("topK=%d owning exp not used: %v", topK, s.probs)
		}
	}
	// 513 candidates with top-p=1 force the full-vocabulary materialization
	// fallback after the 512-element heap; both exp sites must retain token0.
	for _, extent := range []int{2, 513} {
		logits := make([]float32, extent)
		logits[0] = x
		probabilities, err := New(extent).Probabilities(logits, Params{Temperature: 1, TopP: 1, MinP: minimum})
		if err != nil || len(probabilities) != extent || probabilities[0] <= 0 {
			t.Fatalf("extent=%d owning support lost: %v %v", extent, probabilities, err)
		}
	}
}
