// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"math"
	"strings"
	"testing"
)

func TestCompressedTokenizer(t *testing.T) {
	vocab := []string{
		"Hello",
		"hello",
		"  HELLO  ",
		"Ｈｅｌｌｏ", // fullwidth ASCII
		"world",
		"WORLD",
		" ",
		"\t\n",
	}
	ct := NewCompressedTokenizer(vocab)
	if ct.NumCompressed != 3 {
		t.Fatalf("NumCompressed = %d, want 3 (hello, world, space)", ct.NumCompressed)
	}

	ids, err := ct.CompressIDs([]int32{0, 1, 2, 3, 4, 5, 6, 7})
	if err != nil {
		t.Fatalf("CompressIDs unexpected error: %v", err)
	}
	want := []int32{0, 0, 0, 0, 1, 1, 2, 2}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("CompressIDs[%d] = %d, want %d (all=%v)", i, ids[i], want[i], ids)
		}
	}

	if _, err := ct.CompressIDs([]int32{-1}); err == nil {
		t.Fatalf("expected error on out-of-range negative token ID")
	}
	if _, err := ct.CompressIDs([]int32{99}); err == nil {
		t.Fatalf("expected error on out-of-range token ID")
	}
}

func TestEngramHashIndicesCPU(t *testing.T) {
	mults := ComputeEngramMultipliers(3, 2, 42)
	if len(mults) != 3 {
		t.Fatalf("len(mults) = %d, want 3", len(mults))
	}
	for i, m := range mults {
		if m <= 0 || m%2 == 0 {
			t.Fatalf("multiplier[%d] = %d must be positive and odd", i, m)
		}
	}
	// Different layerID produces distinct multipliers.
	multsOther := ComputeEngramMultipliers(3, 3, 42)
	if mults[0] == multsOther[0] && mults[1] == multsOther[1] {
		t.Fatalf("multipliers did not vary with layerID: %v vs %v", mults, multsOther)
	}

	spec := EngramSpec{
		MaxNgramSize:     3, // 2-gram and 3-gram (2 orders)
		NumHeadsPerNgram: 2,
		EmbedDimPerHead:  16,
		LayerID:          2,
		PadID:            0,
		Seed:             42,
		TablePrimes: [][]int64{
			{101, 103}, // 2-gram heads
			{107, 109}, // 3-gram heads
		},
		Multipliers: mults,
	}

	// Two sequences of length 2 each: [10, 20] and [10, 20].
	// Because cuSeqlens resets at index 2, sequence 1 must produce the exact same
	// n-gram hashes as sequence 0 (proving zero cross-sequence n-gram leakage).
	tokens := []int32{10, 20, 10, 20}
	cuSeqlens := []int32{0, 2, 4}

	hashes, err := EngramHashIndicesCPU(tokens, cuSeqlens, spec)
	if err != nil {
		t.Fatalf("EngramHashIndicesCPU unexpected error: %v", err)
	}
	entriesPerTok := (spec.MaxNgramSize - 1) * spec.NumHeadsPerNgram // 4
	if len(hashes) != len(tokens)*entriesPerTok {
		t.Fatalf("len(hashes) = %d, want %d", len(hashes), len(tokens)*entriesPerTok)
	}

	// Verify sequence 0 (tokens 0, 1) and sequence 1 (tokens 2, 3) produce identical hashes.
	for i := 0; i < 2*entriesPerTok; i++ {
		if hashes[i] != hashes[2*entriesPerTok+i] {
			t.Fatalf("cross-sequence leakage at offset %d: seq0=%d vs seq1=%d", i, hashes[i], hashes[2*entriesPerTok+i])
		}
	}

	// Verify exact formula at token 0 (compressedID=10, prefix padded with PadID=0):
	// k=0: 10 * mults[0], k=1: 0 * mults[1] -> xorMix2 = uint64(10 * mults[0]) ^ 0
	wantMix2 := uint64(int64(10)*mults[0]) ^ uint64(int64(0)*mults[1])
	wantH0 := int64(wantMix2 % 101)
	wantH1 := int64(wantMix2 % 103)
	if hashes[0] != wantH0 || hashes[1] != wantH1 {
		t.Fatalf("token 0 2-gram hashes = (%d, %d), want (%d, %d)", hashes[0], hashes[1], wantH0, wantH1)
	}
}

func TestEngramGateCPU(t *testing.T) {
	query := []float32{3.0, 4.0}
	key := []float32{3.0, 4.0}
	value := []float32{2.0, -4.0}
	rmsQW := []float32{1.0, 1.0}
	rmsKW := []float32{1.0, 1.0}
	eps := float32(1e-6)

	got, err := EngramGateCPU(query, key, value, rmsQW, rmsKW, 1, 2, eps)
	if err != nil {
		t.Fatalf("EngramGateCPU unexpected error: %v", err)
	}

	// Manual oracle:
	// mean(q^2) = (9 + 16)/2 = 12.5
	// q_norm = [3, 4] / sqrt(12.5 + 1e-6), k_norm = same
	// dot = (9 + 16) / (12.5 + 1e-6) ≈ 2.0
	// s = dot / sqrt(2) ≈ sqrt(2)
	// signed_sqrt = +sqrt(sqrt(2) + 1e-6)
	// alpha = 1 / (1 + exp(-signed_sqrt))
	rms := math.Sqrt(12.5 + float64(eps))
	dot := (9.0 + 16.0) / (rms * rms)
	s := dot / math.Sqrt(2.0)
	signedSqrt := math.Sqrt(math.Abs(s) + float64(eps))
	alpha := float32(1.0 / (1.0 + math.Exp(-signedSqrt)))

	want0 := alpha * value[0]
	want1 := alpha * value[1]
	if math.Abs(float64(got[0]-want0)) > 1e-5 || math.Abs(float64(got[1]-want1)) > 1e-5 {
		t.Fatalf("EngramGateCPU = %v, want [%v, %v]", got, want0, want1)
	}
}

func TestFusedEngramGateSource(t *testing.T) {
	src, err := FusedEngramGateSource(128, 1e-6)
	if err != nil {
		t.Fatalf("FusedEngramGateSource unexpected error: %v", err)
	}
	for _, needle := range []string{
		`extern "C" __global__ void engram_gate_kernel`,
		`const int hidden_dim = 128;`,
		`rsqrtf`,
		`sqrtf(abs_s + eps)`,
	} {
		if !strings.Contains(src, needle) {
			t.Fatalf("FusedEngramGateSource missing %q in:\n%s", needle, src)
		}
	}

	if _, err := FusedEngramGateSource(0, 1e-6); err == nil {
		t.Fatalf("expected error for hiddenDim=0")
	}
	if _, err := FusedEngramGateSource(128, 0); err == nil {
		t.Fatalf("expected error for eps=0")
	}
}
