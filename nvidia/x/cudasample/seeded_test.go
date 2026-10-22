// SPDX-License-Identifier: Apache-2.0

package cudasample_test

import (
	"math"
	"sort"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasample"
)

func TestPhilox4x32_10(t *testing.T) {
	// Independent reference vector for Philox-4x32-10 with zero key and zero counter
	// (Salmon et al. SC11, "Parallel Random Numbers: As Easy as 1, 2, 3").
	gotZero := cudasample.Philox4x32_10(0, 0, 0)
	wantZero := [4]uint32{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}
	if gotZero != wantZero {
		t.Fatalf("Philox4x32_10(0, 0, 0) = %#x, want %#x", gotZero, wantZero)
	}

	// Verify determinism and sensitivity to each of (seed, offset, subseq) across >= 1024 cases.
	const numCases = 1024
	seen := make(map[[4]uint32]int, numCases)
	for i := 0; i < numCases; i++ {
		seed := uint64(0x123456789abcdef0) + uint64(i*7)
		offset := uint64(i * 13)
		subseq := uint64(i / 4)
		a := cudasample.Philox4x32_10(seed, offset, subseq)
		b := cudasample.Philox4x32_10(seed, offset, subseq)
		if a != b {
			t.Fatalf("case %d: non-deterministic Philox4x32_10 output %v != %v", i, a, b)
		}
		if prev, dup := seen[a]; dup {
			t.Fatalf("case %d collided with case %d: %#x", i, prev, a)
		}
		seen[a] = i

		if diffSeed := cudasample.Philox4x32_10(seed+1, offset, subseq); diffSeed == a {
			t.Fatalf("case %d: changing seed did not change Philox output", i)
		}
		if diffOffset := cudasample.Philox4x32_10(seed, offset+1, subseq); diffOffset == a {
			t.Fatalf("case %d: changing offset did not change Philox output", i)
		}
		if diffSubseq := cudasample.Philox4x32_10(seed, offset, subseq+1); diffSubseq == a {
			t.Fatalf("case %d: changing subseq did not change Philox output", i)
		}
	}
	if len(seen) < 1000 {
		t.Fatalf("compared %d cases, want >= 1000", len(seen))
	}
}

func TestPhiloxUniformFloat32(t *testing.T) {
	const N = 2048
	var sum float64
	for i := 0; i < N; i++ {
		u := cudasample.PhiloxUniformFloat32(42, uint64(i/16), i%16)
		if !(u > 0 && u < 1) {
			t.Fatalf("draw %d: u = %v out of open interval (0, 1)", i, u)
		}
		uAgain := cudasample.PhiloxUniformFloat32(42, uint64(i/16), i%16)
		if math.Float32bits(u) != math.Float32bits(uAgain) {
			t.Fatalf("draw %d: non-deterministic uniform %v != %v", i, u, uAgain)
		}
		sum += float64(u)
	}
	mean := sum / N
	if math.Abs(mean-0.5) > 0.03 {
		t.Fatalf("mean over %d Philox uniforms = %v, want ~0.5", N, mean)
	}
}

// oracleSumSmallerProbs computes the smaller-probability mass by sorting tokens
// in descending probability order with ascending index tie-breaking.
func oracleSumSmallerProbs(probs []float32) []float32 {
	n := len(probs)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		pi, pj := probs[order[i]], probs[order[j]]
		if pi != pj {
			return pi > pj
		}
		return order[i] < order[j]
	})

	out := make([]float32, n)
	var suffix float64
	for rank := n - 1; rank >= 0; rank-- {
		idx := order[rank]
		out[idx] = float32(suffix)
		suffix += float64(probs[idx])
	}
	return out
}

func TestSumSmallerProbsCPU(t *testing.T) {
	tests := []struct {
		name  string
		probs []float32
	}{
		{"distinct", []float32{0.5, 0.2, 0.3}},
		{"all_ties", []float32{0.25, 0.25, 0.25, 0.25}},
		{"partial_ties", []float32{0.1, 0.3, 0.3, 0.2, 0.1}},
		{"with_zeros", []float32{0.0, 0.7, 0.0, 0.3}},
		{"single", []float32{1.0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := cudasample.SumSmallerProbsCPU(tt.probs)
			if err != nil {
				t.Fatalf("SumSmallerProbsCPU: %v", err)
			}
			want := oracleSumSmallerProbs(tt.probs)
			for i := range want {
				if math.Abs(float64(got[i]-want[i])) > 1e-6 {
					t.Fatalf("index %d: got %v, want %v (full got=%v want=%v)", i, got[i], want[i], got, want)
				}
			}
		})
	}

	if _, err := cudasample.SumSmallerProbsCPU(nil); err == nil {
		t.Fatal("expected error on empty slice")
	}
	if _, err := cudasample.SumSmallerProbsCPU([]float32{0.5, -0.1, 0.6}); err == nil {
		t.Fatal("expected error on negative probability")
	}
	if _, err := cudasample.SumSmallerProbsCPU([]float32{float32(math.NaN()), 1.0}); err == nil {
		t.Fatal("expected error on NaN probability")
	}
}

func TestSampleSeededCPU(t *testing.T) {
	logits := []float32{1.0, 4.0, 2.5, 0.5, 3.0}

	// 1. Greedy argmax when Temperature <= 0.
	tok, probs, err := cudasample.SampleSeededCPU(logits, cudasample.SequenceSampleParams{
		SeqSeed:     99,
		StepOffset:  0,
		Temperature: 0,
	})
	if err != nil {
		t.Fatalf("greedy SampleSeededCPU: %v", err)
	}
	if tok != 1 || probs[1] != 1.0 {
		t.Fatalf("greedy got tok=%d probs=%v, want tok=1 probs[1]=1.0", tok, probs)
	}

	// 2. Bitmask disallowing token 1 (the unmasked argmax) and allowing only tokens 2 and 4.
	mask := []uint32{(1 << 2) | (1 << 4)}
	tokMasked, probsMasked, err := cudasample.SampleSeededCPU(logits, cudasample.SequenceSampleParams{
		SeqSeed:        99,
		StepOffset:     0,
		Temperature:    0,
		AllowedBitmask: mask,
	})
	if err != nil {
		t.Fatalf("masked greedy SampleSeededCPU: %v", err)
	}
	if tokMasked != 4 || probsMasked[1] != 0 {
		t.Fatalf("masked greedy got tok=%d probs=%v, want tok=4 and probs[1]=0", tokMasked, probsMasked)
	}

	// 3. TopK = 2 keeps only tokens 1 (logit 4.0) and 4 (logit 3.0).
	tokTopK, probsTopK, err := cudasample.SampleSeededCPU(logits, cudasample.SequenceSampleParams{
		SeqSeed:     12345,
		StepOffset:  7,
		Temperature: 1.0,
		TopK:        2,
	})
	if err != nil {
		t.Fatalf("TopK SampleSeededCPU: %v", err)
	}
	if tokTopK != 1 && tokTopK != 4 {
		t.Fatalf("TopK=2 sampled disallowed token %d (probs=%v)", tokTopK, probsTopK)
	}
	for i, p := range probsTopK {
		if (i == 1 || i == 4) && p <= 0 {
			t.Fatalf("TopK=2 expected positive prob at %d, got %v", i, p)
		}
		if i != 1 && i != 4 && p != 0 {
			t.Fatalf("TopK=2 expected zero prob at %d, got %v", i, p)
		}
	}

	// 4. TopP filtering via SumSmallerProbsCPU.
	// Construct logits with known softmax probabilities: [0.5, 0.3, 0.2].
	topPLogits := []float32{
		float32(math.Log(0.5)),
		float32(math.Log(0.3)),
		float32(math.Log(0.2)),
	}
	tokTopP, probsTopP, err := cudasample.SampleSeededCPU(topPLogits, cudasample.SequenceSampleParams{
		SeqSeed:     77,
		StepOffset:  3,
		Temperature: 1.0,
		TopP:        0.8,
	})
	if err != nil {
		t.Fatalf("TopP SampleSeededCPU: %v", err)
	}
	if probsTopP[2] != 0 {
		t.Fatalf("TopP=0.8 should filter token 2 (prob 0.2), got probs=%v", probsTopP)
	}
	if tokTopP != 0 && tokTopP != 1 {
		t.Fatalf("TopP=0.8 sampled filtered token %d", tokTopP)
	}

	// 5. MinP filtering: maxProb = 0.5, MinP = 0.7 -> threshold = 0.35 -> only token 0 survives.
	tokMinP, probsMinP, err := cudasample.SampleSeededCPU(topPLogits, cudasample.SequenceSampleParams{
		SeqSeed:     77,
		StepOffset:  3,
		Temperature: 1.0,
		MinP:        0.7,
	})
	if err != nil {
		t.Fatalf("MinP SampleSeededCPU: %v", err)
	}
	if tokMinP != 0 || probsMinP[0] != 1.0 || probsMinP[1] != 0 || probsMinP[2] != 0 {
		t.Fatalf("MinP=0.7 got tok=%d probs=%v, want tok=0 probs=[1, 0, 0]", tokMinP, probsMinP)
	}

	// Error cases.
	if _, _, err := cudasample.SampleSeededCPU(nil, cudasample.SequenceSampleParams{}); err == nil {
		t.Fatal("expected error on empty logits")
	}
	if _, _, err := cudasample.SampleSeededCPU(logits, cudasample.SequenceSampleParams{AllowedBitmask: []uint32{0}}); err == nil {
		t.Fatal("expected error when all tokens are masked out")
	}
}

func TestSampleBatchSeededInvariance(t *testing.T) {
	const numSeqs = 16
	const vocabSize = 64

	batchLogits := make([][]float32, numSeqs)
	batchParams := make([]cudasample.SequenceSampleParams, numSeqs)
	soloTokens := make([]int32, numSeqs)

	for b := 0; b < numSeqs; b++ {
		row := make([]float32, vocabSize)
		for v := 0; v < vocabSize; v++ {
			u := cudasample.PhiloxUniformFloat32(uint64(1000+b), uint64(v), v)
			row[v] = (u - 0.5) * 6.0
		}
		batchLogits[b] = row
		batchParams[b] = cudasample.SequenceSampleParams{
			SeqSeed:     uint64(5000 + b*17),
			StepOffset:  uint64(b * 3),
			Temperature: 0.85,
			TopK:        16,
			TopP:        0.9,
			MinP:        0.05,
		}
		tok, _, err := cudasample.SampleSeededCPU(row, batchParams[b])
		if err != nil {
			t.Fatalf("solo SampleSeededCPU[%d]: %v", b, err)
		}
		soloTokens[b] = tok
	}

	// Full batch in forward order.
	gotBatch, err := cudasample.SampleBatchSeededCPU(batchLogits, batchParams)
	if err != nil {
		t.Fatalf("SampleBatchSeededCPU: %v", err)
	}
	for b := 0; b < numSeqs; b++ {
		if gotBatch[b] != soloTokens[b] {
			t.Fatalf("row %d: batched token %d != solo token %d", b, gotBatch[b], soloTokens[b])
		}
	}

	// Reversed permutation and arbitrary subsets.
	revLogits := make([][]float32, numSeqs)
	revParams := make([]cudasample.SequenceSampleParams, numSeqs)
	for b := 0; b < numSeqs; b++ {
		revLogits[b] = batchLogits[numSeqs-1-b]
		revParams[b] = batchParams[numSeqs-1-b]
	}
	gotRev, err := cudasample.SampleBatchSeededCPU(revLogits, revParams)
	if err != nil {
		t.Fatalf("SampleBatchSeededCPU(reversed): %v", err)
	}
	for b := 0; b < numSeqs; b++ {
		want := soloTokens[numSeqs-1-b]
		if gotRev[b] != want {
			t.Fatalf("reversed row %d: got %d, want %d", b, gotRev[b], want)
		}
	}

	// Subset of [3, 0, 11, 7].
	subIndices := []int{3, 0, 11, 7}
	subLogits := make([][]float32, len(subIndices))
	subParams := make([]cudasample.SequenceSampleParams, len(subIndices))
	for i, idx := range subIndices {
		subLogits[i] = batchLogits[idx]
		subParams[i] = batchParams[idx]
	}
	gotSub, err := cudasample.SampleBatchSeededCPU(subLogits, subParams)
	if err != nil {
		t.Fatalf("SampleBatchSeededCPU(subset): %v", err)
	}
	for i, idx := range subIndices {
		if gotSub[i] != soloTokens[idx] {
			t.Fatalf("subset[%d] (seq %d): got %d, want %d", i, idx, gotSub[i], soloTokens[idx])
		}
	}
}

func TestVerifySpeculativeSeededCPU(t *testing.T) {
	// Step 0: targetProbs[0][1] = 0.8 >= draftProbs[0][1] = 0.5 -> always accepted (ratio >= 1).
	// Step 1: targetProbs[1][2] = 0.0 < draftProbs[1][2] = 0.9 -> always rejected (ratio == 0).
	// Residual at step 1: max(0, [0.0, 0.3, 0.0, 0.7] - [0.0, 0.3, 0.9, 0.0]) = [0.0, 0.0, 0.0, 0.7] -> bonus token must be 3!
	// Step 2: never reached because step 1 was rejected.
	targetProbs := [][]float32{
		{0.1, 0.8, 0.05, 0.05},
		{0.0, 0.3, 0.0, 0.7},
		{0.25, 0.25, 0.25, 0.25},
		{1.0, 0.0, 0.0, 0.0},
	}
	draftProbs := [][]float32{
		{0.2, 0.5, 0.2, 0.1},
		{0.0, 0.3, 0.7, 0.0},
		{0.25, 0.25, 0.25, 0.25},
	}
	draftTokens := []int32{1, 2, 0}

	accepted, bonus, err := cudasample.VerifySpeculativeSeededCPU(targetProbs, draftProbs, draftTokens, 42, 100)
	if err != nil {
		t.Fatalf("VerifySpeculativeSeededCPU: %v", err)
	}
	if len(accepted) != 1 || accepted[0] != 1 {
		t.Fatalf("acceptedTokens = %v, want [1]", accepted)
	}
	if bonus != 3 {
		t.Fatalf("bonusToken = %d, want 3 (only positive residual entry)", bonus)
	}

	// All-accepted case: p_target >= p_draft at every draft step, bonus sampled from targetProbs[3].
	draftTokensAllPass := []int32{1, 1, 0}
	acceptedAll, bonusAll, err := cudasample.VerifySpeculativeSeededCPU(targetProbs, draftProbs, draftTokensAllPass, 42, 100)
	if err != nil {
		t.Fatalf("VerifySpeculativeSeededCPU(all pass): %v", err)
	}
	if len(acceptedAll) != 3 || acceptedAll[0] != 1 || acceptedAll[1] != 1 || acceptedAll[2] != 0 {
		t.Fatalf("acceptedAll = %v, want [1 1 0]", acceptedAll)
	}
	if bonusAll != 0 {
		t.Fatalf("bonusAll = %d, want 0", bonusAll)
	}
}

func TestSeededSampleKernelSource(t *testing.T) {
	if _, err := cudasample.SeededSampleKernelSource(0); err == nil {
		t.Fatal("expected error for vocabSize=0")
	}
	if _, err := cudasample.SeededSampleKernelSource(-16); err == nil {
		t.Fatal("expected error for negative vocabSize")
	}

	src, err := cudasample.SeededSampleKernelSource(129280)
	if err != nil {
		t.Fatalf("SeededSampleKernelSource: %v", err)
	}
	for _, needle := range []string{
		"seeded_batch_invariant_sample_kernel",
		"philox4x32_10",
		"0xD2511F53ull",
		"0xCD9E8D57ull",
		"0x9E3779B9u",
		"0xBB67AE85u",
		"seq_seed",
		"step_offset",
		"129280",
	} {
		if !strings.Contains(src, needle) {
			t.Errorf("SeededSampleKernelSource missing %q", needle)
		}
	}
}
