// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"math"
	"testing"
)

// TestRouteTiesAreBrokenByIndex guards against sort.Slice, which is not stable:
// without a tiebreak, equal scores route to an arbitrary subset of the tied
// experts that shifts with the input arrangement and the sort implementation.
func TestRouteTiesAreBrokenByIndex(t *testing.T) {
	router := NewGroupedTopKRouter(MoEConfig{NumRoutedExperts: 16, TopK: 4})

	// Experts 8..15 tie for the lead, so the top 4 must be the four
	// lowest-indexed of them. The losing half ascends, which forces the sort
	// to move the tied block rather than leave it in place.
	logits := make([]float32, 16)
	for i := range logits {
		logits[i] = float32(i)
	}
	for i := 8; i < 16; i++ {
		logits[i] = 100
	}

	got := router.Route(logits)
	if want := []int{8, 9, 10, 11}; !equalInts(got.SelectedExperts, want) {
		t.Errorf("SelectedExperts = %v, want %v", got.SelectedExperts, want)
	}
	for i, w := range got.Weights {
		if math.Abs(float64(w)-0.25) > 1e-6 {
			t.Errorf("weight %d = %v, want 0.25 for tied logits", i, w)
		}
	}
}

// TestRouteBiasSelectsWithoutReweighting is the DeepSeek-V3 load-balancing
// contract: the bias changes which experts are chosen, but the weight an
// expert receives is the softmax of its unbiased logit. Folding the bias into
// the weight would silently rescale every expert's contribution.
func TestRouteBiasSelectsWithoutReweighting(t *testing.T) {
	router := NewGroupedTopKRouter(MoEConfig{NumRoutedExperts: 4, TopK: 2})
	logits := []float32{1.0, 0.9, 0.2, 0.1}

	unbiased := router.Route(logits)
	if want := []int{0, 1}; !equalInts(unbiased.SelectedExperts, want) {
		t.Fatalf("SelectedExperts = %v, want %v", unbiased.SelectedExperts, want)
	}

	// Push expert 2 to the front without touching its logit.
	router.biases = []float32{0, 0, 5, 0}
	biased := router.Route(logits)
	if want := []int{2, 0}; !equalInts(biased.SelectedExperts, want) {
		t.Fatalf("biased SelectedExperts = %v, want %v", biased.SelectedExperts, want)
	}

	// Expert 2 and expert 0 must be weighted by logits 0.2 and 1.0, so expert 0
	// still dominates despite being ranked second.
	ratio := float64(biased.Weights[1] / biased.Weights[0])
	if want := math.Exp(1.0 - 0.2); math.Abs(ratio-want) > 1e-5 {
		t.Errorf("weight ratio expert0/expert2 = %v, want %v (the bias leaked into the weights)", ratio, want)
	}
	var sum float32
	for _, w := range biased.Weights {
		sum += w
	}
	if math.Abs(float64(sum)-1) > 1e-6 {
		t.Errorf("weights sum to %v, want 1", sum)
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
