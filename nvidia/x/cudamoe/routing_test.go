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
