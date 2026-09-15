// SPDX-License-Identifier: Apache-2.0

package metalmoe

import (
	"context"
	"math"
	"testing"
)

func sum(vals []float32) float32 {
	var s float32
	for _, v := range vals {
		s += v
	}
	return s
}

// TestRouteWeightsAreASoftmax checks the routing weights against a softmax
// computed directly over the selected experts' logits.
func TestRouteWeightsAreASoftmax(t *testing.T) {
	router := NewGroupedTopKRouter(MoEConfig{NumRoutedExperts: 8, TopK: 3})
	logits := []float32{0.1, 2.5, 0.3, 0.4, 5.0, 1.2, 0.8, 0.2}

	got := router.Route(logits)
	if len(got.SelectedExperts) != 3 {
		t.Fatalf("selected %d experts, want 3", len(got.SelectedExperts))
	}
	if want := []int{4, 1, 5}; !equalInts(got.SelectedExperts, want) {
		t.Fatalf("SelectedExperts = %v, want %v", got.SelectedExperts, want)
	}

	var denom float64
	for _, id := range got.SelectedExperts {
		denom += math.Exp(float64(logits[id] - 5.0))
	}
	for i, id := range got.SelectedExperts {
		want := math.Exp(float64(logits[id]-5.0)) / denom
		if math.Abs(float64(got.Weights[i])-want) > 1e-6 {
			t.Errorf("weight for expert %d = %v, want %v", id, got.Weights[i], want)
		}
	}
	if s := sum(got.Weights); math.Abs(float64(s)-1) > 1e-6 {
		t.Errorf("weights sum to %v, want 1", s)
	}
}

// TestRouteBiasSelectsWithoutReweighting is the DeepSeek-V3 load-balancing
// contract: the bias changes which experts are chosen, but the weight an expert
// receives is the softmax of its unbiased logit. Folding the bias into the
// weight would silently rescale every expert's contribution.
func TestRouteBiasSelectsWithoutReweighting(t *testing.T) {
	router := NewGroupedTopKRouter(MoEConfig{NumRoutedExperts: 4, TopK: 2})
	logits := []float32{1.0, 0.9, 0.2, 0.1}

	unbiased := router.Route(logits)
	if want := []int{0, 1}; !equalInts(unbiased.SelectedExperts, want) {
		t.Fatalf("SelectedExperts = %v, want %v", unbiased.SelectedExperts, want)
	}

	// Push expert 2 to the front without touching its logit.
	if err := router.SetBiases([]float32{0, 0, 5, 0}); err != nil {
		t.Fatalf("SetBiases: %v", err)
	}
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
	if s := sum(biased.Weights); math.Abs(float64(s)-1) > 1e-6 {
		t.Errorf("weights sum to %v, want 1", s)
	}

	if err := router.SetBiases([]float32{0, 0}); err == nil {
		t.Error("SetBiases accepted 2 biases for 4 experts, want an error")
	}
}

// TestRouteTiesAreDeterministic guards against sort.Slice, which is not stable:
// equal logits used to route to an arbitrary subset that varied between runs.
func TestRouteTiesAreDeterministic(t *testing.T) {
	router := NewGroupedTopKRouter(MoEConfig{NumRoutedExperts: 16, TopK: 4})
	logits := make([]float32, 16) // every expert scores identically

	first := router.Route(logits)
	if want := []int{0, 1, 2, 3}; !equalInts(first.SelectedExperts, want) {
		t.Fatalf("SelectedExperts = %v, want %v", first.SelectedExperts, want)
	}
	for i := 0; i < 50; i++ {
		got := router.Route(logits)
		if !equalInts(got.SelectedExperts, first.SelectedExperts) {
			t.Fatalf("run %d selected %v, first run selected %v", i, got.SelectedExperts, first.SelectedExperts)
		}
	}
	for i, w := range first.Weights {
		if math.Abs(float64(w)-0.25) > 1e-6 {
			t.Errorf("weight %d = %v, want 0.25 for equal logits", i, w)
		}
	}
}

func TestRouteDegenerateConfigs(t *testing.T) {
	// TopK larger than the expert count clamps rather than panicking.
	r := NewGroupedTopKRouter(MoEConfig{NumRoutedExperts: 2, TopK: 8})
	got := r.Route([]float32{1, 2})
	if len(got.SelectedExperts) != 2 || len(got.Weights) != 2 {
		t.Errorf("selected %d experts and %d weights, want 2 of each",
			len(got.SelectedExperts), len(got.Weights))
	}

	// Fewer logits than configured experts uses what it is given.
	r = NewGroupedTopKRouter(MoEConfig{NumRoutedExperts: 8, TopK: 2})
	got = r.Route([]float32{1})
	if len(got.SelectedExperts) != 1 {
		t.Errorf("selected %v, want a single expert", got.SelectedExperts)
	}

	// TopK of zero yields no experts and no weights.
	r = NewGroupedTopKRouter(MoEConfig{NumRoutedExperts: 4, TopK: 0})
	got = r.Route([]float32{1, 2, 3, 4})
	if len(got.SelectedExperts) != 0 || len(got.Weights) != 0 {
		t.Errorf("TopK=0 selected %v with weights %v, want neither", got.SelectedExperts, got.Weights)
	}
}

// TestExpertCacheEnforcesMaxHot covers the resident limit, which the manager
// stored and never applied: experts accumulated in memory without bound.
func TestExpertCacheEnforcesMaxHot(t *testing.T) {
	mgr, err := NewExpertCacheManager(t.TempDir(), 2)
	if err != nil {
		t.Fatalf("NewExpertCacheManager: %v", err)
	}
	defer mgr.Close()

	for id := 0; id < 5; id++ {
		if err := mgr.RegisterExpert(id, []byte{byte(id)}); err != nil {
			t.Fatalf("RegisterExpert(%d): %v", id, err)
		}
		if got := mgr.HotExperts(); got > 2 {
			t.Fatalf("after registering %d experts, %d are resident, want at most 2", id+1, got)
		}
	}

	// The two most recent registrations stay in memory; the rest spilled and
	// must page back in with their bytes intact.
	if got := mgr.HotExperts(); got != 2 {
		t.Errorf("HotExperts = %d, want 2", got)
	}
	for id := 0; id < 5; id++ {
		data, err := mgr.GetExpert(context.Background(), id)
		if err != nil {
			t.Fatalf("GetExpert(%d): %v", id, err)
		}
		if len(data) != 1 || data[0] != byte(id) {
			t.Errorf("expert %d paged back in as %v, want [%d]", id, data, id)
		}
		if got := mgr.HotExperts(); got > 2 {
			t.Errorf("paging in expert %d left %d resident, want at most 2", id, got)
		}
	}
}

func TestExpertCacheRejectsZeroMaxHot(t *testing.T) {
	if _, err := NewExpertCacheManager(t.TempDir(), 0); err == nil {
		t.Error("NewExpertCacheManager accepted maxHot=0, want an error")
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
