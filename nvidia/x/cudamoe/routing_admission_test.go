// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"math"
	"testing"
	"unsafe"
)

func TestNewGroupedTopKRouterRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  MoEConfig
	}{
		{"zero routed experts", MoEConfig{NumRoutedExperts: 0, TopK: 1}},
		{"negative routed experts", MoEConfig{NumRoutedExperts: -1, TopK: 1}},
		{"zero top-k", MoEConfig{NumRoutedExperts: 4, TopK: 0}},
		{"negative top-k", MoEConfig{NumRoutedExperts: 4, TopK: -1}},
		{"top-k above routed experts", MoEConfig{NumRoutedExperts: 2, TopK: 3}},
		{"negative shared experts", MoEConfig{NumRoutedExperts: 4, TopK: 2, NumSharedExperts: -1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := NewGroupedTopKRouter(tt.cfg)
			if err == nil {
				t.Fatalf("NewGroupedTopKRouter(%+v) = %v, want an error", tt.cfg, r)
			}
		})
	}
	if _, err := NewGroupedTopKRouter(MoEConfig{NumRoutedExperts: 2, TopK: 2}); err != nil {
		t.Errorf("NewGroupedTopKRouter rejected TopK == NumRoutedExperts: %v", err)
	}
}

func TestRouteRejectsBadLogits(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	tests := []struct {
		name   string
		logits []float32
	}{
		{"nil", nil},
		{"short", []float32{1, 2, 3}},
		{"long", []float32{1, 2, 3, 4, 5}},
		{"nan", []float32{1, nan, 3, 4}},
		{"nan unselected", []float32{9, 8, 7, nan}},
		{"+inf", []float32{1, inf, 3, 4}},
		{"-inf", []float32{1, 2, 3, -inf}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newRouter(t, MoEConfig{NumRoutedExperts: 4, TopK: 2})
			if d, err := r.Route(tt.logits); err == nil {
				t.Fatalf("Route(%v) = %+v, want an error", tt.logits, d)
			}
		})
	}
}

func TestRouteRejectsNonfiniteBiases(t *testing.T) {
	for _, b := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		r := newRouter(t, MoEConfig{NumRoutedExperts: 4, TopK: 2})
		if err := r.SetBiases([]float32{0, 0, 0, b}); err != nil {
			t.Fatalf("SetBiases: %v", err)
		}
		if d, err := r.Route([]float32{1, 2, 3, 4}); err == nil {
			t.Errorf("Route with bias %v = %+v, want an error", b, d)
		}
	}
}

// TestRouteWeightsNormalizedAtExtremeScale checks that the softmax is shifted
// by the largest selected logit, not a fixed floor or the first-ranked logit:
// at -1e30 a floor of -1e20 sends every exp to zero, and when a bias ranks a
// small logit first, shifting by it overflows exp for the larger one.
func TestRouteWeightsNormalizedAtExtremeScale(t *testing.T) {
	const big = math.MaxFloat32
	tests := []struct {
		name   string
		topK   int
		logits []float32
		biases []float32 // nil: all zero
		want   []float32 // nil: only the sum is checked
	}{
		{"both -1e30", 2, []float32{-1e30, -1e30}, nil, []float32{0.5, 0.5}},
		{"one -1e30", 1, []float32{-1e30, -1e30}, nil, []float32{1}},
		{"-big", 2, []float32{-big, -big}, nil, []float32{0.5, 0.5}},
		{"+big", 2, []float32{big, big}, nil, []float32{0.5, 0.5}},
		{"span overflows", 2, []float32{big, -big}, nil, []float32{1, 0}},
		{"-1e30 and below", 3, []float32{-1e30, -2e30, -1e30 - 1e24}, nil, nil},
		{"tiny", 2, []float32{-1e-45, 1e-45}, nil, []float32{0.5, 0.5}},
		{"bias reorders", 2, []float32{0, 100}, []float32{1000, 0}, []float32{0, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newRouter(t, MoEConfig{NumRoutedExperts: len(tt.logits), TopK: tt.topK})
			if tt.biases != nil {
				if err := r.SetBiases(tt.biases); err != nil {
					t.Fatalf("SetBiases: %v", err)
				}
			}
			d := route(t, r, tt.logits)
			var sum float64
			for _, w := range d.Weights {
				if !(w >= 0 && w <= 1) {
					t.Errorf("weight %v outside [0, 1]", w)
				}
				sum += float64(w)
			}
			if math.Abs(sum-1) > 1e-6 {
				t.Errorf("weights %v sum to %v, want 1", d.Weights, sum)
			}
			for i, w := range tt.want {
				if math.Abs(float64(d.Weights[i]-w)) > 1e-6 {
					t.Errorf("Weights = %v, want %v", d.Weights, tt.want)
					break
				}
			}
		})
	}
}

// TestRouteZeroRouterReturnsError checks that a GroupedTopKRouter not made by
// NewGroupedTopKRouter reports an error instead of panicking.
func TestRouteZeroRouterReturnsError(t *testing.T) {
	var r GroupedTopKRouter
	if d, err := r.Route(nil); err == nil {
		t.Fatalf("zero GroupedTopKRouter.Route(nil) = %+v, want an error", d)
	}
}

// TestRouteOrdinaryWeightsUnchanged pins the exact bits produced before Route
// gained input checks, so the checks cannot perturb ordinary routing.
func TestRouteOrdinaryWeightsUnchanged(t *testing.T) {
	r := newRouter(t, MoEConfig{NumRoutedExperts: 8, TopK: 3, NumSharedExperts: 1})
	if err := r.SetBiases([]float32{0, 0.5, 0, -0.25, 0, 0, 1.5, 0}); err != nil {
		t.Fatalf("SetBiases: %v", err)
	}
	d := route(t, r, []float32{0.1, 2.5, 0.3, 0.4, 5.0, 1.2, 0.8, -3.75})
	if want := []int{4, 1, 6}; !equalInts(d.SelectedExperts, want) {
		t.Fatalf("SelectedExperts = %v, want %v", d.SelectedExperts, want)
	}
	want := []uint32{0x3f6958b9, 0x3d993be9, 0x3c5ff25b}
	for i, w := range d.Weights {
		if got := math.Float32bits(w); got != want[i] {
			t.Errorf("weight %d bits = %#08x, want %#08x", i, got, want[i])
		}
	}
	if !d.HasShared {
		t.Error("HasShared = false, want true")
	}
}

// TestNewGroupedTopKRouterRejectsUnrepresentableSize checks that sizes whose
// routing slices make cannot represent are refused before any allocation.
// Every case is refused, so none allocates.
func TestNewGroupedTopKRouterRejectsUnrepresentableSize(t *testing.T) {
	tests := []struct {
		name string
		cfg  MoEConfig
	}{
		{"MaxInt experts", MoEConfig{NumRoutedExperts: math.MaxInt, TopK: 1}},
		{"MaxInt experts and top-k", MoEConfig{NumRoutedExperts: math.MaxInt, TopK: math.MaxInt}},
		{"MaxInt/4+1 experts", MoEConfig{NumRoutedExperts: math.MaxInt/4 + 1, TopK: 1}},
		{"MaxInt/16+1 experts", MoEConfig{NumRoutedExperts: math.MaxInt/16 + 1, TopK: 1}},
		{"MaxInt32 experts", MoEConfig{NumRoutedExperts: math.MaxInt32, TopK: 1}},
		{"first refused expert count", MoEConfig{NumRoutedExperts: scoreLimit() + 1, TopK: 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewGroupedTopKRouter(tt.cfg); err == nil {
				t.Fatalf("NewGroupedTopKRouter(%+v) succeeded, want an error", tt.cfg)
			}
		})
	}
	r := newRouter(t, MoEConfig{NumRoutedExperts: 256, TopK: 8, NumSharedExperts: 1})
	logits := make([]float32, 256)
	logits[7] = 1
	if d := route(t, r, logits); d.SelectedExperts[0] != 7 {
		t.Errorf("SelectedExperts = %v, want 7 first", d.SelectedExperts)
	}
}

// scoreLimit returns the largest expert count whose expert scores, the
// router's widest per-expert slice, fit in maxRouterAlloc bytes.
func scoreLimit() int {
	return maxRouterAlloc / int(unsafe.Sizeof(expertScore{}))
}

// TestCheckRouterAllocBoundary checks the bound on each routing slice at its
// exact edge. checkRouterAlloc only computes, so the accepted edges allocate
// nothing.
func TestCheckRouterAllocBoundary(t *testing.T) {
	biases := maxRouterAlloc / int(unsafe.Sizeof(float32(0)))
	selected := maxRouterAlloc / int(unsafe.Sizeof(int(0)))
	tests := []struct {
		name string
		n, k int
		ok   bool
	}{
		{"ordinary", 256, 8, true},
		{"scores at limit", scoreLimit(), 1, true},
		{"scores past limit", scoreLimit() + 1, 1, false},
		{"biases past limit", biases + 1, 1, false},
		{"selected at limit", 1, selected, true},
		{"selected past limit", 1, selected + 1, false},
		{"MaxInt", math.MaxInt, 1, false},
		{"MaxInt top-k", 1, math.MaxInt, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkRouterAlloc(tt.n, tt.k)
			if (err == nil) != tt.ok {
				t.Errorf("checkRouterAlloc(%d, %d) = %v, want ok %v", tt.n, tt.k, err, tt.ok)
			}
		})
	}
}
