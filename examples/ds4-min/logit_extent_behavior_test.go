package main

import (
	"math"
	"testing"
)

// TestNonFiniteExtentDistinguishesShapes fixes the behaviour the whole
// corruption tabulation rests on: that a wholly poisoned vector and a single
// bad element at the front do not report the same thing. The two name
// different bugs -- memory-pressure corruption against a layout or offset
// fault -- and want different fixes.
//
// This exists because the sibling scanner in logit_extent_test.go matches
// message spellings, so it detects a copy of the OLD early-returning check and
// cannot detect the shared walk being weakened underneath it. Every other
// caller of nonFiniteExtent in this package asks only bad(), which needs
// presence and not extent; before this test, making nonFiniteExtent return at
// its first non-finite value broke nothing in the suite. The count and the
// last index are asserted precisely because an early return is what they
// detect.
func TestNonFiniteExtentDistinguishesShapes(t *testing.T) {
	const n = 64
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))

	poison := func(idx ...int) []float32 {
		v := make([]float32, n)
		for _, i := range idx {
			v[i] = nan
		}
		return v
	}
	all := func() []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = inf
		}
		return v
	}

	tests := []struct {
		name  string
		in    []float32
		want  logitExtent
		print string
	}{
		{"clean", make([]float32, n), logitExtent{first: -1, last: -1, n: 0, len: n}, "none"},
		{"single at front", poison(0), logitExtent{first: 0, last: 0, n: 1, len: n}, "0-0/1/64"},
		{"single at back", poison(n - 1), logitExtent{first: n - 1, last: n - 1, n: 1, len: n}, "63-63/1/64"},
		{"short front run", poison(0, 1, 2), logitExtent{first: 0, last: 2, n: 3, len: n}, "0-2/3/64"},
		{"sparse", poison(0, 31, 63), logitExtent{first: 0, last: 63, n: 3, len: n}, "0-63/3/64"},
		{"wholly poisoned", all(), logitExtent{first: 0, last: n - 1, n: n, len: n}, "0-63/64/64"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := nonFiniteExtent(tt.in)
			if got != tt.want {
				t.Errorf("nonFiniteExtent = %+v, want %+v", got, tt.want)
			}
			if s := got.String(); s != tt.print {
				t.Errorf("String = %q, want %q", s, tt.print)
			}
			if got.bad() != (tt.want.n > 0) {
				t.Errorf("bad() = %v, want %v", got.bad(), tt.want.n > 0)
			}
		})
	}

	// The property the tabulation actually reads, stated on its own: these two
	// must not be confusable. An early-returning walk makes them identical.
	front, whole := nonFiniteExtent(poison(0)), nonFiniteExtent(all())
	if front.String() == whole.String() {
		t.Fatalf("a single bad element at the front and a wholly poisoned vector both print %q; "+
			"the extent is the diagnostic and this instrument cannot tell the two apart", front)
	}
	if front.n == whole.n {
		t.Fatalf("both shapes report n=%d", front.n)
	}
}
