package main

import (
	"math"
	"testing"

	"golang.org/x/perf/benchmath"
)

func TestCompareNormal(t *testing.T) {
	base := &sample{
		xs:   []float64{0.801, 0.803, 0.800, 0.805, 0.802},
		reps: []string{"1", "2", "3", "4", "5"},
		vals: []string{"a", "b", "c", "d", "e"},
	}
	next := &sample{
		xs:   []float64{0.806, 0.809, 0.804, 0.812, 0.808},
		reps: base.reps,
		vals: base.vals,
	}
	// The differences are 0.005 0.006 0.004 0.007 0.006: mean 0.0056,
	// sd √1.3e-6, and t(0.975, 4) = 2.7764451051977987.
	hw := 2.7764451051977987 * math.Sqrt(1.3e-6) / math.Sqrt(5)
	otherVals := &sample{xs: next.xs, reps: next.reps, vals: []string{"a", "b", "x", "d", "e"}}
	tests := []struct {
		name       string
		base, s    *sample
		pair       bool
		method     string
		note       string
		delta, lo  float64
		hi         float64
		wantP      func(p float64) bool
		wantSignif bool
	}{
		{"paired", base, next, true, "paired", "", 0.0056, 0.0056 - hw, 0.0056 + hw, func(p float64) bool { return p < 0.001 }, true},
		{"welch", base, next, false, "welch", "", 0.0056, math.NaN(), math.NaN(), func(p float64) bool { return p > 0 && p < 0.05 }, true},
		{"vals", base, otherVals, true, "welch", "unpaired: validation sets differ for replicate 3", 0.0056, math.NaN(), math.NaN(), func(p float64) bool { return p < 0.05 }, true},
		{"point", &sample{xs: []float64{1}}, &sample{xs: []float64{1.5}}, true, "point", "", 0.5, math.NaN(), math.NaN(), math.IsNaN, false},
		{"zero-se", &sample{xs: []float64{1, 1}}, &sample{xs: []float64{2, 2}}, false, "welch", "", 1, 1, 1, func(p float64) bool { return p == 0 }, true},
		{"zero-se-equal", &sample{xs: []float64{1, 1}}, &sample{xs: []float64{1, 1}}, false, "welch", "", 0, 0, 0, func(p float64) bool { return p == 1 }, false},
	}
	st := newStat(0.95, 0.05)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := st.compare(tt.base, tt.s, true, tt.pair, benchmath.AssumeNothing)
			if c.method != tt.method || c.note != tt.note {
				t.Errorf("method, note = %q, %q, want %q, %q", c.method, c.note, tt.method, tt.note)
			}
			if !near(c.delta, tt.delta) {
				t.Errorf("delta = %v, want %v", c.delta, tt.delta)
			}
			if !math.IsNaN(tt.lo) && (!near(c.lo, tt.lo) || !near(c.hi, tt.hi)) {
				t.Errorf("CI = [%v, %v], want [%v, %v]", c.lo, c.hi, tt.lo, tt.hi)
			}
			if !tt.wantP(c.p) {
				t.Errorf("p = %v", c.p)
			}
			if c.significant(0.05) != tt.wantSignif {
				t.Errorf("significant = %v, want %v", !tt.wantSignif, tt.wantSignif)
			}
		})
	}
}

// near reports whether x and y agree to 1e-9: moremath inverts the t
// distribution numerically.
func near(x, y float64) bool { return math.Abs(x-y) < 1e-9 }

func TestSummarizeNormal(t *testing.T) {
	st := newStat(0.95, 0.05)
	s := st.summarize(&sample{xs: []float64{1, 2, 3}}, true, benchmath.AssumeNothing)
	// t(0.975, 2) = 4.302652729911275, sd 1.
	hw := 4.302652729911275 / math.Sqrt(3)
	if s.center != 2 || !near(s.lo, 2-hw) || !near(s.hi, 2+hw) {
		t.Errorf("summary = %v [%v, %v], want 2 [%v, %v]", s.center, s.lo, s.hi, 2-hw, 2+hw)
	}
	s = st.summarize(&sample{xs: []float64{1}}, true, benchmath.AssumeNothing)
	if s.center != 1 || !math.IsNaN(s.lo) {
		t.Errorf("n=1 summary = %v [%v, %v], want 1 [NaN, NaN]", s.center, s.lo, s.hi)
	}
}

func TestInfo(t *testing.T) {
	tests := []struct {
		c    comparison
		want string
	}{
		{comparison{n1: 5, n2: 5, p: 0.0123, method: "paired"}, "p=0.012 n=5 paired"},
		{comparison{n1: 4, n2: 5, p: 0.5, method: "welch"}, "p=0.500 n=4+5 welch"},
		{comparison{n1: 1, n2: 1, method: "point"}, "n=1"},
		{comparison{n1: 1, n2: 5, method: "point"}, "n=1+5 no CI"},
		{comparison{n1: 6, n2: 6, p: 0.002, method: "U"}, "p=0.002 n=6"},
		{comparison{n1: 5, n2: 5, p: 0.0004, method: "paired"}, "p<0.001 n=5 paired"},
	}
	for _, tt := range tests {
		if got := tt.c.info(); got != tt.want {
			t.Errorf("info() = %q, want %q", got, tt.want)
		}
	}
}

func TestFormat(t *testing.T) {
	tests := []struct {
		hw     float64
		v      float64
		signed bool
		want   string
	}{
		{0.0024, 0.8022, false, "0.8022"},
		{0.0014, 0.0056, true, "+0.0056"},
		{0.25, 2.1, false, "2.10"},
		{3, 120, false, "120.0"},
		{300, 1234, false, "1234"},
		{0, 0.8, false, "0.8"},
		{math.Inf(1), 0.8, true, "+0.8"},
		{math.NaN(), 1.5, false, "1.5"},
	}
	for _, tt := range tests {
		if got := format(tt.v, decimals(tt.hw), tt.signed); got != tt.want {
			t.Errorf("format(%v, decimals(%v), %v) = %q, want %q", tt.v, tt.hw, tt.signed, got, tt.want)
		}
	}
}
