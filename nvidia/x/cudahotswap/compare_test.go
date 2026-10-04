package cudahotswap

import (
	"math"
	"testing"
)

func TestCompare(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	tol := Tolerance{Abs: 1e-6, Rel: 1e-6}
	for _, tt := range []struct {
		name           string
		got, want      []float32
		tol            Tolerance
		wantOK         bool
		wantMismatches int
		wantNonFinite  int
	}{
		{"identical", []float32{1, 2, 3}, []float32{1, 2, 3}, tol, true, 0, 0},
		{"within tolerance", []float32{1, 2.0000001}, []float32{1, 2}, tol, true, 0, 0},
		{"one element wrong", []float32{1, 5}, []float32{1, 2}, tol, false, 1, 0},
		{"sign flipped", []float32{-1, -2}, []float32{1, 2}, tol, false, 2, 0},
		{"nan fails any tolerance", []float32{nan}, []float32{0}, Tolerance{Abs: 1e30, Rel: 1e30}, false, 1, 1},
		{"inf fails any tolerance", []float32{inf}, []float32{0}, Tolerance{Abs: 1e30, Rel: 1e30}, false, 1, 1},
		{"relative admits large values", []float32{1e6 + 0.5}, []float32{1e6}, Tolerance{Rel: 1e-5}, true, 0, 0},
		{"relative does not admit small ones", []float32{0.5}, []float32{0}, Tolerance{Rel: 1e-5}, false, 1, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c, err := Compare(tt.got, tt.want, tt.tol)
			if err != nil {
				t.Fatal(err)
			}
			if c.OK() != tt.wantOK {
				t.Errorf("OK() = %v, want %v (%+v)", c.OK(), tt.wantOK, c)
			}
			if c.Mismatches != tt.wantMismatches {
				t.Errorf("Mismatches = %d, want %d", c.Mismatches, tt.wantMismatches)
			}
			if c.NonFinite != tt.wantNonFinite {
				t.Errorf("NonFinite = %d, want %d", c.NonFinite, tt.wantNonFinite)
			}
			if !tt.wantOK && c.Worst == nil {
				t.Error("a failed comparison reported no worst element")
			}
		})
	}
}

// An empty comparison passes every tolerance and so proves nothing. Compare
// must refuse it rather than return a passing result over zero elements.
func TestCompareRefusesEmpty(t *testing.T) {
	if _, err := Compare(nil, nil, Tolerance{}); err == nil {
		t.Fatal("Compare admitted an empty comparison")
	}
}

func TestCompareRefusesLengthMismatch(t *testing.T) {
	if _, err := Compare([]float32{1}, []float32{1, 2}, Tolerance{}); err == nil {
		t.Fatal("Compare admitted mismatched lengths")
	}
}

// The worst element is the one needing the largest tolerance increase, not the
// one with the largest absolute error: a large-magnitude element gets a
// proportionally larger allowance and must not win on its magnitude alone.
func TestCompareWorstIsScaledByItsAllowance(t *testing.T) {
	// Element 0: error 1.0 against want 1e6, allowance 1.0 -> ratio 1.
	// Element 1: error 0.01 against want 1, allowance 1e-6 -> ratio 10000.
	got := []float32{1e6 + 1, 1.01}
	want := []float32{1e6, 1}
	c, err := Compare(got, want, Tolerance{Rel: 1e-6})
	if err != nil {
		t.Fatal(err)
	}
	if c.Worst == nil || c.Worst.Index != 1 {
		t.Fatalf("worst = %+v, want index 1", c.Worst)
	}
}
