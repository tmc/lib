package cudagradcheck

import (
	"math"
	"strings"
	"testing"
)

func TestCompare(t *testing.T) {
	tol := Tolerance{Atol: 1e-3, Rtol: 1e-2}
	tests := []struct {
		name  string
		got   []float32
		want  []float64
		index int
		ok    bool
	}{
		{"equal", []float32{1, 2}, []float64{1, 2}, 0, true},
		{"within rtol", []float32{100, 1}, []float64{100.5, 1}, 0, true},
		{"within atol", []float32{0, 0.0005}, []float64{0, 0}, 1, true},
		{"outside", []float32{1, 2.1}, []float64{1, 2}, 1, false},
		{"nan", []float32{1, float32(math.NaN())}, []float64{1, 1}, 1, false},
		{"both nan", []float32{float32(math.NaN())}, []float64{math.NaN()}, 0, false},
		{"equal inf", []float32{float32(math.Inf(1))}, []float64{math.Inf(1)}, 0, true},
		{"inf", []float32{float32(math.Inf(1))}, []float64{1e30}, 0, false},
		{"empty", nil, nil, -1, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := Compare(tt.got, tt.want, tol)
			if (err == nil) != tt.ok || r.Index != tt.index {
				t.Fatalf("Compare = %v, %v; want index %d, ok %v", r, err, tt.index, tt.ok)
			}
		})
	}
	if _, err := Compare([]float32{1}, nil, tol); err == nil {
		t.Error("Compare accepted mismatched lengths")
	}
}

// TestCheckGradTolerance checks the torch.gradcheck criterion on the
// boundary: an error just inside atol + rtol*|numeric| passes, just outside
// fails.
func TestCheckGradTolerance(t *testing.T) {
	f := func(x []float64) float64 { return 3*x[0] + 0.5*x[1]*x[1] }
	x := []float64{1, 2} // gradient 3, 2
	tests := []struct {
		name     string
		analytic []float64
		ok       bool
		index    int
	}{
		{"exact", []float64{3, 2}, true, 0},
		{"inside", []float64{3 + 0.9*(1e-5+3e-3), 2}, true, 0},
		{"outside", []float64{3, 2 + 1.1*(1e-5+2e-3)}, false, 1},
		{"sign", []float64{-3, 2}, false, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := CheckGrad(f, x, tt.analytic, GradOptions{})
			if (err == nil) != tt.ok || r.Index != tt.index {
				t.Fatalf("CheckGrad = %v, %v; want index %d, ok %v", r, err, tt.index, tt.ok)
			}
			if x[0] != 1 || x[1] != 2 {
				t.Fatalf("x not restored: %v", x)
			}
		})
	}
	_, err := CheckGrad(f, x, []float64{3}, GradOptions{})
	if err == nil || !strings.Contains(err.Error(), "elements") {
		t.Errorf("mismatched length: %v", err)
	}
}

// TestAccumulated checks that the TF32 tolerance accepts TF32 rounding on a
// length-k dot product and the float32 tolerance rejects it. Every term
// rounds the same way, the case where rounding errors do not cancel.
func TestAccumulated(t *testing.T) {
	const k = 256
	var exact, tf32 float64
	for range k {
		exact += 1.1 * 1.1
		tf32 += roundTF32(1.1) * roundTF32(1.1)
	}
	got := []float32{float32(tf32)}
	want := []float64{exact}
	if _, err := Compare(got, want, Accumulated(TF32Roundoff, k, 1)); err != nil {
		t.Errorf("TF32 tolerance: %v", err)
	}
	if _, err := Compare(got, want, Accumulated(Float32Roundoff, k, 1)); err == nil {
		t.Error("float32 tolerance accepted TF32 rounding")
	}
}

// roundTF32 rounds x to the 10 explicit mantissa bits of TF32.
func roundTF32(x float64) float64 {
	b := math.Float32bits(float32(x))
	b = (b + 0xfff + (b>>13)&1) &^ 0x1fff
	return float64(math.Float32frombits(b))
}
