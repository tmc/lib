package cudagradcheck

import (
	"fmt"
	"math"
)

// A Tolerance bounds the difference between a computed value and a
// reference value want: the value passes when |got-want| <= Atol + Rtol*|want|,
// as torch.allclose and torch.autograd.gradcheck weigh them.
type Tolerance struct {
	Atol, Rtol float64
}

// Unit roundoffs of the formats GPU kernels compute in. TF32 keeps 10
// explicit mantissa bits, float32 keeps 23.
const (
	Float32Roundoff = 0x1p-24
	TF32Roundoff    = 0x1p-11
)

// Accumulated returns a tolerance for values produced by summing k products
// whose inputs are rounded to unit roundoff u, such as a GEMM with inner
// dimension k in float32 (u = [Float32Roundoff]) or TF32 (u = [TF32Roundoff]).
// scale is the typical magnitude of the values and sets Atol.
//
// The bound is statistical, 4u√k, not the worst case ku: rounding errors of
// independent terms add like a random walk. It suits kernels whose
// reductions sum terms of similar size, not ones that cancel large terms.
func Accumulated(u float64, k int, scale float64) Tolerance {
	e := 4 * u * math.Sqrt(float64(max(k, 1)))
	return Tolerance{Atol: e * scale, Rtol: e}
}

// A Report describes the worst element of a comparison: the one whose
// difference uses the largest fraction of its tolerance.
type Report struct {
	Index     int     // index of the worst element, or -1 if there were none
	Got, Want float64 // the value checked and its reference
	// Ratio is |Got-Want| / (Atol + Rtol*|Want|). The comparison passes
	// when Ratio is at most 1. A NaN on either side, or an infinity where
	// the other side differs, makes it +Inf.
	Ratio float64
}

func (r Report) String() string {
	if r.Index < 0 {
		return "no elements"
	}
	return fmt.Sprintf("element %d: got %.9g, want %.9g (%.3g of tolerance)", r.Index, r.Got, r.Want, r.Ratio)
}

func ratio(got, want float64, tol Tolerance) float64 {
	if got == want {
		return 0 // includes equal infinities
	}
	d := math.Abs(got - want)
	if math.IsNaN(d) || math.IsInf(d, 0) {
		return math.Inf(1)
	}
	bound := tol.Atol + tol.Rtol*math.Abs(want)
	if bound == 0 {
		return math.Inf(1)
	}
	return d / bound
}

// compare returns the report for got against want and an error if the
// worst element fails.
func compare(n int, got, want func(i int) float64, tol Tolerance) (Report, error) {
	r := Report{Index: -1}
	for i := range n {
		g, w := got(i), want(i)
		q := ratio(g, w, tol)
		if r.Index < 0 || q > r.Ratio {
			r = Report{Index: i, Got: g, Want: w, Ratio: q}
		}
	}
	if r.Ratio > 1 {
		return r, fmt.Errorf("mismatch at %v", r)
	}
	return r, nil
}

// Compare checks float32 kernel output got against a float64 reference want
// of the same length. It returns the worst element and an error if any
// element falls outside tol.
func Compare(got []float32, want []float64, tol Tolerance) (Report, error) {
	if len(got) != len(want) {
		return Report{Index: -1}, fmt.Errorf("got %d values, want %d", len(got), len(want))
	}
	return compare(len(got), func(i int) float64 { return float64(got[i]) }, func(i int) float64 { return want[i] }, tol)
}

// GradOptions configures [CheckGrad]. The zero value selects the defaults of
// torch.autograd.gradcheck.
type GradOptions struct {
	Eps float64 // finite-difference step; 0 means 1e-6
	// Tolerance compares the analytic gradient with the numerical one;
	// the zero value means Atol 1e-5, Rtol 1e-3.
	Tolerance
}

// CheckGrad compares analytic, a claimed gradient of f at x, with central
// differences of f computed in float64. It returns the worst element, with
// Want holding the numerical derivative, and an error if any element falls
// outside the tolerance.
//
// f is a host oracle, typically the sum of a forward's outputs weighted by a
// fixed upstream gradient. CheckGrad perturbs x in place, one element at a
// time, and restores it before returning.
func CheckGrad(f func(x []float64) float64, x, analytic []float64, opts GradOptions) (Report, error) {
	if len(x) != len(analytic) {
		return Report{Index: -1}, fmt.Errorf("gradient has %d elements, x has %d", len(analytic), len(x))
	}
	eps := opts.Eps
	if eps == 0 {
		eps = 1e-6
	}
	tol := opts.Tolerance
	if tol == (Tolerance{}) {
		tol = Tolerance{Atol: 1e-5, Rtol: 1e-3}
	}
	numeric := make([]float64, len(x))
	for i, xi := range x {
		x[i] = xi + eps
		hi := f(x)
		x[i] = xi - eps
		lo := f(x)
		x[i] = xi
		numeric[i] = (hi - lo) / (2 * eps)
	}
	return compare(len(x), func(i int) float64 { return analytic[i] }, func(i int) float64 { return numeric[i] }, tol)
}

// Dot returns the sum of a[i]*b[i]. With b a fixed upstream gradient, it
// turns a forward with many outputs into the scalar CheckGrad differentiates:
// the gradient of Dot(forward(x), dy) with respect to x is the backward of dy.
func Dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
