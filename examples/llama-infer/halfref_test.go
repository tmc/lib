package main

import (
	"math"
	"math/rand"
	"testing"
)

// CPU references for the weight-consuming kernels.
//
// Each function computes what its device counterpart must compute, given the
// weights the device will actually hold. The caller passes weights that have
// already been through WeightDType.RoundTrip, so these references model the
// implementation and not the precision: any difference they expose is a
// difference in what the kernel does, not in how the weights were stored.
//
// Every accumulation here is float32, matching the kernels, which accumulate in
// float32 registers in all modes.

func refLinearGEMV(x, w, bias []float32, inDim, outDim int) []float32 {
	out := make([]float32, outDim)
	for row := 0; row < outDim; row++ {
		var sum float32
		for i := 0; i < inDim; i++ {
			sum += x[i] * w[row*inDim+i]
		}
		if bias != nil {
			sum += bias[row]
		}
		out[row] = sum
	}
	return out
}

func refRMSNorm(x, weight []float32, hiddenDim int, eps float32) []float32 {
	var sumSq float32
	for i := 0; i < hiddenDim; i++ {
		sumSq += x[i] * x[i]
	}
	scale := float32(1) / float32(math.Sqrt(float64(sumSq/float32(hiddenDim)+eps)))
	out := make([]float32, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		w := float32(1)
		if weight != nil {
			w = weight[i]
		}
		out[i] = x[i] * scale * w
	}
	return out
}

func refSwiGLUDown(gate, up, wDown []float32, hiddenDim, interDim int) []float32 {
	act := make([]float32, interDim)
	for i := range act {
		g := gate[i]
		act[i] = (g / (1 + float32(math.Exp(float64(-g))))) * up[i]
	}
	out := make([]float32, hiddenDim)
	for row := 0; row < hiddenDim; row++ {
		var sum float32
		for i := 0; i < interDim; i++ {
			sum += act[i] * wDown[row*interDim+i]
		}
		out[row] = sum
	}
	return out
}

// Tolerances, declared here before any comparison is run.
//
// They bound the difference between a half-weight computation and the float32
// computation of the same operation, which is quantization error, not
// implementation error. They are derived from the formats rather than fitted to
// observed output: a weight rounded to F16 carries at most 2**-11 relative
// error and one rounded to BF16 at most 2**-8, and a dot product of n terms
// with independent signs accumulates roughly sqrt(n) of that relative to the
// magnitude of the terms it summed. The absolute floor covers the case where
// the true result is near zero and a relative bound is meaningless.
//
// The relative bound is against the magnitude of the terms the operation
// summed, not against its result. A dot product whose terms cancel can produce
// a small result from large products, and the rounding error it carries is set
// by those products; measuring against the result would report an unbounded
// relative error for a computation that is behaving exactly as the format says
// it must. compareDot takes that scale explicitly.
//
// These are budgets for the equivalence tests further down. The reference tests
// that compare a half computation to the same computation on the same rounded
// weights use no tolerance at all: those must agree exactly.
const (
	tolRelF16  = 4e-3
	tolRelBF16 = 3e-2
	tolAbsF16  = 1e-3
	tolAbsBF16 = 8e-3
)

func tolerancesFor(d WeightDType) (rel, abs float32) {
	if d == WeightF16 {
		return tolRelF16, tolAbsF16
	}
	return tolRelBF16, tolAbsBF16
}

// compareFloats reports the maximum absolute and relative difference between
// got and want.
//
// A NaN or an infinity on either side is a failure rather than something the
// comparison arithmetic absorbs: NaN compares unequal to everything, so a
// tolerance check on a NaN would silently pass or silently fail depending on
// which way it was written.
func compareFloats(t *testing.T, name string, got, want []float32, rel, abs float32) (maxAbs, maxRel float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("%s[%d] is %v", name, i, g)
		}
		if math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatalf("%s[%d] reference is %v", name, i, w)
		}
		d := math.Abs(g - w)
		if d > maxAbs {
			maxAbs = d
		}
		if scale := math.Abs(w); scale > 0 {
			if r := d / scale; r > maxRel {
				maxRel = r
			}
		}
		if d > float64(abs) && (math.Abs(w) == 0 || d/math.Abs(w) > float64(rel)) {
			t.Errorf("%s[%d] = %v, want %v (abs %.3g, rel %.3g)", name, i, g, w, d, d/math.Abs(w))
		}
	}
	return maxAbs, maxRel
}

// dotScale returns, for each output row, the sum of the absolute values of the
// products that row accumulated.
//
// This is the quantity a dot product's rounding error is proportional to, and
// so the denominator its relative bound belongs over.
func dotScale(x, w []float32, inDim, outDim int) []float32 {
	out := make([]float32, outDim)
	for row := 0; row < outDim; row++ {
		var s float32
		for i := 0; i < inDim; i++ {
			s += float32(math.Abs(float64(x[i] * w[row*inDim+i])))
		}
		out[row] = s
	}
	return out
}

// compareDot is compareFloats for a reduction, bounding each difference by the
// magnitude of the terms that reduction summed rather than by its result.
func compareDot(t *testing.T, name string, got, want, scale []float32, rel, abs float32) (maxAbs, maxRel float64) {
	t.Helper()
	if len(got) != len(want) || len(scale) != len(want) {
		t.Fatalf("%s: %d values, %d scales, want %d", name, len(got), len(scale), len(want))
	}
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("%s[%d] is %v", name, i, g)
		}
		if math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatalf("%s[%d] reference is %v", name, i, w)
		}
		s := math.Abs(float64(scale[i]))
		if math.IsNaN(s) || math.IsInf(s, 0) {
			t.Fatalf("%s[%d] scale is %v", name, i, s)
		}
		d := math.Abs(g - w)
		if d > maxAbs {
			maxAbs = d
		}
		if s > 0 {
			if r := d / s; r > maxRel {
				maxRel = r
			}
		}
		if d > float64(abs) && (s == 0 || d/s > float64(rel)) {
			t.Errorf("%s[%d] = %v, want %v (abs %.3g, rel to term magnitude %.3g)", name, i, g, w, d, d/s)
		}
	}
	return maxAbs, maxRel
}

// randomWeights returns deterministic values spanning several magnitudes.
//
// Real weights are not uniform over one decade, and a test that only used one
// would not exercise the part of the half range where the relative error is
// largest. Exact zeros and both signs are included on purpose.
func randomWeights(rng *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		switch i % 16 {
		case 0:
			out[i] = 0
		case 1:
			out[i] = float32(math.Copysign(0, -1))
		case 2:
			out[i] = float32(math.Ldexp(rng.Float64()*2-1, -12)) // small
		case 3:
			out[i] = float32(math.Ldexp(rng.Float64()*2-1, 5)) // large
		default:
			out[i] = float32(rng.NormFloat64() * 0.05)
		}
	}
	return out
}

// TestHalfWeightGEMVWithinTolerance checks that a half-weight projection stays
// inside the declared budget against the float32 result.
//
// Tail dimensions are included because the kernels take a vectorized path for
// whole groups of four and a scalar path for the remainder; a dimension that is
// not a multiple of four exercises both, and one that is exercises only the
// first.
func TestHalfWeightGEMVWithinTolerance(t *testing.T) {
	dims := []struct{ in, out int }{
		{128, 64},  // both divisible by 4
		{129, 64},  // input tail
		{130, 33},  // input tail and odd output count
		{131, 17},  // three-element tail
		{4, 1},     // one vector group only
		{3, 2},     // no vector group at all, scalar path only
		{1024, 12}, // long reduction, where accumulation order matters most
	}
	for _, d := range []WeightDType{WeightF16, WeightBF16} {
		rel, abs := tolerancesFor(d)
		for _, dim := range dims {
			t.Run(d.String()+"/"+itoa(dim.in)+"x"+itoa(dim.out), func(t *testing.T) {
				rng := rand.New(rand.NewSource(int64(dim.in*1000 + dim.out)))
				x := randomWeights(rng, dim.in)
				w := randomWeights(rng, dim.in*dim.out)

				rw := d.RoundTrip(w)
				want := refLinearGEMV(x, w, nil, dim.in, dim.out)
				got := refLinearGEMV(x, rw, nil, dim.in, dim.out)
				maxAbs, maxRel := compareDot(t, "gemv", got, want, dotScale(x, rw, dim.in, dim.out), rel, abs)
				t.Logf("%s %dx%d max abs %.3g max rel %.3g (budget abs %g rel %g)",
					d, dim.in, dim.out, maxAbs, maxRel, abs, rel)
			})
		}
	}
}

// TestHalfWeightRMSNormWithinTolerance covers the norm weights, which are the
// one weight tensor whose values cluster near 1 rather than near 0.
func TestHalfWeightRMSNormWithinTolerance(t *testing.T) {
	for _, d := range []WeightDType{WeightF16, WeightBF16} {
		rel, abs := tolerancesFor(d)
		for _, hidden := range []int{64, 65, 66, 67, 512} {
			t.Run(d.String()+"/"+itoa(hidden), func(t *testing.T) {
				rng := rand.New(rand.NewSource(int64(hidden)))
				x := make([]float32, hidden)
				weight := make([]float32, hidden)
				for i := range x {
					x[i] = float32(rng.NormFloat64())
					weight[i] = 1 + float32(rng.NormFloat64()*0.1)
				}
				want := refRMSNorm(x, weight, hidden, 1e-5)
				got := refRMSNorm(x, d.RoundTrip(weight), hidden, 1e-5)
				maxAbs, maxRel := compareFloats(t, "rmsnorm", got, want, rel, abs)
				t.Logf("%s hidden=%d max abs %.3g max rel %.3g", d, hidden, maxAbs, maxRel)
			})
		}
	}
}

// TestHalfWeightSwiGLUDownWithinTolerance covers the fused down projection,
// whose reduction is over the intermediate dimension and so is the longest in
// the model.
func TestHalfWeightSwiGLUDownWithinTolerance(t *testing.T) {
	for _, d := range []WeightDType{WeightF16, WeightBF16} {
		rel, abs := tolerancesFor(d)
		for _, dims := range []struct{ hidden, inter int }{{32, 128}, {32, 129}, {17, 131}} {
			t.Run(d.String()+"/"+itoa(dims.hidden)+"x"+itoa(dims.inter), func(t *testing.T) {
				rng := rand.New(rand.NewSource(int64(dims.hidden*100 + dims.inter)))
				gate := randomWeights(rng, dims.inter)
				up := randomWeights(rng, dims.inter)
				down := randomWeights(rng, dims.hidden*dims.inter)

				rd := d.RoundTrip(down)
				act := make([]float32, dims.inter)
				for i := range act {
					g := gate[i]
					act[i] = (g / (1 + float32(math.Exp(float64(-g))))) * up[i]
				}
				want := refSwiGLUDown(gate, up, down, dims.hidden, dims.inter)
				got := refSwiGLUDown(gate, up, rd, dims.hidden, dims.inter)
				maxAbs, maxRel := compareDot(t, "swiglu_down", got, want,
					dotScale(act, rd, dims.inter, dims.hidden), rel, abs)
				t.Logf("%s %dx%d max abs %.3g max rel %.3g", d, dims.hidden, dims.inter, maxAbs, maxRel)
			})
		}
	}
}

// TestF32ModeChangesNothing checks that the reference mode is untouched.
//
// WeightF32 must round nothing and change nothing, so a float32 run produces
// exactly the values it produced before half storage existed.
func TestF32ModeChangesNothing(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	w := randomWeights(rng, 4096)
	got := WeightF32.RoundTrip(w)
	for i := range w {
		if math.Float32bits(got[i]) != math.Float32bits(w[i]) {
			t.Fatalf("element %d changed: %v, want %v", i, got[i], w[i])
		}
	}
}

// TestCompareFloatsRejectsNaNAndInf is the negative control for the comparison
// helper.
//
// A tolerance check that quietly accepted a NaN would make every test above
// meaningless, so the helper's refusal is itself tested.
func TestCompareFloatsRejectsNaNAndInf(t *testing.T) {
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		t.Run("value", func(t *testing.T) {
			fake := &testing.T{}
			done := make(chan bool)
			go func() {
				defer func() { recover(); close(done) }()
				compareFloats(fake, "x", []float32{bad}, []float32{1}, 1, 1)
			}()
			<-done
			if !fake.Failed() {
				t.Errorf("compareFloats accepted %v", bad)
			}

			fakeDot := &testing.T{}
			doneDot := make(chan bool)
			go func() {
				defer func() { recover(); close(doneDot) }()
				compareDot(fakeDot, "x", []float32{bad}, []float32{1}, []float32{1}, 1, 1)
			}()
			<-doneDot
			if !fakeDot.Failed() {
				t.Errorf("compareDot accepted %v", bad)
			}
		})
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
