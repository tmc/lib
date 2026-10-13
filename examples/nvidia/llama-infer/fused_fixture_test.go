package main

import (
	"math"
	"reflect"
	"testing"
)

// The tolerances below are preregistered hypotheses for float32 summation
// reordering between the device kernels and the host primitives in model.go.
// They are not derived bounds, and a failure must not be fixed by widening
// them. Every device value must also be finite. Both values were fixed before
// any device run (FUSION-FIX-DESIGN.md, task record).
const (
	kernelTolerance = 1e-5 // per element, relative to 1+|want|
	logitTolerance  = 1e-4 // end-to-end logits, relative to 1+|want|
)

// parityMismatch reports the first index where got is non-finite or differs
// from want by more than tol*(1+|want|), or -1.
func parityMismatch(got, want []float32, tol float64) int {
	if len(got) != len(want) {
		return 0
	}
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) || !(math.Abs(g-w) <= tol*(1+math.Abs(w))) {
			return i
		}
	}
	return -1
}

// parityValues returns deterministic, asymmetric test values. Index i is zero
// when (i*7+salt)%23 == 11.
func parityValues(n, salt int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((i*7+salt)%23-11) / 31
	}
	return v
}

func addBias(v, bias []float32) []float32 {
	for i := range bias {
		v[i] += bias[i]
	}
	return v
}

// A parityControl is the output a known defective kernel would produce. A
// device result that passes against the correct output must fail against it.
type parityControl struct {
	name string
	out  int // index of the output the defect changes
	want []float32
}

// discriminates reports whether every result within tol of want is more than
// tol from bad: some element differs by more than tol*(2+|want|+|bad|), the
// sum of both comparisons' allowances. A control that only differs from want
// by more than tol*(1+|want|) can still match a correct result that sits at
// the edge of the primary tolerance.
func discriminates(want, bad []float32, tol float64) bool {
	if len(want) != len(bad) {
		return true
	}
	for i := range want {
		w, b := float64(want[i]), float64(bad[i])
		if math.Abs(w-b) > tol*(2+math.Abs(w)+math.Abs(b)) {
			return true
		}
	}
	return false
}

// undiscriminating returns the names of controls a correct result could also
// satisfy.
func undiscriminating(want [][]float32, controls []parityControl) []string {
	var names []string
	for _, c := range controls {
		if !discriminates(want[c.out], c.want, kernelTolerance) {
			names = append(names, c.name)
		}
	}
	return names
}

var qkvParityCases = []struct {
	name            string
	in, qDim, kvDim int
}{
	{"odd rows", 7, 12, 6},
	{"aligned", 8, 8, 8},
	{"odd in and kv", 13, 10, 2},
	{"wide", 64, 64, 32},
}

var qkvBiasModes = []string{"none", "q", "k", "v", "all"}

const qkvParityEps = 1e-5

type qkvFixture struct {
	in, qDim, kvDim              int
	x, g, wq, wk, wv, bq, bk, bv []float32
	want                         [][]float32 // q, k, v
	controls                     []parityControl
}

func newQKVFixture(in, qDim, kvDim int, bias string) qkvFixture {
	f := qkvFixture{in: in, qDim: qDim, kvDim: kvDim}
	f.x, f.g = parityValues(in, 1), parityValues(in, 3)
	f.wq, f.wk, f.wv = parityValues(qDim*in, 5), parityValues(kvDim*in, 9), parityValues(kvDim*in, 13)
	if bias == "q" || bias == "all" {
		f.bq = parityValues(qDim, 17)
	}
	if bias == "k" || bias == "all" {
		f.bk = parityValues(kvDim, 19)
	}
	if bias == "v" || bias == "all" {
		f.bv = parityValues(kvDim, 21)
	}
	n := rmsNorm(f.x, f.g, qkvParityEps)
	f.want = [][]float32{
		addBias(matmulVec(n, f.wq, in, qDim), f.bq),
		addBias(matmulVec(n, f.wk, in, kvDim), f.bk),
		addBias(matmulVec(n, f.wv, in, kvDim), f.bv),
	}

	// Defects of the previous fused kernel.
	noRsqrt := make([]float32, in)
	for i := range noRsqrt {
		noRsqrt[i] = f.x[i] * f.g[i]
	}
	f.controls = append(f.controls, parityControl{"missing rsqrt", 0, addBias(matmulVec(noRsqrt, f.wq, in, qDim), f.bq)})
	if in >= 4 {
		// forward.cu before this change, with xg = x*g and no rsqrt:
		// xg.x*wv.x + xg.y*wv.y + xg.z*wk.w*wv.w per float4 group.
		vterm := make([]float32, kvDim)
		for o := range vterm {
			k, v := f.wk[o*in:], f.wv[o*in:]
			for i := 0; i+3 < in; i += 4 {
				vterm[o] += f.x[i]*f.g[i]*v[i] + f.x[i+1]*f.g[i+1]*v[i+1] + f.x[i+2]*f.g[i+2]*k[i+3]*v[i+3]
			}
		}
		f.controls = append(f.controls, parityControl{"previous value term", 2, addBias(vterm, f.bv)})
		// The value term alone, with correct normalization.
		vonly := make([]float32, kvDim)
		for o := range vonly {
			k, v := f.wk[o*in:], f.wv[o*in:]
			for i := 0; i+3 < in; i += 4 {
				vonly[o] += n[i]*v[i] + n[i+1]*v[i+1] + n[i+2]*k[i+3]*v[i+3]
			}
			for i := in / 4 * 4; i < in; i++ {
				vonly[o] += n[i] * v[i]
			}
		}
		f.controls = append(f.controls, parityControl{"value term alone", 2, addBias(vonly, f.bv)})
	}
	if f.bk != nil {
		f.controls = append(f.controls, parityControl{"dropped key bias", 1, matmulVec(n, f.wk, in, kvDim)})
	}
	return f
}

var swigluParityCases = []struct {
	name          string
	hidden, inter int
}{
	{"odd", 5, 7},
	{"aligned", 8, 12},
	{"odd inter", 13, 9},
	{"wide", 64, 128},
}

// swigluGateSalt makes every tail gate value nonzero for the cases above.
// Salt 1 made gate[8] exactly zero at inter 9, so dropping the tail changed
// nothing and a correct kernel failed the dropped-tail control.
const swigluGateSalt = 2

type swigluFixture struct {
	hidden, inter int
	gate, up, w   []float32
	want          []float32
	controls      []parityControl
}

func newSwiGLUFixture(hidden, inter, gateSalt int) swigluFixture {
	f := swigluFixture{hidden: hidden, inter: inter}
	f.gate, f.up, f.w = parityValues(inter, gateSalt), parityValues(inter, 5), parityValues(hidden*inter, 9)
	act := make([]float32, inter)
	for i := range act {
		act[i] = f.gate[i] / (1 + float32(math.Exp(float64(-f.gate[i])))) * f.up[i]
	}
	f.want = matmulVec(act, f.w, inter, hidden)

	if inter%4 != 0 {
		// The previous kernel dropped the last inter%4 terms.
		short := append([]float32(nil), act...)
		for i := inter / 4 * 4; i < inter; i++ {
			short[i] = 0
		}
		f.controls = append(f.controls, parityControl{"dropped tail", 0, matmulVec(short, f.w, inter, hidden)})
	}
	noSilu := make([]float32, inter)
	for i := range noSilu {
		noSilu[i] = f.gate[i] * f.up[i]
	}
	f.controls = append(f.controls, parityControl{"missing silu", 0, matmulVec(noSilu, f.w, inter, hidden)})
	return f
}

// TestParityControlsDiscriminate checks on the host that every must-fail
// control in the device parity tests is detectable by a correct kernel.
func TestParityControlsDiscriminate(t *testing.T) {
	for _, tt := range qkvParityCases {
		for _, bias := range qkvBiasModes {
			f := newQKVFixture(tt.in, tt.qDim, tt.kvDim, bias)
			n := 1
			if tt.in >= 4 {
				n += 2
			}
			if f.bk != nil {
				n++
			}
			if len(f.controls) != n {
				t.Errorf("qkv %s/bias %s: %d controls, want %d", tt.name, bias, len(f.controls), n)
			}
			if bad := undiscriminating(f.want, f.controls); bad != nil {
				t.Errorf("qkv %s/bias %s: controls %q match the correct result", tt.name, bias, bad)
			}
		}
	}
	for _, tt := range swigluParityCases {
		f := newSwiGLUFixture(tt.hidden, tt.inter, swigluGateSalt)
		n := 1
		if tt.inter%4 != 0 {
			n++
		}
		if len(f.controls) != n {
			t.Errorf("swiglu %s: %d controls, want %d", tt.name, len(f.controls), n)
		}
		if bad := undiscriminating([][]float32{f.want}, f.controls); bad != nil {
			t.Errorf("swiglu %s: controls %q match the correct result", tt.name, bad)
		}
	}

	// The check must reject the fixture it replaced.
	old := newSwiGLUFixture(13, 9, 1)
	if bad := undiscriminating([][]float32{old.want}, old.controls); !reflect.DeepEqual(bad, []string{"dropped tail"}) {
		t.Errorf("gate salt 1, inter 9: undiscriminating controls %q, want [dropped tail]", bad)
	}
}
