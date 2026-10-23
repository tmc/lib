package cublasltx_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/x/cublasltx"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// geluAct is the exact (erf) GELU. cuBLASLt may use the tanh approximation,
// which differs from it by less than 5e-4 everywhere, so the tolerance below
// covers either while still rejecting a missing activation.
func geluAct(x float64) float64 { return float64(gelu(float32(x))) }

// epilogueRef computes D = f(A*B + bias) for column-major operands, where A is
// m by k with leading dimension m, B is k by n with leading dimension k and D
// is m by n with leading dimension m. bias holds one value per row of D.
func epilogueRef(a, b, bias []float32, m, k, n int, f func(float64) float64) []float32 {
	d := make([]float32, m*n)
	for j := range n {
		for i := range m {
			sum := 0.0
			for l := range k {
				sum += float64(a[l*m+i]) * float64(b[j*k+l])
			}
			if bias != nil {
				sum += float64(bias[i])
			}
			if f != nil {
				sum = f(sum)
			}
			d[j*m+i] = float32(sum)
		}
	}
	return d
}

// epilogueFixture is one column-major problem and its device buffers.
type epilogueFixture struct {
	dev                 *cuda.Device
	m, k, n             int
	hostA, hostB, hostC []float32
	a, b, bias, d       cudamem.Slice[float32]
	stream              *cuda.Stream
	engine              *cublasltx.Engine
	owner               *testOwner
}

func newEpilogueFixture(t *testing.T) *epilogueFixture {
	t.Helper()
	dev, owner := requireDevice(t)
	stream, err := dev.NewStream()
	owner.stream(stream)
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	engine, err := cublasltx.NewEngine(dev)
	owner.add(engine.Close)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	const m, k, n = 3, 4, 2
	f := &epilogueFixture{dev: dev, m: m, k: k, n: n, stream: stream, engine: engine, owner: owner}
	f.hostA = make([]float32, m*k)
	for i := range f.hostA {
		f.hostA[i] = float32(i%5)*0.25 - 0.5
	}
	f.hostB = make([]float32, k*n)
	for i := range f.hostB {
		f.hostB[i] = float32(i%7)*0.5 - 1
	}
	f.hostC = []float32{-1.5, 0.25, 2}

	for _, e := range []struct {
		dst *cudamem.Slice[float32]
		src []float32
	}{{&f.a, f.hostA}, {&f.b, f.hostB}, {&f.bias, f.hostC}, {&f.d, make([]float32, m*n)}} {
		s, err := cudamem.Alloc[float32](dev, len(e.src))
		owner.add(s.Free)
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		if err := s.CopyFromHost(e.src); err != nil {
			t.Fatalf("upload: %v", err)
		}
		*e.dst = s
	}
	return f
}

// run executes cfg over the fixture and returns D.
func (f *epilogueFixture) run(t *testing.T, cfg cublasltx.MatMulConfig) []float32 {
	t.Helper()
	if err := f.engine.MatMulRaw(f.stream, cfg,
		f.a.Ptr(), f.m, f.k, f.m,
		f.b.Ptr(), f.k, f.n, f.k,
		f.d.Ptr(), f.m, f.n, f.m); err != nil {
		t.Fatalf("MatMulRaw: %v", err)
	}
	if err := f.stream.Synchronize(nil); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	got := make([]float32, f.m*f.n)
	if err := f.d.CopyToHost(got); err != nil {
		t.Fatalf("download: %v", err)
	}
	return got
}

func TestMatMulEpilogue(t *testing.T) {
	f := newEpilogueFixture(t)
	for _, tt := range []struct {
		name     string
		epilogue cublaslt.Epilogue
		act      func(float64) float64
	}{
		{"none", 0, nil},
		{"bias", cublaslt.EpilogueBias, nil},
		{"relu_bias", cublaslt.EpilogueReluBias, func(x float64) float64 { return math.Max(x, 0) }},
		{"gelu_bias", cublaslt.EpilogueGeluBias, geluAct},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := cublasltx.MatMulConfig{Alpha: 1, Epilogue: tt.epilogue}
			var bias []float32
			if tt.epilogue != 0 {
				cfg.Bias = f.bias.Ptr()
				bias = f.hostC
			}
			got := f.run(t, cfg)
			want := epilogueRef(f.hostA, f.hostB, bias, f.m, f.k, f.n, tt.act)
			for i := range want {
				if math.Abs(float64(got[i]-want[i])) > 2e-3 {
					t.Fatalf("D = %v, want %v", got, want)
				}
			}
		})
	}
}

func TestMatMulEpilogueMissingBias(t *testing.T) {
	f := newEpilogueFixture(t)
	for _, ep := range []cublaslt.Epilogue{cublaslt.EpilogueBias, cublaslt.EpilogueGeluBias} {
		err := f.engine.MatMulRaw(f.stream, cublasltx.MatMulConfig{Alpha: 1, Epilogue: ep},
			f.a.Ptr(), f.m, f.k, f.m,
			f.b.Ptr(), f.k, f.n, f.k,
			f.d.Ptr(), f.m, f.n, f.m)
		if err == nil || !strings.Contains(err.Error(), "bias pointer") {
			t.Fatalf("epilogue %v with no bias: err = %v, want a bias pointer error", ep, err)
		}
	}
}

// TestMatMulEpilogueMissingAux checks that an aux epilogue without an Aux
// pointer is rejected here rather than handed to cuBLASLt. The bias pointer is
// supplied so the only thing missing is Aux.
func TestMatMulEpilogueMissingAux(t *testing.T) {
	f := newEpilogueFixture(t)
	for _, ep := range []cublaslt.Epilogue{cublaslt.EpilogueGeluAux, cublaslt.EpilogueGeluAuxBias} {
		err := f.engine.MatMulRaw(f.stream, cublasltx.MatMulConfig{Alpha: 1, Epilogue: ep, Bias: f.bias.Ptr()},
			f.a.Ptr(), f.m, f.k, f.m,
			f.b.Ptr(), f.k, f.n, f.k,
			f.d.Ptr(), f.m, f.n, f.m)
		if err == nil || !strings.Contains(err.Error(), "auxiliary pointer") {
			t.Fatalf("epilogue %v with no aux: err = %v, want an auxiliary pointer error", ep, err)
		}
	}
}

// TestMatMulEpilogueUnchangedWithoutEpilogue pins that a zero Epilogue leaves
// the result bit-for-bit identical to the same config run through MatMul, which
// is the path that existed before epilogues were configurable.
func TestMatMulEpilogueUnchangedWithoutEpilogue(t *testing.T) {
	f := newEpilogueFixture(t)
	raw := f.run(t, cublasltx.MatMulConfig{Alpha: 1})
	if err := f.engine.MatMul(f.stream, f.a, f.m, f.k, f.b, f.n, f.d); err != nil {
		t.Fatalf("MatMul: %v", err)
	}
	if err := f.stream.Synchronize(nil); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	plain := make([]float32, f.m*f.n)
	if err := f.d.CopyToHost(plain); err != nil {
		t.Fatalf("download: %v", err)
	}
	for i := range plain {
		if raw[i] != plain[i] {
			t.Fatalf("zero epilogue changed the result: D = %v, want %v", raw, plain)
		}
	}
}

// TestMatMulEpilogueGELUAux checks that EpilogueGeluAuxBias writes the
// pre-activation matrix to Aux while D still holds the activated result, so
// both halves of the fused epilogue are observed rather than just D.
func TestMatMulEpilogueGELUAux(t *testing.T) {
	f := newEpilogueFixture(t)
	aux, err := cudamem.Alloc[float32](f.dev, f.m*f.n)
	f.owner.add(aux.Free)
	if err != nil {
		t.Fatalf("alloc aux: %v", err)
	}

	got := f.run(t, cublasltx.MatMulConfig{
		Alpha:    1,
		Epilogue: cublaslt.EpilogueGeluAuxBias,
		Bias:     f.bias.Ptr(),
		Aux:      aux.Ptr(),
		AuxLd:    f.m,
	})
	wantD := epilogueRef(f.hostA, f.hostB, f.hostC, f.m, f.k, f.n, geluAct)
	for i := range wantD {
		if math.Abs(float64(got[i]-wantD[i])) > 2e-3 {
			t.Fatalf("D = %v, want %v", got, wantD)
		}
	}

	gotAux := make([]float32, f.m*f.n)
	if err := aux.CopyToHost(gotAux); err != nil {
		t.Fatalf("download aux: %v", err)
	}
	wantAux := epilogueRef(f.hostA, f.hostB, f.hostC, f.m, f.k, f.n, nil)
	for i := range wantAux {
		if math.Abs(float64(gotAux[i]-wantAux[i])) > 2e-3 {
			t.Fatalf("aux = %v, want the pre-activation %v", gotAux, wantAux)
		}
	}
}

// TestMatMulEpilogueRowMajorBias confirms what the row-major helpers imply but
// never state: because they swap A and B and keep OrderCol, cuBLASLt's bias
// vector indexes rows of the column-major D, which are the output features of
// the row-major result. A bias that landed on the token axis instead would
// disagree with the reference for every row but the first.
func TestMatMulEpilogueRowMajorBias(t *testing.T) {
	dev, owner := requireDevice(t)
	stream, err := dev.NewStream()
	owner.stream(stream)
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	engine, err := cublasltx.NewEngine(dev)
	owner.add(engine.Close)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	// Row-major X is tokens by inFeatures, W is outFeatures by inFeatures, and
	// Y is tokens by outFeatures. tokens and outFeatures differ so a bias on
	// the wrong axis cannot accidentally line up.
	const tokens, inFeatures, outFeatures = 3, 4, 5
	hostX := make([]float32, tokens*inFeatures)
	for i := range hostX {
		hostX[i] = float32(i%5)*0.25 - 0.5
	}
	hostW := make([]float32, outFeatures*inFeatures)
	for i := range hostW {
		hostW[i] = float32(i%7)*0.5 - 1
	}
	hostBias := make([]float32, outFeatures)
	for i := range hostBias {
		hostBias[i] = float32(i)*0.75 - 1.5
	}

	upload := func(src []float32) cudamem.Slice[float32] {
		t.Helper()
		s, err := cudamem.Alloc[float32](dev, len(src))
		owner.add(s.Free)
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		if err := s.CopyFromHost(src); err != nil {
			t.Fatalf("upload: %v", err)
		}
		return s
	}
	x, w, bias := upload(hostX), upload(hostW), upload(hostBias)
	y := upload(make([]float32, tokens*outFeatures))

	// Y = X * W^T, matching LinearRowMajor's operand order: op(A)=W^T with W
	// read as column-major inFeatures by outFeatures, B = X.
	if err := engine.MatMulRaw(stream, cublasltx.MatMulConfig{
		Alpha:    1,
		TransA:   true,
		Epilogue: cublaslt.EpilogueBias,
		Bias:     bias.Ptr(),
	},
		w.Ptr(), inFeatures, outFeatures, inFeatures,
		x.Ptr(), inFeatures, tokens, inFeatures,
		y.Ptr(), outFeatures, tokens, outFeatures); err != nil {
		t.Fatalf("MatMulRaw: %v", err)
	}
	if err := stream.Synchronize(nil); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	got := make([]float32, tokens*outFeatures)
	if err := y.CopyToHost(got); err != nil {
		t.Fatalf("download: %v", err)
	}

	// Row-major reference: Y[t][o] = sum_i X[t][i]*W[o][i] + bias[o].
	for tok := range tokens {
		for o := range outFeatures {
			sum := float64(hostBias[o])
			for i := range inFeatures {
				sum += float64(hostX[tok*inFeatures+i]) * float64(hostW[o*inFeatures+i])
			}
			if diff := math.Abs(float64(got[tok*outFeatures+o]) - sum); diff > 2e-3 {
				t.Fatalf("Y[%d][%d] = %g, want %g (bias is on the wrong axis if this is off by a constant per row)",
					tok, o, got[tok*outFeatures+o], sum)
			}
		}
	}
}
