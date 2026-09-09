package gomlx

import (
	"math"
	"os"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
)

func TestNativeActivations(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	input := []float64{math.Inf(-1), -20, -5, -1, -1e-12, math.Copysign(0, -1), 0, 1e-12, 1, 5, 20, math.Inf(1), math.NaN()}
	for _, tc := range []struct {
		name string
		op   func(compute.Function, compute.Value) (compute.Value, error)
		want func(float64) float64
	}{
		{"logistic", func(f compute.Function, v compute.Value) (compute.Value, error) { return f.Logistic(v) }, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }},
		{"erf", func(f compute.Function, v compute.Value) (compute.Value, error) { return f.Erf(v) }, math.Erf},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := testutil.Exec1(b, []any{input}, func(f compute.Function, v []compute.Value) (compute.Value, error) { return tc.op(f, v[0]) })
			if err != nil {
				t.Fatal(err)
			}
			for i, got := range out.([]float64) {
				want := tc.want(input[i])
				if math.IsNaN(want) {
					if !math.IsNaN(got) {
						t.Fatalf("input %g: got %g want NaN", input[i], got)
					}
					continue
				}
				if math.IsNaN(got) || math.Abs(got-want) > 1e-12*math.Max(math.Abs(want), 1e-15) || want == 0 && math.Signbit(got) != math.Signbit(want) {
					t.Fatalf("input %g: got %.17g want %.17g", input[i], got, want)
				}
			}
		})
	}
}

func TestNativeActivationGradients(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	e, err := graph.NewExec(b, func(x *graph.Node) []*graph.Node {
		sigmoid, erf, gelu := graph.Sigmoid(x), graph.Erf(x), activation.Gelu(x)
		return []*graph.Node{sigmoid, erf, gelu, graph.Gradient(graph.ReduceAllSum(sigmoid), x)[0], graph.Gradient(graph.ReduceAllSum(erf), x)[0], graph.Gradient(graph.ReduceAllSum(gelu), x)[0]}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	input := []float64{-5, -2, -.25, 0, .25, 2, 5}
	out, err := e.Call(input)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, v := range out {
			v.FinalizeAll()
		}
	}()
	for i, x := range input {
		sigmoid := 1 / (1 + math.Exp(-x))
		phi := .5 * (1 + math.Erf(x/math.Sqrt2))
		want := []float64{sigmoid, math.Erf(x), x * phi, sigmoid * (1 - sigmoid), 2 / math.Sqrt(math.Pi) * math.Exp(-x*x), phi + x/math.Sqrt(2*math.Pi)*math.Exp(-x*x/2)}
		for j, v := range out {
			got := v.Value().([]float64)[i]
			if math.IsNaN(got) || math.Abs(got-want[j]) > 1e-12*math.Max(1, math.Abs(want[j])) {
				t.Fatalf("output %d at %g: got %.17g want %.17g", j, x, got, want[j])
			}
		}
	}
}

func TestActivationValidation(t *testing.T) {
	for _, name := range []string{"erf", "logistic"} {
		for _, dtype := range []dtypes.DType{dtypes.Bool, dtypes.Int64, dtypes.Uint32} {
			t.Run(name+"/"+dtype.String(), func(t *testing.T) {
				f := (&Backend{}).Builder("activation").Main().(*function)
				x, err := f.Parameter("x", shapes.Make(dtype, 2), nil)
				if err != nil {
					t.Fatal(err)
				}
				n := len(f.raw.Ops)
				if name == "erf" {
					_, err = f.Erf(x)
				} else {
					_, err = f.Logistic(x)
				}
				if err == nil || len(f.raw.Ops) != n {
					t.Fatal("invalid dtype accepted or emitted IR")
				}
			})
		}
	}
}

func TestNativeHalfActivations(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	input := []float32{-5, -2, -.5, 0, .5, 2, 5}
	for _, dtype := range []dtypes.DType{dtypes.Float16, dtypes.BFloat16} {
		t.Run(dtype.String(), func(t *testing.T) {
			var values any
			tolerance := 2.0 / 1024
			if dtype == dtypes.Float16 {
				v := make([]float16.Float16, len(input))
				for i, x := range input {
					v[i] = float16.FromFloat32(x)
				}
				values = v
			} else {
				v := make([]bfloat16.BFloat16, len(input))
				for i, x := range input {
					v[i] = bfloat16.FromFloat32(x)
				}
				values = v
				tolerance = 2.0 / 128
			}
			e, err := graph.NewExec(b, func(x *graph.Node) []*graph.Node {
				return []*graph.Node{graph.ConvertDType(graph.Logistic(x), dtypes.Float32), graph.ConvertDType(graph.Erf(x), dtypes.Float32)}
			})
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			out, err := e.Call(values)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				for _, v := range out {
					v.FinalizeAll()
				}
			}()
			for i, x := range input {
				want := []float64{1 / (1 + math.Exp(-float64(x))), math.Erf(float64(x))}
				for j, v := range out {
					got := float64(v.Value().([]float32)[i])
					// Two machine epsilons allow final rounding and backend excess precision.
					if math.IsNaN(got) || math.Abs(got-want[j]) > tolerance*math.Max(1, math.Abs(want[j])) {
						t.Fatalf("output %d at %g: got %g want %g", j, x, got, want[j])
					}
				}
			}
		})
	}
}
