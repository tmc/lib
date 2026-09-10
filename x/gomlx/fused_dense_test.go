package gomlx

import (
	"errors"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/nn"
)

func TestFusedDenseValidationDoesNotEmit(t *testing.T) {
	tests := []struct {
		name string
		make func(*function) (compute.Value, compute.Value, compute.Value, compute.DenseConfig)
		want string
	}{
		{"input weight mismatch", func(f *function) (compute.Value, compute.Value, compute.Value, compute.DenseConfig) {
			x, _ := f.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
			w, _ := f.Parameter("w", shapes.Make(dtypes.Float32, 4, 5), nil)
			return x, w, nil, compute.DenseConfig{}
		}, "input feature"},
		{"bias mismatch", func(f *function) (compute.Value, compute.Value, compute.Value, compute.DenseConfig) {
			x, _ := f.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
			w, _ := f.Parameter("w", shapes.Make(dtypes.Float32, 3, 5), nil)
			b, _ := f.Parameter("b", shapes.Make(dtypes.Float32, 4), nil)
			return x, w, b, compute.DenseConfig{}
		}, "bias shape"},
		{"dtype mismatch", func(f *function) (compute.Value, compute.Value, compute.Value, compute.DenseConfig) {
			x, _ := f.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
			w, _ := f.Parameter("w", shapes.Make(dtypes.Float64, 3, 5), nil)
			return x, w, nil, compute.DenseConfig{}
		}, "matching"},
		{"layout", func(f *function) (compute.Value, compute.Value, compute.Value, compute.DenseConfig) {
			x, _ := f.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
			w, _ := f.Parameter("w", shapes.Make(dtypes.Float32, 3, 5), nil)
			return x, w, nil, compute.DenseConfig{WeightLayout: compute.DenseLayout(99)}
		}, "layout"},
		{"activation", func(f *function) (compute.Value, compute.Value, compute.Value, compute.DenseConfig) {
			x, _ := f.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
			w, _ := f.Parameter("w", shapes.Make(dtypes.Float32, 3, 5), nil)
			return x, w, nil, compute.DenseConfig{Activation: compute.ActivationType(99)}
		}, "activation"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := (&Backend{}).Builder(tt.name)
			f := b.Main().(*function)
			x, w, bias, options := tt.make(f)
			before := len(f.raw.Ops)
			if _, err := f.FusedDense(x, w, bias, options); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
			if got := len(f.raw.Ops); got != before {
				t.Fatalf("invalid fused dense emitted %d ops", got-before)
			}
		})
	}
}

func TestFusedDenseUnsupportedDTypeUsesFallbackSentinel(t *testing.T) {
	b := (&Backend{}).Builder("unsupported")
	f := b.Main().(*function)
	x, _ := f.Parameter("x", shapes.Make(dtypes.Int32, 2, 3), nil)
	w, _ := f.Parameter("w", shapes.Make(dtypes.Int32, 3, 2), nil)
	_, err := f.FusedDense(x, w, nil, compute.DenseConfig{})
	if !errors.Is(err, compute.ErrNotImplemented) {
		t.Fatalf("error = %v, want ErrNotImplemented", err)
	}
}

func TestFusedDenseShapesAndActivations(t *testing.T) {
	for _, dtype := range []dtypes.DType{dtypes.Float32, dtypes.Float64, dtypes.Float16, dtypes.BFloat16} {
		for _, layout := range []compute.DenseLayout{compute.DenseLayoutInputOutputs, compute.DenseLayoutOutputsInput} {
			for _, activation := range []compute.ActivationType{compute.ActivationNone, compute.ActivationGelu, compute.ActivationRelu, compute.ActivationSilu, compute.ActivationHardSwish, compute.ActivationTanh} {
				b := (&Backend{}).Builder("shape")
				f := b.Main().(*function)
				x, _ := f.Parameter("x", shapes.Make(dtype, 2, 4, 3), nil)
				wdims := []int{3, 5}
				if layout == compute.DenseLayoutOutputsInput {
					wdims = []int{5, 3}
				}
				w, _ := f.Parameter("w", shapes.Make(dtype, wdims...), nil)
				bias, _ := f.Parameter("bias", shapes.Make(dtype, 5), nil)
				out, err := f.FusedDense(x, w, bias, compute.DenseConfig{WeightLayout: layout, Activation: activation})
				if err != nil {
					t.Fatalf("layout=%v activation=%v: %v", layout, activation, err)
				}
				got, err := f.Shape(out)
				if err != nil {
					t.Fatal(err)
				}
				if want := (shapes.Shape{DType: dtype, Dimensions: []int{2, 4, 5}}); !got.Equal(want) {
					t.Fatalf("layout=%v activation=%v shape=%v want %v", layout, activation, got, want)
				}
			}
		}
	}
}

func TestNativeFusedDenseContract(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close() }()
	x := []float32{-2, 1, 2, -1, 0.5, 3, 1, -2, 0, 4, -3, 2}
	bias := []float32{0.25, -0.5}
	for _, tc := range []struct {
		name   string
		layout compute.DenseLayout
		weight []float32
	}{
		{"input_outputs", compute.DenseLayoutInputOutputs, []float32{1, -2, 0.5, 3, -1, 2}},
		{"outputs_input", compute.DenseLayoutOutputsInput, []float32{1, 0.5, -1, -2, 3, 2}},
	} {
		for _, activationType := range []compute.ActivationType{compute.ActivationNone, compute.ActivationGelu, compute.ActivationRelu, compute.ActivationSilu, compute.ActivationHardSwish, compute.ActivationTanh} {
			t.Run(tc.name+"/"+activationType.String(), func(t *testing.T) {
				got, err := testutil.Exec1(b, []any{x, tc.weight, bias}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
					xv, err := f.Reshape(p[0], 2, 2, 3)
					if err != nil {
						return nil, err
					}
					wdims := []int{3, 2}
					if tc.layout == compute.DenseLayoutOutputsInput {
						wdims = []int{2, 3}
					}
					wv, err := f.Reshape(p[1], wdims...)
					if err != nil {
						return nil, err
					}
					return f.FusedDense(xv, wv, p[2], compute.DenseConfig{WeightLayout: tc.layout, Activation: activationType})
				})
				if err != nil {
					t.Fatal(err)
				}
				got3, ok := got.([][][]float32)
				if !ok {
					t.Fatalf("output type %T", got)
				}
				gotv := make([]float32, 0, 8)
				for _, batch := range got3 {
					for _, row := range batch {
						gotv = append(gotv, row...)
					}
				}
				want := denseOracle(x, tc.weight, bias, tc.layout, activationType)
				if len(gotv) != len(want) {
					t.Fatalf("output length %d want %d", len(gotv), len(want))
				}
				for i := range want {
					if math.Abs(float64(gotv[i]-want[i])) > 2e-3 {
						t.Fatalf("output[%d]=%g want %g", i, gotv[i], want[i])
					}
				}
			})
		}
	}
}

func TestNativeMLNNDenseConsumer(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close() }()
	g := NewGraph(b, "ml_nn_dense")
	x := Reshape(Const(g, []float32{-2, 1, 2, -1, 0.5, 3, 1, -2, 0, 4, -3, 2}), 2, 2, 3)
	w := Reshape(Const(g, []float32{1, -2, 0.5, 3, -1, 2}), 3, 2)
	bias := Const(g, []float32{0.25, -0.5})
	y := nn.Dense(x, w, bias, compute.DenseLayoutInputOutputs, activation.TypeRelu)
	if y.Type() != NodeTypeFusedDense {
		t.Fatalf("nn.Dense selected graph node %v, want fused dense", y.Type())
	}
	if err := g.Compile(y); err != nil {
		t.Fatal(err)
	}
	result := g.Run()[0].Value()
	got3, ok := result.([][][]float32)
	if !ok {
		t.Fatalf("output type %T", result)
	}
	got := make([]float32, 0, 8)
	for _, batch := range got3 {
		for _, row := range batch {
			got = append(got, row...)
		}
	}
	want := denseOracle([]float32{-2, 1, 2, -1, 0.5, 3, 1, -2, 0, 4, -3, 2}, []float32{1, -2, 0.5, 3, -1, 2}, []float32{0.25, -0.5}, compute.DenseLayoutInputOutputs, compute.ActivationRelu)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 2e-3 {
			t.Fatalf("output[%d]=%g want %g", i, got[i], want[i])
		}
	}
	gi := NewGraph(b, "ml_nn_dense_integer_fallback")
	ix := Reshape(Const(gi, []int32{1, 2, 3}), 1, 3)
	iw := Reshape(Const(gi, []int32{1, 2, 3, 4, 5, 6}), 3, 2)
	iy := nn.Dense(ix, iw, nil, compute.DenseLayoutInputOutputs)
	if iy.Type() == NodeTypeFusedDense {
		t.Fatal("integer nn.Dense unexpectedly selected fused dense")
	}
	if err := gi.Compile(iy); err != nil {
		t.Fatal(err)
	}
	if got := gi.Run()[0].Value().([][]int32); got[0][0] != 22 || got[0][1] != 28 {
		t.Fatalf("integer fallback output %v", got)
	}
}

func TestNativeFusedDenseMultidimensionalOutputs(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close() }()
	x := []float32{1, 2, 3}
	bias := []float32{1, 2, 3, 4}
	for _, tc := range []struct {
		name   string
		layout compute.DenseLayout
		weight []float32
	}{
		{"input_outputs", compute.DenseLayoutInputOutputs, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
		{"outputs_input", compute.DenseLayoutOutputsInput, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{x, tc.weight, bias}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				xv, e := f.Reshape(p[0], 1, 3)
				if e != nil {
					return nil, e
				}
				wdims := []int{3, 2, 2}
				if tc.layout == compute.DenseLayoutOutputsInput {
					wdims = []int{2, 2, 3}
				}
				wv, e := f.Reshape(p[1], wdims...)
				if e != nil {
					return nil, e
				}
				bv, e := f.Reshape(p[2], 2, 2)
				if e != nil {
					return nil, e
				}
				return f.FusedDense(xv, wv, bv, compute.DenseConfig{WeightLayout: tc.layout})
			})
			if err != nil {
				t.Fatal(err)
			}
			values := got.([][][]float32)
			if len(values) != 1 || len(values[0]) != 2 || len(values[0][0]) != 2 {
				t.Fatalf("shape/value %T %v", got, got)
			}
			var want [2][2]float32
			for j := 0; j < 2; j++ {
				for l := 0; l < 2; l++ {
					var sum float32
					for k := 0; k < 3; k++ {
						wi := k*4 + j*2 + l
						if tc.layout == compute.DenseLayoutOutputsInput {
							wi = j*6 + l*3 + k
						}
						sum += x[k] * tc.weight[wi]
					}
					want[j][l] = sum + bias[j*2+l]
				}
			}
			for j := range want {
				for l := range want[j] {
					if math.Abs(float64(values[0][j][l]-want[j][l])) > 2e-3 {
						t.Fatalf("output[%d][%d]=%g want %g", j, l, values[0][j][l], want[j][l])
					}
				}
			}
		})
	}
}

func TestNativeFusedDenseOtherFloatDTypes(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close() }()
	t.Run("f64", func(t *testing.T) {
		got, err := testutil.Exec1(b, []any{[]float64{1, 2, 3}, []float64{1, 2, 3, 4, 5, 6}, []float64{1, 2}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
			x, e := f.Reshape(p[0], 1, 1, 3)
			if e != nil {
				return nil, e
			}
			w, e := f.Reshape(p[1], 3, 2)
			if e != nil {
				return nil, e
			}
			return f.FusedDense(x, w, p[2], compute.DenseConfig{})
		})
		if err != nil {
			t.Fatal(err)
		}
		if got.([][][]float64)[0][0][0] != 23 || got.([][][]float64)[0][0][1] != 30 {
			t.Fatalf("got %v", got)
		}
	})
	t.Run("f16", func(t *testing.T) {
		x := []float16.Float16{float16.FromFloat32(1), float16.FromFloat32(2), float16.FromFloat32(3)}
		w := []float16.Float16{float16.FromFloat32(1), float16.FromFloat32(2), float16.FromFloat32(3), float16.FromFloat32(4), float16.FromFloat32(5), float16.FromFloat32(6)}
		bias := []float16.Float16{float16.FromFloat32(1), float16.FromFloat32(2)}
		got, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
			x, e := f.Reshape(p[0], 1, 1, 3)
			if e != nil {
				return nil, e
			}
			w, e := f.Reshape(p[1], 3, 2)
			if e != nil {
				return nil, e
			}
			return f.FusedDense(x, w, p[2], compute.DenseConfig{})
		})
		if err != nil {
			t.Fatal(err)
		}
		values := got.([][][]float16.Float16)
		if math.Abs(float64(values[0][0][0].Float32()-23)) > 0.1 || math.Abs(float64(values[0][0][1].Float32()-30)) > 0.1 {
			t.Fatalf("got %v", got)
		}
	})
	for _, activationType := range []compute.ActivationType{compute.ActivationGelu, compute.ActivationHardSwish} {
		t.Run("f16/"+activationType.String(), func(t *testing.T) {
			x := []float16.Float16{float16.FromFloat32(-3), float16.FromFloat32(0), float16.FromFloat32(3)}
			w := []float16.Float16{float16.FromFloat32(1)}
			got, err := testutil.Exec1(b, []any{x, w}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				x, e := f.Reshape(p[0], 3, 1)
				if e != nil {
					return nil, e
				}
				w, e := f.Reshape(p[1], 1, 1)
				if e != nil {
					return nil, e
				}
				return f.FusedDense(x, w, nil, compute.DenseConfig{Activation: activationType})
			})
			if err != nil {
				t.Fatal(err)
			}
			values := got.([][]float16.Float16)
			want := []float32{-0.004, 0, 2.996}
			if activationType == compute.ActivationHardSwish {
				want = []float32{-0.0, 0, 3}
			}
			for i := range want {
				if math.Abs(float64(values[i][0].Float32()-want[i])) > 0.03 {
					t.Fatalf("output[%d]=%g want %g", i, values[i][0].Float32(), want[i])
				}
			}
		})
	}
	t.Run("bf16", func(t *testing.T) {
		x := []bfloat16.BFloat16{bfloat16.FromFloat32(1), bfloat16.FromFloat32(2), bfloat16.FromFloat32(3)}
		w := []bfloat16.BFloat16{bfloat16.FromFloat32(1), bfloat16.FromFloat32(2), bfloat16.FromFloat32(3), bfloat16.FromFloat32(4), bfloat16.FromFloat32(5), bfloat16.FromFloat32(6)}
		bias := []bfloat16.BFloat16{bfloat16.FromFloat32(1), bfloat16.FromFloat32(2)}
		got, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
			x, e := f.Reshape(p[0], 1, 1, 3)
			if e != nil {
				return nil, e
			}
			w, e := f.Reshape(p[1], 3, 2)
			if e != nil {
				return nil, e
			}
			return f.FusedDense(x, w, p[2], compute.DenseConfig{})
		})
		if err != nil {
			t.Fatal(err)
		}
		values := got.([][][]bfloat16.BFloat16)
		if math.Abs(float64(values[0][0][0].Float32()-23)) > 0.2 || math.Abs(float64(values[0][0][1].Float32()-30)) > 0.2 {
			t.Fatalf("got %v", got)
		}
	})
}

func denseOracle(x, w, bias []float32, layout compute.DenseLayout, activation compute.ActivationType) []float32 {
	out := make([]float32, len(x)/3*2)
	for i := 0; i < len(x)/3; i++ {
		for j := 0; j < 2; j++ {
			var sum float32
			for k := 0; k < 3; k++ {
				wi := k*2 + j
				if layout == compute.DenseLayoutOutputsInput {
					wi = j*3 + k
				}
				sum += x[i*3+k] * w[wi]
			}
			v := sum + bias[j]
			switch activation {
			case compute.ActivationGelu:
				v = v * float32(0.5*(1+math.Erf(float64(v)/math.Sqrt2)))
			case compute.ActivationRelu:
				if v < 0 {
					v = 0
				}
			case compute.ActivationSilu:
				v *= float32(1 / (1 + math.Exp(-float64(v))))
			case compute.ActivationHardSwish:
				g := v/6 + 0.5
				if g < 0 {
					g = 0
				}
				if g > 1 {
					g = 1
				}
				v *= g
			case compute.ActivationTanh:
				v = float32(math.Tanh(float64(v)))
			}
			out[i*2+j] = v
		}
	}
	return out
}
