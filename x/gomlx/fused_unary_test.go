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
)

func TestFusedUnaryValidationDoesNotEmit(t *testing.T) {
	tests := []struct {
		name string
		call func(*function, compute.Value) error
		want string
	}{
		{"softmax negative axis", func(f *function, x compute.Value) error { _, err := f.FusedSoftmax(x, -1); return err }, "axis"},
		{"softmax large axis", func(f *function, x compute.Value) error { _, err := f.FusedSoftmax(x, 2); return err }, "axis"},
		{"gelu integer", func(f *function, x compute.Value) error { _, err := f.FusedGelu(x, true); return err }, "not implemented"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := (&Backend{}).Builder(tt.name).Main().(*function)
			dtype := dtypes.Float32
			if tt.name == "gelu integer" {
				dtype = dtypes.Int32
			}
			x, err := f.Parameter("x", shapes.Make(dtype, 2, 2), nil)
			if err != nil {
				t.Fatal(err)
			}
			before := len(f.raw.Ops)
			err = tt.call(f, x)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
			if len(f.raw.Ops) != before {
				t.Fatalf("invalid fused unary emitted %d ops", len(f.raw.Ops)-before)
			}
		})
	}
}

func TestFusedUnaryUnsupportedTypesUseFallbackSentinel(t *testing.T) {
	f := (&Backend{}).Builder("unsupported").Main().(*function)
	x, err := f.Parameter("x", shapes.Make(dtypes.Int32, 2, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, call := range []func() error{
		func() error { _, err := f.FusedGelu(x, true); return err },
		func() error { _, err := f.FusedSoftmax(x, 0); return err },
	} {
		if err := call(); !errors.Is(err, compute.ErrNotImplemented) {
			t.Fatalf("error = %v, want ErrNotImplemented", err)
		}
	}
}

func TestFusedUnaryShapes(t *testing.T) {
	for _, dtype := range []dtypes.DType{dtypes.Float32, dtypes.Float64, dtypes.Float16, dtypes.BFloat16} {
		for _, exact := range []bool{true, false} {
			f := (&Backend{}).Builder("gelu").Main().(*function)
			x, _ := f.Parameter("x", shapes.Make(dtype, 2, 3, 4), nil)
			y, err := f.FusedGelu(x, exact)
			if err != nil {
				t.Fatalf("dtype=%v exact=%v: %v", dtype, exact, err)
			}
			if got, _ := f.Shape(y); !got.Equal(shapes.Make(dtype, 2, 3, 4)) {
				t.Fatalf("dtype=%v exact=%v shape=%v", dtype, exact, got)
			}
		}
		for _, axis := range []int{0, 1, 2} {
			f := (&Backend{}).Builder("softmax").Main().(*function)
			x, _ := f.Parameter("x", shapes.Make(dtype, 2, 3, 4), nil)
			y, err := f.FusedSoftmax(x, axis)
			if err != nil {
				t.Fatalf("dtype=%v axis=%d: %v", dtype, axis, err)
			}
			if got, _ := f.Shape(y); !got.Equal(shapes.Make(dtype, 2, 3, 4)) {
				t.Fatalf("dtype=%v axis=%d shape=%v", dtype, axis, got)
			}
		}
	}
}

func TestNativeFusedUnaryContract(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close() }()

	x := []float32{-20, -3, -1, 0, 0.5, 2, 20}
	for _, exact := range []bool{true, false} {
		t.Run("gelu", func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{x}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedGelu(p[0], exact)
			})
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range got.([]float32) {
				want := geluOracle(float64(x[i]), exact)
				if math.Abs(float64(v)-want) > 3e-4 {
					t.Fatalf("exact=%v output[%d]=%g want %g", exact, i, v, want)
				}
			}
		})
	}

	for _, axis := range []int{0, 1} {
		t.Run("softmax", func(t *testing.T) {
			input := []float32{-1000, -999, -998, 1000, 1001, 1002}
			got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				x, err := f.Reshape(p[0], 2, 3)
				if err != nil {
					return nil, err
				}
				return f.FusedSoftmax(x, axis)
			})
			if err != nil {
				t.Fatal(err)
			}
			want := softmaxOracle(input, 2, 3, axis)
			for i, v := range got.([][]float32) {
				for j, v := range v {
					idx := i*3 + j
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-want[idx]) > 3e-4 {
						t.Fatalf("axis=%d output[%d]=%g want %g", axis, idx, v, want[idx])
					}
				}
			}
		})
	}
}

func TestNativeFusedUnaryConsumers(t *testing.T) {
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
	g := NewGraph(b, "fused_unary_consumers")
	x := Const(g, []float32{-2, -1, 0, 1, 2, 3})
	gelu := activation.Gelu(x)
	if gelu.Type() != NodeTypeFusedGelu {
		t.Fatalf("activation.Gelu selected %v, want fused gelu", gelu.Type())
	}
	softmax := Softmax(Reshape(x, 2, 3), -1)
	if softmax.Type() != NodeTypeFusedSoftmax {
		t.Fatalf("graph.Softmax selected %v, want fused softmax", softmax.Type())
	}
	if err := g.Compile(gelu, softmax); err != nil {
		t.Fatal(err)
	}
	got := g.Run()
	if len(got) != 2 {
		t.Fatalf("result count %d, want 2", len(got))
	}
	defer func() {
		for _, tensor := range got {
			if err := tensor.FinalizeAll(); err != nil {
				t.Error(err)
			}
		}
		g.Finalize()
	}()
	geluValues := got[0].Value().([]float32)
	for i, v := range geluValues {
		if want := geluOracle(float64([]float32{-2, -1, 0, 1, 2, 3}[i]), true); math.Abs(float64(v)-want) > 3e-4 {
			t.Fatalf("gelu output[%d]=%g want %g", i, v, want)
		}
	}
	softmaxValues := got[1].Value().([][]float32)
	wantSoftmax := softmaxOracle([]float32{-2, -1, 0, 1, 2, 3}, 2, 3, 1)
	for i, row := range softmaxValues {
		for j, v := range row {
			if want := wantSoftmax[i*3+j]; math.Abs(float64(v)-want) > 3e-4 {
				t.Fatalf("softmax output[%d,%d]=%g want %g", i, j, v, want)
			}
		}
	}
}

func TestNativeFusedUnaryOtherFloatDTypes(t *testing.T) {
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
	tests := []struct {
		name  string
		input any
		tol   float64
	}{
		{"f64", []float64{-20, -3, -1, 0, 0.5, 2, 20}, 1e-10},
		{"f16", half16([]float32{-20, -3, -1, 0, 0.5, 2, 20}), 2e-2},
		{"bf16", bhalf16([]float32{-20, -3, -1, 0, 0.5, 2, 20}), 4e-2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := unaryFloat64(tt.input)
			for _, exact := range []bool{true, false} {
				got, err := testutil.Exec1(b, []any{tt.input}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedGelu(p[0], exact)
				})
				if err != nil {
					t.Fatal(err)
				}
				for i, v := range unaryFloat64(got) {
					want := geluOracle(input[i], exact)
					if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v-want) > tt.tol {
						t.Fatalf("exact=%v output[%d]=%g want %g", exact, i, v, want)
					}
				}
			}
			softmaxInput := reshapeUnaryInput(tt.input)
			got, err := testutil.Exec1(b, []any{softmaxInput}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				x, err := f.Reshape(p[0], 2, 3)
				if err != nil {
					return nil, err
				}
				return f.FusedSoftmax(x, 1)
			})
			if err != nil {
				t.Fatal(err)
			}
			want := softmaxOracle(unaryFloat32(softmaxInput), 2, 3, 1)
			for i, v := range unaryFloat64(flatMatrix(got)) {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v-want[i]) > tt.tol {
					t.Fatalf("softmax output[%d]=%g want %g", i, v, want[i])
				}
			}
		})
	}
}

func TestNativeFusedSoftmaxMaxSubtractionMutant(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close() }()
	mutant, err := testutil.Exec1(b, []any{[]float32{1000, 1001}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		x, err := f.Exp(p[0])
		if err != nil {
			return nil, err
		}
		sum, err := f.ReduceSum(x, 0)
		if err != nil {
			return nil, err
		}
		return f.Div(x, sum)
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range mutant.([]float32) {
		if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
			t.Fatalf("mutant output=%g, want non-finite overflow witness", v)
		}
	}
	got, err := testutil.Exec1(b, []any{[]float32{1000, 1001}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.FusedSoftmax(p[0], 0)
	})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got.([]float32) {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("restored output[%d]=%g", i, v)
		}
	}
}

func TestNativeFusedSoftmaxSingletonAxis(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close() }()
	for _, input := range [][]float32{{7}} {
		got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
			return f.FusedSoftmax(p[0], 0)
		})

		if err != nil {
			t.Fatalf("input=%v: %v", input, err)
		}
		values := got.([]float32)
		if len(values) != len(input) {
			t.Fatalf("input=%v output length %d", input, len(values))
		}
		if len(values) == 1 && values[0] != 1 {
			t.Fatalf("singleton output=%v, want 1", values)
		}
	}
}

func geluOracle(x float64, exact bool) float64 {
	if exact {
		return x * 0.5 * (1 + math.Erf(x/math.Sqrt2))
	}
	return x * 0.5 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
}

func softmaxOracle(input []float32, rows, cols, axis int) []float64 {
	out := make([]float64, len(input))
	for outer := 0; outer < rows; outer++ {
		for inner := 0; inner < cols; inner++ {
			first, count, stride := 0, 0, 0
			if axis == 0 {
				first, count, stride = inner, rows, cols
			} else {
				first, count, stride = outer*cols, cols, 1
			}
			max := math.Inf(-1)
			for i := 0; i < count; i++ {
				max = math.Max(max, float64(input[first+i*stride]))
			}
			sum := 0.0
			for i := 0; i < count; i++ {
				sum += math.Exp(float64(input[first+i*stride]) - max)
			}
			for i := 0; i < count; i++ {
				out[first+i*stride] = math.Exp(float64(input[first+i*stride])-max) / sum
			}
		}
	}
	return out
}

func half16(values []float32) []float16.Float16 {
	out := make([]float16.Float16, len(values))
	for i, v := range values {
		out[i] = float16.FromFloat32(v)
	}
	return out
}

func bhalf16(values []float32) []bfloat16.BFloat16 {
	out := make([]bfloat16.BFloat16, len(values))
	for i, v := range values {
		out[i] = bfloat16.FromFloat32(v)
	}
	return out
}

func reshapeUnaryInput(v any) any {
	switch v := v.(type) {
	case []float64:
		return append([]float64(nil), v[:6]...)
	case []float16.Float16:
		return append([]float16.Float16(nil), v[:6]...)
	case []bfloat16.BFloat16:
		return append([]bfloat16.BFloat16(nil), v[:6]...)
	default:
		panic("unexpected unary dtype")
	}
}

func flatMatrix(v any) any {
	switch v := v.(type) {
	case [][]float64:
		out := make([]float64, 0, len(v)*3)
		for _, row := range v {
			out = append(out, row...)
		}
		return out
	case [][]float16.Float16:
		out := make([]float16.Float16, 0, len(v)*3)
		for _, row := range v {
			out = append(out, row...)
		}
		return out
	case [][]bfloat16.BFloat16:
		out := make([]bfloat16.BFloat16, 0, len(v)*3)
		for _, row := range v {
			out = append(out, row...)
		}
		return out
	default:
		panic("unexpected softmax dtype")
	}
}

func unaryFloat32(v any) []float32 {
	switch v := v.(type) {
	case []float64:
		out := make([]float32, len(v))
		for i, x := range v {
			out[i] = float32(x)
		}
		return out
	case []float16.Float16:
		out := make([]float32, len(v))
		for i, x := range v {
			out[i] = x.Float32()
		}
		return out
	case []bfloat16.BFloat16:
		out := make([]float32, len(v))
		for i, x := range v {
			out[i] = x.Float32()
		}
		return out
	default:
		panic("unexpected unary dtype")
	}
}

func unaryFloat64(v any) []float64 {
	switch v := v.(type) {
	case []float64:
		return append([]float64(nil), v...)
	case []float16.Float16:
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x.Float32())
		}
		return out
	case []bfloat16.BFloat16:
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x.Float32())
		}
		return out
	default:
		panic("unexpected unary dtype")
	}
}

func TestNativeFusedSoftmaxEmptyAxes(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	})
	for _, dims := range [][]int{{0}, {2, 0}, {0, 3}} {
		shape := shapes.Make(dtypes.Float32, dims...)
		input, err := b.BufferFromFlatData(0, []float32{}, shape)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := input.Finalize(); err != nil {
				t.Error(err)
			}
		})
		for axis := range dims {
			builder := b.Builder("empty_softmax")
			f := builder.Main()
			x, err := f.Parameter("x", shape, nil)
			if err != nil {
				t.Fatal(err)
			}
			y, err := f.FusedSoftmax(x, axis)
			if err != nil {
				t.Fatal(err)
			}
			if err := f.Return([]compute.Value{y}, nil); err != nil {
				t.Fatal(err)
			}
			e, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(e.Finalize)
			out, err := e.Execute([]compute.Buffer{input}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			if len(out) != 1 {
				t.Fatalf("output count %d", len(out))
			}
			t.Cleanup(func() {
				if err := out[0].Finalize(); err != nil {
					t.Error(err)
				}
			})
			got, err := out[0].Shape()
			if err != nil || !got.Equal(shape) {
				t.Fatalf("shape=%v err=%v want=%v", got, err, shape)
			}
			if err := out[0].ToFlatData([]float32{}); err != nil {
				t.Fatal(err)
			}
		}
	}
}
