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
	"github.com/gomlx/gomlx/ml/nn"
)

func TestFusedLayerNormValidationDoesNotEmit(t *testing.T) {
	tests := []struct {
		name string
		call func(*function, compute.Value) error
		want string
	}{
		{"empty axes", func(f *function, x compute.Value) error {
			_, err := f.FusedLayerNorm(x, nil, 1e-5, nil, nil)
			return err
		}, "axes"},
		{"duplicate axes", func(f *function, x compute.Value) error {
			_, err := f.FusedLayerNorm(x, []int{1, 1}, 1e-5, nil, nil)
			return err
		}, "axis"},
		{"negative epsilon", func(f *function, x compute.Value) error {
			_, err := f.FusedLayerNorm(x, []int{1}, -1, nil, nil)
			return err
		}, "epsilon"},
		{"nan epsilon", func(f *function, x compute.Value) error {
			_, err := f.FusedLayerNorm(x, []int{1}, math.NaN(), nil, nil)
			return err
		}, "epsilon"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := (&Backend{}).Builder(tt.name).Main().(*function)
			x, _ := f.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
			before := len(f.raw.Ops)
			if err := tt.call(f, x); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
			if len(f.raw.Ops) != before {
				t.Fatalf("invalid fused layer norm emitted %d ops", len(f.raw.Ops)-before)
			}
		})
	}
}

func TestNativeFusedLayerNorm(t *testing.T) {
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
	x := []float32{1, 2, 3, 4, 10, 20, 30, 40}
	gamma := []float32{2, 3, 4, 5}
	beta := []float32{-1, 1, -2, 2}
	got, err := testutil.Exec1(b, []any{x, gamma, beta}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		x, err := f.Reshape(p[0], 2, 2, 2)
		if err != nil {
			return nil, err
		}
		g, err := f.Reshape(p[1], 2, 2)
		if err != nil {
			return nil, err
		}
		be, err := f.Reshape(p[2], 2, 2)
		if err != nil {
			return nil, err
		}
		return f.FusedLayerNorm(x, []int{2, 1}, 1e-5, g, be)
	})
	if err != nil {
		t.Fatal(err)
	}
	values := got.([][][]float32)
	for batch := range values {
		mean := 0.0
		for i := 0; i < 4; i++ {
			mean += float64(x[batch*4+i])
		}
		mean /= 4
		variance := 0.0
		for i := 0; i < 4; i++ {
			d := float64(x[batch*4+i]) - mean
			variance += d * d
		}
		variance /= 4
		for row := range values[batch] {
			for col, v := range values[batch][row] {
				idx := row*2 + col
				scaleIndex := col*2 + row
				want := (float64(x[batch*4+idx])-mean)/math.Sqrt(variance+1e-5)*float64(gamma[scaleIndex]) + float64(beta[scaleIndex])
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-want) > 4e-4 {
					t.Fatalf("[%d,%d,%d]=%g want %g", batch, row, col, v, want)
				}
			}
		}
	}
}

func TestNativeFusedLayerNormOtherFloatDTypes(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close() }()
	tests := []struct {
		name  string
		input any
		tol   float64
	}{
		{"f64", []float64{1, 2}, 1e-10},
		{"f16", []float16.Float16{float16.FromFloat32(1), float16.FromFloat32(2)}, 2e-2},
		{"bf16", []bfloat16.BFloat16{bfloat16.FromFloat32(1), bfloat16.FromFloat32(2)}, 3e-2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{tt.input}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedLayerNorm(p[0], []int{0}, 1e-5, nil, nil)
			})
			if err != nil {
				t.Fatal(err)
			}
			var values []float64
			switch v := got.(type) {
			case []float64:
				values = v
			case []float16.Float16:
				for _, x := range v {
					values = append(values, float64(x.Float32()))
				}
			case []bfloat16.BFloat16:
				for _, x := range v {
					values = append(values, float64(x.Float32()))
				}
			}
			want := 0.5 / math.Sqrt(0.25+1e-5)
			for i, v := range values {
				sign := 1.0
				if i == 0 {
					sign = -1
				}
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v-sign*want) > tt.tol {
					t.Fatalf("output[%d]=%g want %g", i, v, sign*want)
				}
			}
		})
	}
}

func TestNativeFusedLayerNormConsumer(t *testing.T) {
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
	g := NewGraph(b, "layernorm_consumer")
	x := Reshape(Const(g, []float32{1, 2, 3, 4}), 2, 2)
	y := nn.LayerNorm(x, []int{1}, 1e-5, nil, nil, nil)
	if y.Type() != NodeTypeFusedLayerNorm {
		t.Fatalf("nn.LayerNorm selected %v", y.Type())
	}
	if err := g.Compile(y); err != nil {
		t.Fatal(err)
	}
	out := g.Run()
	defer func() {
		for _, v := range out {
			if err := v.FinalizeAll(); err != nil {
				t.Error(err)
			}
		}
		g.Finalize()
	}()
	values := out[0].Value().([][]float32)
	for i, row := range values {
		for j, v := range row {
			want := float64(-1)
			if j == 1 {
				want = 1
			}
			want *= 0.5 / math.Sqrt(.25+1e-5)
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-want) > 4e-4 {
				t.Fatalf("[%d,%d]=%g want %g", i, j, v, want)
			}
		}
	}
}

func TestNativeFusedLayerNormEpsilonMutant(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close() }()
	mutant, err := testutil.Exec1(b, []any{[]float32{3, 3}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		mean, err := f.ReduceSum(p[0], 0)
		if err != nil {
			return nil, err
		}
		two, err := f.Constant([]float32{2})
		if err != nil {
			return nil, err
		}
		mean, err = f.Div(mean, two)
		if err != nil {
			return nil, err
		}
		mean, err = f.BroadcastInDim(mean, shapes.Make(dtypes.Float32, 2), nil)
		if err != nil {
			return nil, err
		}
		z, err := f.Sub(p[0], mean)
		if err != nil {
			return nil, err
		}
		return f.Div(z, z)
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range mutant.([]float32) {
		if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
			t.Fatalf("mutant=%g want nonfinite", v)
		}
	}
	restored, err := testutil.Exec1(b, []any{[]float32{3, 3}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.FusedLayerNorm(p[0], []int{0}, 1e-5, nil, nil)
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range restored.([]float32) {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("restored=%g", v)
		}
	}
}

func TestFusedLayerNormUnsupportedTypeUsesFallback(t *testing.T) {
	f := (&Backend{}).Builder("unsupported").Main().(*function)
	x, _ := f.Parameter("x", shapes.Make(dtypes.Int32, 2, 3), nil)
	_, err := f.FusedLayerNorm(x, []int{1}, 1e-5, nil, nil)
	if !errors.Is(err, compute.ErrNotImplemented) {
		t.Fatalf("error = %v, want ErrNotImplemented", err)
	}
}

func TestFusedLayerNormShapes(t *testing.T) {
	for _, dtype := range []dtypes.DType{dtypes.Float32, dtypes.Float64, dtypes.Float16, dtypes.BFloat16} {
		f := (&Backend{}).Builder("shape").Main().(*function)
		x, _ := f.Parameter("x", shapes.Make(dtype, 2, 3, 4), nil)
		gamma, _ := f.Parameter("gamma", shapes.Make(dtype, 3, 4), nil)
		beta, _ := f.Parameter("beta", shapes.Make(dtype, 3, 4), nil)
		y, err := f.FusedLayerNorm(x, []int{1, 2}, 1e-5, gamma, beta)
		if err != nil {
			t.Fatalf("dtype=%v: %v", dtype, err)
		}
		if got, _ := f.Shape(y); !got.Equal(shapes.Make(dtype, 2, 3, 4)) {
			t.Fatalf("dtype=%v shape=%v", dtype, got)
		}
	}
}

func TestFusedLayerNormUnsortedAxesAndEmptyShapeDoNotEmit(t *testing.T) {
	f := (&Backend{}).Builder("unsorted").Main().(*function)
	x, _ := f.Parameter("x", shapes.Make(dtypes.Float32, 2, 3, 4), nil)
	gamma, _ := f.Parameter("gamma", shapes.Make(dtypes.Float32, 4, 3), nil)
	beta, _ := f.Parameter("beta", shapes.Make(dtypes.Float32, 4, 3), nil)
	y, err := f.FusedLayerNorm(x, []int{2, 1}, 1e-5, gamma, beta)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := f.Shape(y); !got.Equal(shapes.Make(dtypes.Float32, 2, 3, 4)) {
		t.Fatalf("shape=%v", got)
	}

	f = (&Backend{}).Builder("empty").Main().(*function)
	empty, _ := f.Parameter("x", shapes.Make(dtypes.Float16, 2, 0), nil)
	before := len(f.raw.Ops)
	_, err = f.FusedLayerNorm(empty, []int{1}, 1e-5, nil, nil)
	if !errors.Is(err, compute.ErrNotImplemented) {
		t.Fatalf("empty error=%v, want ErrNotImplemented", err)
	}
	if len(f.raw.Ops) != before {
		t.Fatalf("empty fused layer norm emitted %d ops", len(f.raw.Ops)-before)
	}
}
