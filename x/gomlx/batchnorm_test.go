package gomlx

import (
	"math"
	"os"
	"slices"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
)

// batchNormReference computes population statistics independently of the graph.
func batchNormReference(x, scale, offset []float64, dims []int, axis int, epsilon float64) (y, mean, variance []float64) {
	features, stride := dims[axis], 1
	for _, d := range dims[axis+1:] {
		stride *= d
	}
	count := float64(len(x) / features)
	mean, variance = make([]float64, features), make([]float64, features)
	for i, v := range x {
		mean[(i/stride)%features] += v / count
	}
	for i, v := range x {
		c := (i / stride) % features
		delta := v - mean[c]
		variance[c] += delta * delta / count
	}
	y = make([]float64, len(x))
	for i, v := range x {
		c := (i / stride) % features
		y[i] = (v-mean[c])/math.Sqrt(variance[c]+epsilon)*scale[c] + offset[c]
	}
	return
}

func TestNativeBatchNormGradients(t *testing.T) {
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
	for _, tc := range []struct {
		name     string
		dims     []int
		axis     int
		constant bool
	}{
		{"columns", []int{3, 2}, 1, false}, {"rows", []int{3, 2}, 0, false}, {"negative axis", []int{3, 2}, -1, false}, {"spatial", []int{2, 2, 2, 2}, 1, false}, {"constant", []int{3, 2}, 1, true}, {"one per feature", []int{1, 2}, 1, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			axis := tc.axis
			if axis < 0 {
				axis += len(tc.dims)
			}
			n := 1
			for _, d := range tc.dims {
				n *= d
			}
			x, g := make([]float64, n), make([]float64, n)
			for i := range x {
				x[i] = float64((i*7)%13) - 3
				g[i] = float64(i%5) - 1.5
				if tc.constant {
					x[i] = 3
				}
			}
			scale, offset := make([]float64, tc.dims[axis]), make([]float64, tc.dims[axis])
			for i := range scale {
				scale[i] = 1.5 - float64(i)*2.25
				offset[i] = float64(i) + .3
			}
			const epsilon = .125
			e, err := graph.NewExec(b, func(x, s, o, g *graph.Node) []*graph.Node {
				input := graph.Reshape(x, tc.dims...)
				y, mean, variance := graph.InternalBatchNormForTraining(input, s, o, epsilon, tc.axis)
				inference := graph.InternalBatchNormForInference(input, s, o, mean, variance, epsilon, tc.axis)
				loss := graph.ReduceAllSum(graph.Mul(graph.Reshape(y, -1), g))
				grad := graph.Gradient(loss, x, s, o)
				return []*graph.Node{graph.Reshape(y, -1), mean, variance, graph.Reshape(inference, -1), grad[0], grad[1], grad[2]}
			})
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			out, err := e.Call(x, scale, offset, g)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				for _, v := range out {
					v.FinalizeAll()
				}
			}()
			y, mean, variance := batchNormReference(x, scale, offset, tc.dims, axis, epsilon)
			loss := func() float64 {
				y, _, _ := batchNormReference(x, scale, offset, tc.dims, axis, epsilon)
				sum := 0.0
				for i, v := range y {
					sum += v * g[i]
				}
				return sum
			}
			gradients := make([][]float64, 3)
			for j, v := range [][]float64{x, scale, offset} {
				gradients[j] = make([]float64, len(v))
				for i, old := range v {
					const delta = 1e-5
					v[i] = old + delta
					hi := loss()
					v[i] = old - delta
					lo := loss()
					v[i] = old
					gradients[j][i] = (hi - lo) / (2 * delta)
				}
			}
			want := [][]float64{y, mean, variance, y, gradients[0], gradients[1], gradients[2]}
			for j, v := range out {
				got := v.Value().([]float64)
				if len(got) != len(want[j]) {
					t.Fatal("output length")
				}
				for i, w := range want[j] {
					tol := 1e-10
					if j >= 4 {
						tol = 2e-8
					}
					if math.IsNaN(got[i]) || math.Abs(got[i]-w) > tol*math.Max(1, math.Abs(w)) {
						t.Fatalf("output %d[%d]: got %.17g want %.17g", j, i, got[i], w)
					}
				}
			}
		})
	}
}

func TestBatchNormValidation(t *testing.T) {
	for _, op := range []string{"training", "inference", "grad"} {
		for _, name := range []string{"axis", "scalar", "dtype", "vector length", "vector rank", "vector dtype", "gradient shape", "foreign"} {
			if name == "gradient shape" && op != "grad" {
				continue
			}
			t.Run(op+"/"+name, func(t *testing.T) {
				f := (&Backend{}).Builder("bn").Main().(*function)
				param := func(fn compute.Function, n string, s shapes.Shape) compute.Value {
					v, err := fn.Parameter(n, s, nil)
					if err != nil {
						t.Fatal(err)
					}
					return v
				}
				input := shapes.Make(dtypes.Float32, 2, 3)
				axis := 1
				switch name {
				case "axis":
					axis = 2
				case "scalar":
					input = shapes.Make(dtypes.Float32)
				case "dtype":
					input = shapes.Make(dtypes.Int32, 2, 3)
				}
				x := param(f, "x", input)
				feature := shapes.Make(input.DType, 3)
				bad := feature.Clone()
				switch name {
				case "vector length":
					bad.Dimensions = []int{2}
				case "vector rank":
					bad.Dimensions = []int{1, 3}
				case "vector dtype":
					bad.DType = dtypes.Float64
				}
				scale := param(f, "scale", bad)
				if name == "foreign" {
					scale = param((&Backend{}).Builder("other").Main(), "scale", feature)
				}
				offset := param(f, "offset", feature)
				variance := param(f, "variance", feature)
				gradShape := input.Clone()
				if name == "gradient shape" {
					gradShape = shapes.Make(input.DType, 3, 2)
				}
				grad := param(f, "grad", gradShape)
				n := len(f.raw.Ops)
				var err error
				switch op {
				case "training":
					_, _, _, err = f.BatchNormForTraining(x, scale, offset, .1, axis)
				case "inference":
					_, err = f.BatchNormForInference(x, scale, offset, offset, variance, .1, axis)
				case "grad":
					_, _, _, err = f.BatchNormGradient(x, scale, offset, variance, grad, .1, axis)
				}
				if err == nil || len(f.raw.Ops) != n {
					t.Fatal("invalid contract accepted or emitted IR")
				}
			})
		}
	}
}

func TestNativeBatchNormDTypes(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	for _, dtype := range []dtypes.DType{dtypes.Float16, dtypes.BFloat16, dtypes.Float32, dtypes.Float64} {
		t.Run(dtype.String(), func(t *testing.T) {
			e, err := graph.NewExec(b, func(x *graph.Node) []*graph.Node {
				x = graph.ConvertDType(x, dtype)
				s := graph.Ones(x.Graph(), shapes.Make(dtype, 2))
				o := graph.ZerosLike(s)
				y, m, v := graph.InternalBatchNormForTraining(x, s, o, .125, 1)
				dx, ds, do := graph.InternalBatchNormGradient(x, s, m, v, graph.OnesLike(x), .125, 1)
				inference := graph.InternalBatchNormForInference(x, s, o, m, v, .125, 1)
				out := []*graph.Node{y, m, v, dx, ds, do, inference}
				for i, v := range out {
					out[i] = graph.ConvertDType(graph.Reshape(v, -1), dtypes.Float64)
				}
				return out
			})
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			out, err := e.Call([][]float32{{1, 2}, {3, 4}})
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				for _, v := range out {
					v.FinalizeAll()
				}
			}()
			a := 1 / math.Sqrt(1.125)
			want := [][]float64{{-a, -a, a, a}, {2, 3}, {1, 1}, {0, 0, 0, 0}, {0, 0}, {2, 2}, {-a, -a, a, a}}
			tol := 1e-12
			switch dtype {
			case dtypes.Float16:
				tol = 2.0 / 1024
			case dtypes.BFloat16:
				tol = 2.0 / 128
			case dtypes.Float32:
				tol = 2e-6
			}
			for j, v := range out {
				got := v.Value().([]float64)
				if !slices.EqualFunc(got, want[j], func(a, b float64) bool { return !math.IsNaN(a) && math.Abs(a-b) <= tol }) {
					t.Fatalf("output%d got%v want%v", j, got, want[j])
				}
			}
		})
	}
}
