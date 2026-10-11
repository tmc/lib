package cudafuse

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// transposeOracle explicitly reconstructs coordinates and flattens them in
// input-axis order. It does not use the emitter's stride expression.
func transposeOracle(input []float32, shape, perm []int) []float32 {
	outputShape := make([]int, len(perm))
	for axis, sourceAxis := range perm {
		outputShape[axis] = shape[sourceAxis]
	}
	output := make([]float32, len(input))
	for linear := range output {
		remaining := linear
		coordinates := make([]int, len(shape))
		for axis := len(outputShape) - 1; axis >= 0; axis-- {
			coordinates[perm[axis]] = remaining % outputShape[axis]
			remaining /= outputShape[axis]
		}
		sourceIndex := 0
		for axis, extent := range shape {
			sourceIndex = sourceIndex*extent + coordinates[axis]
		}
		output[linear] = input[sourceIndex]
	}
	return output
}

func TestIndexedLazyTransposeValues(t *testing.T) {
	for _, tc := range []struct {
		name  string
		shape []int
		perm  []int
		build func(*LazyArray, *LazyArray) *LazyArray
		want  func([]float32, []float32, []int, []int) []float32
	}{
		{"rank2", []int{2, 3}, []int{1, 0}, func(x, _ *LazyArray) *LazyArray { return x.Transpose([]int{1, 0}) }, transposeOracleWithUnusedInput},
		{"rank3", []int{2, 3, 4}, []int{2, 0, 1}, func(x, _ *LazyArray) *LazyArray { return x.Transpose([]int{2, 0, 1}) }, transposeOracleWithUnusedInput},
		{"identity", []int{2, 3}, []int{0, 1}, func(x, _ *LazyArray) *LazyArray { return x.Transpose([]int{0, 1}) }, transposeOracleWithUnusedInput},
		{"scalar", nil, nil, func(x, _ *LazyArray) *LazyArray { return x.Transpose(nil) }, transposeOracleWithUnusedInput},
		{"composed", []int{2, 3}, []int{1, 0}, func(x, _ *LazyArray) *LazyArray { return x.Transpose([]int{1, 0}).Transpose([]int{1, 0}) }, func(x, _ []float32, _, _ []int) []float32 { return slices.Clone(x) }},
		{"before binary", []int{2, 3}, []int{1, 0}, func(x, y *LazyArray) *LazyArray { return x.Map("neg").Add(y).Transpose([]int{1, 0}) }, func(x, y []float32, shape, perm []int) []float32 {
			values := make([]float32, len(x))
			for i := range values {
				values[i] = -x[i] + y[i]
			}
			return transposeOracle(values, shape, perm)
		}},
		{"after binary", []int{2, 3}, []int{1, 0}, func(x, y *LazyArray) *LazyArray { return x.Transpose([]int{1, 0}).Add(y.Reshape([]int{3, 2})) }, func(x, y []float32, shape, perm []int) []float32 {
			values := transposeOracle(x, shape, perm)
			for i := range values {
				values[i] += y[i]
			}
			return values
		}},
		{"reshape", []int{2, 3}, []int{1, 0}, func(x, _ *LazyArray) *LazyArray {
			return x.Reshape([]int{3, 2}).Transpose([]int{1, 0}).Reshape([]int{6})
		}, func(x, _ []float32, _, perm []int) []float32 { return transposeOracle(x, []int{3, 2}, perm) }},
		{"distinct indices", []int{2, 3}, []int{1, 0}, func(x, _ *LazyArray) *LazyArray {
			return x.Transpose([]int{1, 0}).Reshape([]int{6}).Add(x.Reshape([]int{6}))
		}, func(x, _ []float32, shape, perm []int) []float32 {
			values := transposeOracle(x, shape, perm)
			for i := range values {
				values[i] += x[i]
			}
			return values
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := NewLazyContext()
			x, y := NewLazyArray(ctx, tc.shape, cudair.Float32), NewLazyArray(ctx, tc.shape, cudair.Float32)
			root := tc.build(x, y)
			plan, err := root.Materialize()
			if err != nil {
				t.Fatal(err)
			}
			nodes, n, err := executionNodes(plan)
			if err != nil {
				t.Fatal(err)
			}
			if len(plan.Regions) != 1 || ctx.Tracker.Snapshot().Allocations != 1 {
				t.Fatal("transpose caused intermediate materialization")
			}
			a, b := make([]float32, n), make([]float32, n)
			for i := range a {
				a[i], b[i] = float32(i-3), float32(10+2*i)
			}
			bindings := map[string][]float32{x.Node.Name(): a, y.Node.Name(): b}
			want := tc.want(a, b, tc.shape, tc.perm)
			runIndexedSource(t, *plan.Regions[0].Kernel, bindings, [][]float32{want})
			fused, err := executionSource(nodes, root.Node, "indexed_execution", false)
			if err != nil {
				t.Fatal(err)
			}
			runIndexedSource(t, fused, bindings, [][]float32{want})
		})
	}
}

func transposeOracleWithUnusedInput(x, _ []float32, shape, perm []int) []float32 {
	return transposeOracle(x, shape, perm)
}

func runIndexedSource(t *testing.T, kernel FusedKernel, bindings map[string][]float32, outputs [][]float32) {
	t.Helper()
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("host emitted-source check requires clang++")
	}
	dir := t.TempDir()
	header := "#define __global__\nstruct dim {unsigned int x=1,y=1,z=1;};\nextern dim blockDim,blockIdx,threadIdx,gridDim;\n"
	if err := os.WriteFile(filepath.Join(dir, "cuda_runtime.h"), []byte(header), 0600); err != nil {
		t.Fatal(err)
	}
	var main strings.Builder
	fmt.Fprintln(&main, "dim blockDim,blockIdx,threadIdx,gridDim;\nint main() {")
	var args []string
	for _, name := range kernel.Inputs {
		data, ok := bindings[name]
		if !ok {
			t.Fatalf("no fixture for %s", name)
		}
		fmt.Fprintf(&main, "float %s[] = {%s};\n", name, floatValues(data))
		args = append(args, name)
	}
	n := len(outputs[0])
	for i := range outputs {
		fmt.Fprintf(&main, "float output%d[%d]; for (int j=0;j<%d;j++) output%d[j]=-999;\n", i, n+2, n+2, i)
		args = append(args, fmt.Sprintf("output%d+1", i))
	}
	args = append(args, fmt.Sprint(n))
	fmt.Fprintf(&main, "blockDim.x=3;gridDim.x=2;for(blockIdx.x=0;blockIdx.x<2;blockIdx.x++) for(threadIdx.x=0;threadIdx.x<3;threadIdx.x++) %s(%s);\n", kernel.Name, strings.Join(args, ","))
	for i, want := range outputs {
		fmt.Fprintf(&main, "float want%d[] = {%s}; for(int j=0;j<%d;j++) if(!isfinite(output%d[j+1]) || fabsf(output%d[j+1]-want%d[j]) > 1e-6f*(1.0f+fabsf(want%d[j]))) return 1;\n", i, floatValues(want), n, i, i, i, i)
		fmt.Fprintf(&main, "if(output%d[0]!=-999 || output%d[%d]!=-999) return 2;\n", i, i, n+1)
	}
	// A wrong domain and non-1D launch must leave all buffers alone.
	fmt.Fprintln(&main, "output0[1]=-998;")
	wrong := slices.Clone(args)
	wrong[len(wrong)-1] = fmt.Sprint(n + 1)
	fmt.Fprintf(&main, "%s(%s); if(output0[1]!=-998) return 3;\n", kernel.Name, strings.Join(wrong, ","))
	fmt.Fprintf(&main, "gridDim.y=2; %s(%s); if(output0[1]!=-998) return 4;\nreturn 0; }\n", kernel.Name, strings.Join(args, ","))
	path, binary := filepath.Join(dir, "indexed.cc"), filepath.Join(dir, "indexed")
	if err := os.WriteFile(path, []byte("#include <math.h>\n"+kernel.Source+main.String()), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if output, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O1", "-I", dir, path, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile emitted source: %v\n%s", err, output)
	}
	ctx, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelRun()
	if output, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("emitted source values/guards: %v\n%s", err, output)
	}
}

func floatValues(values []float32) string {
	var text []string
	for _, value := range values {
		number := strconv.FormatFloat(float64(value), 'g', -1, 32)
		if !strings.ContainsAny(number, ".eE") {
			number += ".0"
		}
		text = append(text, number+"f")
	}
	return strings.Join(text, ",")
}

func TestIndexedLazyOperators(t *testing.T) {
	for _, tc := range []struct {
		op     string
		binary bool
		value  func(float64, float64) float64
	}{
		{"relu", false, func(a, _ float64) float64 { return math.Max(a, 0) }},
		{"silu", false, func(a, _ float64) float64 { return a / (1 + math.Exp(-a)) }},
		{"exp", false, func(a, _ float64) float64 { return math.Exp(a) }},
		{"sqrt", false, func(a, _ float64) float64 { return math.Sqrt(a) }},
		{"neg", false, func(a, _ float64) float64 { return -a }},
		{"add", true, func(a, b float64) float64 { return a + b }},
		{"sub", true, func(a, b float64) float64 { return a - b }},
		{"mul", true, func(a, b float64) float64 { return a * b }},
		{"div", true, func(a, b float64) float64 { return a / b }},
	} {
		t.Run(tc.op, func(t *testing.T) {
			ctx := NewLazyContext()
			x := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
			y := NewLazyArray(ctx, []int{3, 2}, cudair.Float32)
			view := x.Transpose([]int{1, 0})
			root := view.Map(tc.op)
			inputs := []*LazyNode{x.Node}
			if tc.binary {
				root = view.Map2(tc.op, y)
				inputs = append(inputs, y.Node)
			}
			kernel, err := FuseLazy(inputs, []*LazyNode{root.Node}, "indexed_operator")
			if err != nil {
				t.Fatal(err)
			}
			a := []float32{-1.25, -0.5, 0.25, 0.75, 1.25, 2}
			if tc.op == "sqrt" {
				a = []float32{0.25, 0.5, 1, 2, 3, 4}
			}
			b := []float32{1.5, 2.5, 3.5, 4.5, 5.5, 6.5}
			want := transposeOracle(a, x.Shape, []int{1, 0})
			for i := range want {
				want[i] = float32(tc.value(float64(want[i]), float64(b[i])))
			}
			runIndexedSource(t, kernel, map[string][]float32{x.Node.Name(): a, y.Node.Name(): b}, [][]float32{want})
		})
	}
}

func TestIndexedLazyMultipleOutputs(t *testing.T) {
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
	root := x.Transpose([]int{1, 0}).Map("neg")
	kernel, err := compileIndexedLazyRegion([]*LazyNode{x.Node}, []*LazyNode{root.Node, x.Node}, "indexed_outputs")
	if err != nil {
		t.Fatal(err)
	}
	values := []float32{-3, -2, -1, 0, 1, 2}
	want := transposeOracle(values, []int{2, 3}, []int{1, 0})
	for i := range want {
		want[i] = -want[i]
	}
	runIndexedSource(t, kernel, map[string][]float32{x.Node.Name(): values}, [][]float32{want, values})
}

func TestIndexedLazyContradictoryPointwiseShapes(t *testing.T) {
	for _, op := range []string{"neg", "add"} {
		t.Run(op, func(t *testing.T) {
			ctx := NewLazyContext()
			x := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
			root := x.Map(op)
			if op == "add" {
				root = x.Add(x)
			}
			root.Node.Shape = []int{3, 2}
			if _, err := FuseLazy([]*LazyNode{x.Node}, []*LazyNode{root.Node}, "wrong_pointwise_shape"); err == nil {
				t.Fatal("equal element counts concealed contradictory pointwise shape")
			}
		})
	}
}

func TestIndexedLazyMalformedViews(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*LazyNode)
	}{
		{"missing permutation", func(n *LazyNode) { delete(n.Attrs, "permutation") }},
		{"wrong permutation type", func(n *LazyNode) { n.Attrs["permutation"] = []string{"N", "M"} }},
		{"duplicate axis", func(n *LazyNode) { n.Attrs["permutation"] = []int{0, 0} }},
		{"negative axis", func(n *LazyNode) { n.Attrs["permutation"] = []int{-1, 0} }},
		{"changed output shape", func(n *LazyNode) { n.Shape = []int{2, 3} }},
		{"extra attributes", func(n *LazyNode) { n.Attrs["other"] = 1 }},
		{"negative after zero", func(n *LazyNode) { n.Shape = []int{0, -1} }},
		{"unbounded domain", func(n *LazyNode) { n.Inputs[0].Shape = []int{1 << 30, 3}; n.Shape = []int{3, 1 << 30} }},
		{"wrong dtype", func(n *LazyNode) { n.DType = cudair.Float16 }},
		{"cycle", func(n *LazyNode) { n.Inputs[0] = n; n.Shape = []int{3, 3} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := NewLazyContext()
			x := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
			root := x.Transpose([]int{1, 0})
			tc.mutate(root.Node)
			if _, err := compileIndexedLazyRegion([]*LazyNode{x.Node}, []*LazyNode{root.Node}, "invalid_indexed"); err == nil {
				t.Fatal("malformed indexed region admitted")
			}
			if ctx.Tracker.Snapshot().Allocations != 0 {
				t.Fatal("invalid graph recorded allocation")
			}
		})
	}
}

func TestIndexedLazyEmptyDomainNoDeviceWork(t *testing.T) {
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{2, 0, 3}, cudair.Float32)
	root := x.Transpose([]int{2, 0, 1})
	plan, err := root.Materialize()
	if err != nil {
		t.Fatal(err)
	}
	e, err := plan.Execute(context.Background(), nil, map[string]ExecutionBinding{x.Node.Name(): {Shape: x.Shape, DType: x.DType}})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	output, err := e.Readback(context.Background())
	if err != nil || len(output) != 0 || e.Stats() != (ExecutionStats{}) || len(e.Receipts()) != 0 {
		t.Fatal("empty transpose invented device work", err)
	}
}

func TestIndexedLazyEagerPrimitiveSource(t *testing.T) {
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
	negative := x.Map("neg")
	root := negative.Transpose([]int{1, 0})
	plan, err := root.Materialize()
	if err != nil {
		t.Fatal(err)
	}
	nodes, _, err := executionNodes(plan)
	if err != nil {
		t.Fatal(err)
	}
	values := []float32{-3, -2, -1, 0, 1, 2}
	primitive, err := executionSource(nodes, negative.Node, "eager_negative", true)
	if err != nil {
		t.Fatal(err)
	}
	negativeValues := slices.Clone(values)
	for i := range negativeValues {
		negativeValues[i] = -negativeValues[i]
	}
	runIndexedSource(t, primitive, map[string][]float32{x.Node.Name(): values}, [][]float32{negativeValues})
	transpose, err := executionSource(nodes, root.Node, "eager_transpose", true)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(transpose.Inputs, []string{negative.Node.Name()}) {
		t.Fatal("eager transpose bypassed materialized operand")
	}
	runIndexedSource(t, transpose, map[string][]float32{negative.Node.Name(): negativeValues}, [][]float32{transposeOracle(negativeValues, x.Shape, []int{1, 0})})
}
