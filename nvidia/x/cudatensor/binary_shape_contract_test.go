package cudatensor_test

import (
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudatensor"
)

func TestBinaryShapeContract(t *testing.T) {
	matrix := []float32{1, 2, 4, 1, 2, 4}
	twos := []float32{2, 2, 2, 2, 2, 2}
	for _, op := range []struct {
		name        string
		apply       func(cudatensor.Tensor, cudatensor.Tensor) cudatensor.Tensor
		left, right []float32
	}{
		{"add", cudatensor.Tensor.Add, []float32{3, 4, 6, 3, 4, 6}, []float32{3, 4, 6, 3, 4, 6}},
		{"sub", cudatensor.Tensor.Sub, []float32{1, 0, -2, 1, 0, -2}, []float32{-1, 0, 2, -1, 0, 2}},
		{"mul", cudatensor.Tensor.Mul, []float32{2, 4, 8, 2, 4, 8}, []float32{2, 4, 8, 2, 4, 8}},
		{"div", cudatensor.Tensor.Div, []float32{2, 1, 0.5, 2, 1, 0.5}, []float32{0.5, 1, 2, 0.5, 1, 2}},
	} {
		for _, shape := range [][]int{nil, {1}, {1, 1}} {
			for _, left := range []bool{true, false} {
				t.Run(fmt.Sprintf("%s/scalar_rank_%d/left_%t", op.name, len(shape), left), func(t *testing.T) {
					g := cudair.NewGraph()
					scalar := cudatensor.NewInput(g, "scalar", shape, cudair.Float32)
					x := cudatensor.NewInput(g, "matrix", []int{2, 3}, cudair.Float32)
					var out cudatensor.Tensor
					want := op.left
					if left {
						out = op.apply(scalar, x)
					} else {
						out, want = op.apply(x, scalar), op.right
					}
					checkBinaryShapeResult(t, out, map[string][]float32{"scalar": {2}, "matrix": matrix}, want)
				})
			}
		}
		t.Run(op.name+"/equal_shape", func(t *testing.T) {
			g := cudair.NewGraph()
			x := cudatensor.NewInput(g, "matrix", []int{2, 3}, cudair.Float32)
			y := cudatensor.NewInput(g, "twos", []int{2, 3}, cudair.Float32)
			checkBinaryShapeResult(t, op.apply(x, y), map[string][]float32{"matrix": matrix, "twos": twos}, op.right)
		})
	}
}

func checkBinaryShapeResult(t *testing.T, out cudatensor.Tensor, inputs map[string][]float32, want []float32) {
	t.Helper()
	if err := out.Err(); err != nil {
		t.Fatal(err)
	}
	wantShape := []int{2, 3}
	if !slices.Equal(out.Shape(), wantShape) || !slices.Equal(out.Node().Shape, wantShape) {
		t.Fatalf("frontend shape %v, IR shape %v, want %v", out.Shape(), out.Node().Shape, wantShape)
	}
	shape := out.Shape()
	shape[0] = 99
	if !slices.Equal(out.Shape(), wantShape) || !slices.Equal(out.Node().Shape, wantShape) {
		t.Fatal("Shape accessor changed frontend or IR metadata")
	}
	if err := out.Graph().SetOutputs(out.Node()); err != nil {
		t.Fatal(err)
	}
	snapshot, err := out.Graph().Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	outputs, err := cudaref.EvaluateSnapshot(snapshot, inputs)
	if err != nil {
		t.Fatal(err)
	}
	if got := outputs[out.Node().Name]; !slices.Equal(got, want) {
		t.Fatalf("independent output = %v, want %v", got, want)
	}
}

func TestBinaryShapeChainedConsumers(t *testing.T) {
	g := cudair.NewGraph()
	scalar := cudatensor.NewConstant(g, "scalar", []int{1}, cudair.Float32, []float32{2})
	x := cudatensor.NewInput(g, "x", []int{2, 3}, cudair.Float32)
	w := cudatensor.NewInput(g, "w", []int{3, 2}, cudair.Float32)
	y := scalar.Add(x)
	product := y.MatMul(w)
	if err := product.Err(); err != nil {
		t.Fatalf("valid scalar-left matrix product: %v", err)
	}
	transpose := product.Transpose([]int{1, 0})
	sum := transpose.ReduceSum(1)
	maximum := transpose.ReduceMax(0)
	tests := []struct {
		name   string
		tensor cudatensor.Tensor
		shape  []int
		want   []float32
	}{
		{"scalar_add", y, []int{2, 3}, []float32{3, 4, 5, 6, 7, 8}},
		{"matmul", product, []int{2, 2}, []float32{8, 9, 14, 15}},
		{"transpose", transpose, []int{2, 2}, []float32{8, 14, 9, 15}},
		{"sum", sum, []int{2}, []float32{22, 24}},
		{"maximum", maximum, []int{2}, []float32{9, 15}},
	}
	var nodes []*cudair.Node
	for _, tt := range tests {
		if err := tt.tensor.Err(); err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		if !slices.Equal(tt.tensor.Shape(), tt.shape) || !slices.Equal(tt.tensor.Node().Shape, tt.shape) {
			t.Fatalf("%s: frontend shape %v, IR shape %v, want %v", tt.name, tt.tensor.Shape(), tt.tensor.Node().Shape, tt.shape)
		}
		nodes = append(nodes, tt.tensor.Node())
	}
	if err := g.SetOutputs(nodes...); err != nil {
		t.Fatal(err)
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	outputs, err := cudaref.EvaluateSnapshot(snapshot, map[string][]float32{
		"x": {1, 2, 3, 4, 5, 6},
		"w": {1, 0, 0, 1, 1, 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range tests {
		if got := outputs[tt.tensor.Node().Name]; !slices.Equal(got, tt.want) {
			t.Errorf("%s: independent output = %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestBinaryShapeInvalidOperands(t *testing.T) {
	for _, op := range []struct {
		name  string
		apply func(cudatensor.Tensor, cudatensor.Tensor) cudatensor.Tensor
	}{
		{"add", cudatensor.Tensor.Add},
		{"sub", cudatensor.Tensor.Sub},
		{"mul", cudatensor.Tensor.Mul},
		{"div", cudatensor.Tensor.Div},
	} {
		for _, shape := range [][]int{{2, 4}, {3}, {1, 3}} {
			t.Run(fmt.Sprintf("%s/%v", op.name, shape), func(t *testing.T) {
				g := cudair.NewGraph()
				x := cudatensor.NewInput(g, "x", []int{2, 3}, cudair.Float32)
				y := cudatensor.NewInput(g, "y", shape, cudair.Float32)
				out := op.apply(x, y)
				if !errors.Is(out.Err(), cudair.ErrShapeMismatch) {
					t.Fatalf("invalid shape %v: %v, want ErrShapeMismatch", shape, out.Err())
				}
			})
		}
	}
}
