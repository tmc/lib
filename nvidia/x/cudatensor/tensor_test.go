package cudatensor_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudatensor"
)

func TestFluentTensorMath(t *testing.T) {
	g := cudair.NewGraph()

	a := cudatensor.NewInput(g, "a", []int{1024}, cudair.Float32)
	b := cudatensor.NewInput(g, "b", []int{1024}, cudair.Float32)
	c := cudatensor.NewInput(g, "c", []int{1024}, cudair.Float32)

	// (a + b) * c -> relu -> exp
	out := a.Add(b).Mul(c).Relu().Exp()

	if out.Err() != nil {
		t.Fatalf("unexpected error: %v", out.Err())
	}
	if out.Graph() != g {
		t.Error("expected matching graph reference")
	}

	src, err := out.Compile("FusedMathKernel")
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}

	if !strings.Contains(src, "FusedMathKernel") {
		t.Error("expected kernel name in compiled source")
	}
	if !strings.Contains(src, "expf(") {
		t.Error("expected expf in compiled source")
	}
}

func TestTensorFrontendErrorPropagation(t *testing.T) {
	g := cudair.NewGraph()
	a := cudatensor.NewInput(g, "a", []int{2, 3}, cudair.Float32)
	b := cudatensor.NewInput(g, "b", []int{4, 5}, cudair.Float32) // Incompatible for matmul (3 != 4)

	// Matmul should not panic, but return a Tensor carrying an error
	badMatMul := a.MatMul(b)
	if badMatMul.Err() == nil {
		t.Fatal("expected error on mismatched matmul dimensions")
	}

	// Subsequent chained operations should preserve the error without panicking
	chained := badMatMul.Relu().Exp().Add(a)
	if chained.Err() == nil {
		t.Fatal("expected error to propagate through chained operations")
	}

	// Mixed graph detection
	g2 := cudair.NewGraph()
	foreign := cudatensor.NewInput(g2, "foreign", []int{2, 3}, cudair.Float32)
	mixed := a.Add(foreign)
	if mixed.Err() == nil {
		t.Fatal("expected error when combining tensors from different graphs")
	}

	// Compile should fail gracefully with the accumulated error
	_, err := chained.Compile("FailKernel")
	if err == nil {
		t.Fatal("expected Compile to fail on errored tensor")
	}
}

func TestTensorAuthoritativeOperations(t *testing.T) {
	g := cudair.NewGraph()
	x := cudatensor.NewInput(g, "X", []int{4, 8}, cudair.Float32)
	w := cudatensor.NewInput(g, "W", []int{8, 16}, cudair.Float32)

	y := x.MatMul(w).Gelu().Log()
	if y.Err() != nil {
		t.Fatalf("unexpected error: %v", y.Err())
	}

	// Reduce
	r := y.ReduceSum(1) // [4]
	if r.Err() != nil {
		t.Fatalf("ReduceSum failed: %v", r.Err())
	}
	if len(r.Shape()) != 1 || r.Shape()[0] != 4 {
		t.Errorf("expected shape [4], got %v", r.Shape())
	}

	// Reshape
	reshaped := y.Reshape([]int{2, 32})
	if reshaped.Err() != nil {
		t.Fatalf("Reshape failed: %v", reshaped.Err())
	}

	// Transpose
	transposed := reshaped.Transpose([]int{1, 0}) // [32, 2]
	if transposed.Err() != nil {
		t.Fatalf("Transpose failed: %v", transposed.Err())
	}
	if transposed.Shape()[0] != 32 || transposed.Shape()[1] != 2 {
		t.Errorf("expected [32, 2], got %v", transposed.Shape())
	}

	// Output marking
	marked := transposed.MarkOutput()
	if marked.Err() != nil {
		t.Fatalf("MarkOutput failed: %v", marked.Err())
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if len(snap.Outputs()) != 1 || snap.Outputs()[0].ID != marked.Node().ID {
		t.Errorf("expected marked output in snapshot")
	}
}
