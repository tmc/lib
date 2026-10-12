package cudaref_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

func TestEvaluateSnapshotAllReturnsEveryNodeByID(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	c := g.AddConstant("c", []int{2}, cudair.Float32, []float32{3, 4})
	sum := g.AddBinary("add", x, c)
	if err := g.SetOutputs(sum); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}

	input := []float32{1, 2}
	values, err := cudaref.EvaluateSnapshotAll(snap, map[string][]float32{"x": input})
	if err != nil {
		t.Fatal(err)
	}
	input[0] = 99
	want := map[int][]float32{
		x.ID:   {1, 2},
		c.ID:   {3, 4},
		sum.ID: {4, 6},
	}
	if !reflect.DeepEqual(values, want) {
		t.Fatalf("EvaluateSnapshotAll() = %v, want %v", values, want)
	}

	outputs, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"x": {1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(outputs[sum.Name], want[sum.ID]) {
		t.Fatalf("EvaluateSnapshot() = %v, want %v", outputs[sum.Name], want[sum.ID])
	}
}

func ExampleEvaluateSnapshotAll() {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	neg := g.AddUnary("neg", x)
	snap, err := g.Snapshot()
	if err != nil {
		fmt.Println(err)
		return
	}
	values, err := cudaref.EvaluateSnapshotAll(snap, map[string][]float32{"x": {2, 3}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(values[neg.ID])
	// Output: [-2 -3]
}
