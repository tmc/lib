package cudafuse_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func ExampleMaterializedPlan_Execute() {
	ctx := cudafuse.NewLazyContext()
	a := cudafuse.NewLazyArray(ctx, []int{0}, cudair.Float32)
	root := a.Map("relu")
	plan, err := cudafuse.CompileDeferredGraph(ctx, root.Node)
	if err != nil {
		fmt.Println(err)
		return
	}
	// Empty bindings need no device allocation or stream. For a nonempty graph,
	// bind each input name to actual cudamem storage on the supplied stream.
	e, err := plan.Execute(context.Background(), nil, map[string]cudafuse.ExecutionBinding{
		a.Node.Name(): {Shape: []int{0}, DType: cudair.Float32},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer e.Close()
	out, err := e.Readback(context.Background())
	fmt.Println(len(out), e.Stats(), len(e.Receipts()), err)
	// Output: 0 {0 0 0} 0 <nil>
}

func ExampleMaterializedPlan_ExecuteEager() {
	var plan *cudafuse.MaterializedPlan
	_, err := plan.ExecuteEager(context.Background(), nil, nil)
	fmt.Println(err != nil)
	// Output: true
}

func ExampleExecution_Wait() {
	var e *cudafuse.Execution
	fmt.Println(e.Wait(context.Background()) != nil)
	// Output: true
}

func ExampleExecution_Readback() {
	var e *cudafuse.Execution
	_, err := e.Readback(context.Background())
	fmt.Println(err != nil)
	// Output: true
}

func ExampleExecution_Close() {
	var e *cudafuse.Execution
	fmt.Println(e.Close())
	// Output: <nil>
}

func ExampleExecution_Stats() {
	var e *cudafuse.Execution
	fmt.Println(e.Stats())
	// Output: {0 0 0}
}

func ExampleExecution_Receipts() {
	var e *cudafuse.Execution
	fmt.Println(len(e.Receipts()))
	// Output: 0
}
