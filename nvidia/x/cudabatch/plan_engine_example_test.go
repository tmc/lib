package cudabatch_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

func ExampleCompilePlanEngine() {
	graph := cudair.NewGraph()
	input := graph.AddInput("x", []int{1, 1}, cudair.Float32)
	output := graph.AddMatMul(input, input)
	if err := graph.SetOutputs(output); err != nil {
		return
	}
	snapshot, err := graph.Snapshot()
	if err != nil {
		return
	}
	bind := func(context.Context, *cudabatch.ExecutionStep) (*cudaplan.Bindings, func(), error) {
		// A real binder sets typed inputs and outputs and retains their allocations.
		return cudaplan.NewBindings(), nil, nil
	}
	sample := func(context.Context, *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
		return nil, nil, nil
	}
	engine, err := cudabatch.CompilePlanEngine(snapshot, nil, bind, sample)
	if err != nil {
		return
	}
	defer engine.Close()
	fmt.Println(engine.StepDone(1))
	fmt.Println(engine.WaitStep(context.Background(), 1))
	// Output:
	// false
	// cudabatch: unknown execution step
}
