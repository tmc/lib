package cudaplan_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

func ExampleSelectBackend() {
	decision, err := cudaplan.SelectBackend(cudaplan.BackendRequest{Kind: cudair.KindUnary})
	fmt.Println(decision.Backend, err)
	// Output: Launch <nil>
}

func ExampleCompletion_ExecutedSteps() {
	// An absent completion has no submitted dispatches. A real invocation's
	// receipt becomes available after Execute returns; Wait observes completion.
	var completion *cudaplan.Completion
	fmt.Println(len(completion.ExecutedSteps()))
	// Output: 0
}

func ExampleWithFiniteMega() {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{4}, cudair.Float32)
	y := g.AddUnary("neg", x)
	if err := g.SetOutputs(y); err != nil {
		panic(err)
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		panic(err)
	}
	// Without WithDevice this compiles a checked descriptor only; Execute
	// requires a real device and stream.
	plan, err := cudaplan.Compile(snapshot, cudaplan.WithFiniteMega(true))
	if err != nil {
		panic(err)
	}
	defer plan.Close()
	fmt.Println(plan.Steps()[0].Backend)
	// Output: Mega
}
