package cudasearch_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleSearchMemory_bounded() {
	g := egraph.New()
	root := g.Add(egraph.ENode{Op: "input"})
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	memory := &cudasearch.SearchMemory{Family: cudatrial.MemoryBandwidthLimited, TargetArch: "sm_90"}
	budget := cudasearch.DefaultBudget()
	budget.MaxMemoryBytes, budget.MaxStates = 1<<20, 1
	r, err := cudasearch.NewBeamSearch(cudasearch.BeamConfig{RootName: "root", Budget: budget, Memory: memory}).Search(context.Background(), state, nil, &cudasearch.MockEvaluator{})
	if err != nil {
		panic(err)
	}
	fmt.Println(r.MemoryDecision.State.PendingHardware, r.MemoryDecision.ID != "")
	r.Close() // Releases the frozen decision with the other result owners.
	// Output: true true
}
