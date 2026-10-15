package cudasearch_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type reservedEvaluator struct{}

func (reservedEvaluator) Evaluate(ctx context.Context, _ *egraph.SearchState, _ egraph.Id) (*cudasearch.CandidatePlan, error) {
	release, err := cudasearch.ReserveSearchMemory(ctx, 8)
	if err != nil {
		return nil, err
	}
	data := make([]byte, 8)
	defer release() // data is not retained by the returned candidate.
	return &cudasearch.CandidatePlan{ID: fmt.Sprint("reserved ", len(data)), Cost: 1, Validated: true}, nil
}

func ExampleReserveSearchMemory() {
	graph := egraph.New()
	root := graph.Add(egraph.ENode{Op: "x"})
	state := egraph.NewSearchState(graph, map[string]egraph.Id{"root": root})
	search := cudasearch.NewBeamSearch(cudasearch.BeamConfig{Budget: cudasearch.SearchBudget{
		MaxStates: 1, MaxMemoryBytes: 32 << 10, CallbackMemoryBytes: 4096,
	}})
	result, err := search.Search(context.Background(), state, nil, reservedEvaluator{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.Winner.ID)
	// Output: reserved 8
}
