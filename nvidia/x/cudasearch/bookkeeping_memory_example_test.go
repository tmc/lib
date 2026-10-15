package cudasearch_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleSearchResult_Close_bookkeeping() {
	graph := egraph.New()
	root := graph.Add(egraph.NewLeaf("input"))
	state := egraph.NewSearchState(graph, map[string]egraph.Id{"root": root})
	search := cudasearch.NewBeamSearch(cudasearch.BeamConfig{Budget: cudasearch.SearchBudget{
		MaxStates: 1, MaxMemoryBytes: 64 << 10,
	}})
	result, err := search.Search(context.Background(), state, nil, &cudasearch.MockEvaluator{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("winner admitted:", result.Winner != nil)
	result.Close() // Releases returned collections, branches and extraction.
	_, ok := state.CanonicalRoot("root")
	fmt.Println("caller root retained:", ok)
	// Output:
	// winner admitted: true
	// caller root retained: true
}
