package cudasearch_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleSearchResult_Close_setup() {
	g := egraph.New()
	root := g.Add(egraph.NewLeaf("input"))
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	search := cudasearch.NewBeamSearch(cudasearch.BeamConfig{Budget: cudasearch.SearchBudget{MaxStates: 1, MaxMemoryBytes: 1 << 20}})
	result, err := search.Search(context.Background(), state, nil, &cudasearch.MockEvaluator{})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer result.Close() // Keeps the owned policy data valid until this returns.
	fmt.Println(result.PolicyID.Strategy, result.PolicyID.Options["contract_version"], result.AccountedBytes > 0)
	// Output: beam 1 true
}
