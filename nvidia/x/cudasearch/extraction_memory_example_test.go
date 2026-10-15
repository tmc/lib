package cudasearch_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleCandidatePlan_Close() {
	g := egraph.New()
	root := g.Add(egraph.NewNode("leaf"))
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	eval := &cudasearch.MockEvaluator{Scoring: cudasearch.DefaultScoringModel("epoch")}
	candidate, err := eval.Evaluate(context.Background(), state, root)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(candidate.Expr.Op)
	candidate.Close()
	candidate.Close()
	fmt.Println(candidate.Expr == nil)
	// Output:
	// leaf
	// true
}

func ExampleSearchResult_Close() {
	g := egraph.New()
	root := g.Add(egraph.NewNode("leaf"))
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	engine := cudasearch.NewBeamSearch(cudasearch.BeamConfig{Budget: cudasearch.SearchBudget{MaxMemoryBytes: 1 << 20, MaxStates: 1}})
	result, err := engine.Search(context.Background(), state, nil, &cudasearch.MockEvaluator{Scoring: cudasearch.DefaultScoringModel("epoch")})
	if err != nil {
		result.Close()
		fmt.Println(err)
		return
	}
	fmt.Println(result.Winner.Expr.Op, result.AccountedBytes > 0)
	result.Close()
	result.Close()
	fmt.Println(result.Winner.Expr == nil)
	// Output:
	// leaf true
	// true
}
