package cudasearch_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func Example() {
	// 1. Construct initial e-graph with candidate expressions
	g := egraph.New()
	x := g.Add(egraph.ENode{Op: "x"})
	w := g.Add(egraph.ENode{Op: "w"})
	matmul := g.Add(egraph.ENode{Op: "matmul", Children: []egraph.Id{x, w}})

	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": matmul})

	// 2. Define rewrite rules
	rules := []*egraph.Rewrite{
		egraph.MustRewrite("matmul-to-fused", "(matmul ?x ?w)", "(fused_gemv_pipeline ?x ?w)"),
	}

	// 3. Configure workload bindings
	shape, _ := cudashape.New(1, 2048, 4096)
	bindings := cudasearch.WorkloadBindings{
		Name:            "llama_prefill_gemv",
		Shape:           shape,
		DType:           cudair.Float16,
		NumericalPolicy: cudasearch.NumericalPolicyExact,
	}

	// 4. Initialize search engine and evaluator
	beam := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
		BeamWidth: 2,
		Budget: cudasearch.SearchBudget{
			MaxStates:      20,
			MaxDepth:       4,
			MaxActions:     10,
			MaxEvaluations: 20,
		},
		RootName: "root",
	})

	eval := &cudasearch.MockEvaluator{
		CostModel: egraph.FusedMegakernelCost{},
		Scoring:   cudasearch.DefaultScoringModel("epoch-v1"),
		Workload:  bindings,
	}

	res, err := beam.Search(context.Background(), state, rules, eval)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Strategy: %s\n", res.Strategy)
	fmt.Printf("Winner: %s\n", res.Winner.Expr.Op)
	fmt.Printf("Evaluations: %d\n", res.Evaluations)

	// Output:
	// Strategy: beam
	// Winner: fused_gemv_pipeline
	// Evaluations: 2
}
