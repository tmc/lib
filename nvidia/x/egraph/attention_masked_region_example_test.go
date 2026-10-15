package egraph_test

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleNewCausalAttentionRowMask() {
	mask, err := egraph.NewCausalAttentionRowMask(1, 4)
	if err != nil {
		panic(err)
	}
	fmt.Println(mask.Kind(), mask.Dense())
	// Output: causal [false false true true]
}

func ExampleNewPrefixAttentionRowMask() {
	mask, err := egraph.NewPrefixAttentionRowMask(2, 4)
	if err != nil {
		panic(err)
	}
	fmt.Println(mask.Kind(), mask.Dense())
	// Output: prefix [false false true true]
}

func ExampleNewTreeAttentionRowMask() {
	mask, err := egraph.NewTreeAttentionRowMask([]int{-1, 0, 0, 1}, 3, 1)
	if err != nil {
		panic(err)
	}
	query, prefix, parents := mask.Definition()
	fmt.Println(mask.Kind(), query, prefix, parents)
	fmt.Println(mask.Dense())
	// Output:
	// tree-prefix 3 1 [-1 0 0 1]
	// [false false false true false]
}

func ExampleNewCheckedMaskedOnlineAttentionCandidate() {
	g := cudair.NewGraph()
	scores := g.AddInput("scores", []int{3}, cudair.Float32)
	values := g.AddInput("values", []int{3, 1}, cudair.Float32)
	e := g.AddUnary("exp", g.AddBinary("sub", scores, g.AddReduce("max", scores, 0)))
	p := g.AddBinary("div", e, g.AddReduce("sum", e, 0))
	out := g.AddMatMul(g.AddReshape(p, []int{1, 3}), values)
	if err := g.SetOutputs(out); err != nil {
		panic(err)
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		panic(err)
	}
	mask, err := egraph.NewCausalAttentionRowMask(1, 3)
	if err != nil {
		panic(err)
	}
	// The graph's exact input already encodes exclusion; the constructor
	// verifies this binding rather than masking an unmasked computation.
	inputs := map[string][]float32{"scores": {0, 0, float32(math.Inf(-1))}, "values": {3, -1, 999}}
	candidate, err := egraph.NewCheckedMaskedOnlineAttentionCandidate(snapshot, inputs, mask, 1, egraph.AttentionCandidatePolicy{})
	if err != nil {
		panic(err)
	}
	fmt.Println(candidate.Output(), candidate.Reference())
	fmt.Println(candidate.Derivation().Mask.Dense(), len(candidate.Derivation().Tiles))
	// Output:
	// [1] [1]
	// [false false true] 3
}
