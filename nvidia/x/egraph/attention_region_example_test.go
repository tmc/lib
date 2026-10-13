package egraph_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleNewCheckedOnlineAttentionCandidate() {
	g := cudair.NewGraph()
	scores := g.AddInput("scores", []int{2}, cudair.Float32)
	values := g.AddInput("values", []int{2, 1}, cudair.Float32)
	e := g.AddUnary("exp", g.AddBinary("sub", scores, g.AddReduce("max", scores, 0)))
	p := g.AddBinary("div", e, g.AddReduce("sum", e, 0))
	out := g.AddMatMul(g.AddReshape(p, []int{1, 2}), values)
	if err := g.SetOutputs(out); err != nil {
		panic(err)
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		panic(err)
	}
	candidate, err := egraph.NewCheckedOnlineAttentionCandidate(snapshot,
		map[string][]float32{"scores": {0, 0}, "values": {3, -1}}, 1, egraph.AttentionCandidatePolicy{})
	if err != nil {
		panic(err)
	}
	fmt.Println(candidate.Output(), candidate.Reference(), len(candidate.Derivation().Tiles))
	// Output: [1] [1] 2
}
