package tileir_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/tileir"
)

func ExampleLowerOnlineAttention() {
	g := cudair.NewGraph()
	s := g.AddInput("scores", []int{2}, cudair.Float32)
	v := g.AddInput("values", []int{2, 1}, cudair.Float32)
	e := g.AddUnary("exp", g.AddBinary("sub", s, g.AddReduce("max", s, 0)))
	p := g.AddBinary("div", e, g.AddReduce("sum", e, 0))
	out := g.AddMatMul(g.AddReshape(p, []int{1, 2}), v)
	if err := g.SetOutputs(out); err != nil {
		fmt.Println(err)
		return
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		fmt.Println(err)
		return
	}
	candidate, err := egraph.NewCheckedOnlineAttentionCandidate(snapshot,
		map[string][]float32{"scores": {0, 0}, "values": {3, -1}}, 1, egraph.AttentionCandidatePolicy{})
	if err != nil {
		fmt.Println(err)
		return
	}
	lowered, err := tileir.LowerOnlineAttention(candidate, tileir.AttentionOptions{TileSize: 1})
	if err != nil {
		fmt.Println(err)
		return
	}
	data, err := lowered.Result().Module.EncodeBytecode()
	fmt.Println(lowered.Verify(), err, len(data) > 0, lowered.Result().Schema.TotalParamBytes)
	fmt.Println(lowered.Inputs()["values"], len(lowered.Identity()))
	// Output:
	// <nil> <nil> true 24
	// [3 -1] 64
}
