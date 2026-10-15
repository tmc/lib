package egraph_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleNewOnlineAttentionSchedule() {
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
	candidate, err := egraph.NewCheckedMaskedOnlineAttentionCandidate(snapshot,
		map[string][]float32{"scores": {0, 0, float32(math.Inf(-1))}, "values": {3, -1, 999}},
		mask, 2, egraph.AttentionCandidatePolicy{})
	if err != nil {
		panic(err)
	}
	bound, err := egraph.NewOnlineAttentionSchedule(candidate, 2, "attention")
	if err != nil {
		panic(err)
	}
	fmt.Println(bound.PackedInput(), bound.SharedBytes(), bound.Schema().TotalParamBytes)
	fmt.Println(len(bound.Schedule().Validate()), strings.Contains(bound.Source(), "u[d]/l"))
	fmt.Println(bound.CandidateIdentity() == candidate.Identity(), len(bound.Identity()))
	grid, block := bound.Launch()
	fmt.Println(grid, block)
	// Compilation is a separate call. Cancellation is checked before NVRTC.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	artifact, _, err := bound.Compile(ctx, 8, 0)
	fmt.Println(artifact == nil, errors.Is(err, context.Canceled))
	// Output:
	// [0 0 3 -1 -Inf 0 999 0] 32 16
	// 0 true
	// true 64
	// [1 1 1] [64 1 1]
	// true true
}
