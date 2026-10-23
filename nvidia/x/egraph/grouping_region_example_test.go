package egraph_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func groupingExample() []*egraph.GroupingCandidate {
	g := cudair.NewGraph()
	x := g.AddInput("tokens", []int{3, 1}, cudair.Float32)
	w := g.AddInput("weights", []int{1, 1}, cudair.Float32)
	if err := g.SetOutputs(g.AddMatMul(x, w)); err != nil {
		panic(err)
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		panic(err)
	}
	candidates, err := egraph.NewCheckedGroupingCandidates(snapshot,
		map[string][]float32{"tokens": {1, 2, 3}, "weights": {2}},
		[]int{1, 2}, egraph.GroupingPolicy{MaxOperations: 9})
	if err != nil {
		panic(err)
	}
	return candidates
}

func ExampleNewCheckedGroupingCandidates() {
	for _, candidate := range groupingExample() {
		values, err := candidate.Run()
		fmt.Println(candidate.Derivation().Width, values, err)
	}
	// Output:
	// 1 [2 4 6] <nil>
	// 2 [2 4 6] <nil>
}

func ExampleGroupingCandidate_Run() {
	values, err := groupingExample()[1].Run()
	fmt.Println(values, err)
	// Output: [2 4 6] <nil>
}

func ExampleGroupingCandidate_Derivation() {
	d := groupingExample()[1].Derivation()
	fmt.Println(d.Rule, d.Width, d.Rows)
	// Output: provided-independent-dense-rows-v1 2 [{0 2} {2 3}]
}

func ExampleGroupingCandidate_Inputs() {
	c := groupingExample()[1]
	c.Inputs()["tokens"][0] = 99
	fmt.Println(c.Inputs()["tokens"])
	// Output: [1 2 3]
}

func ExampleGroupingCandidate_Reference() {
	fmt.Println(groupingExample()[1].Reference())
	// Output: [2 4 6]
}

func ExampleGroupingCandidate_Snapshot() {
	fmt.Println(groupingExample()[1].Snapshot().Outputs()[0].Op)
	// Output: matmul
}

func ExampleGroupingCandidate_Identity() {
	cs := groupingExample()
	fmt.Println(len(cs[0].Identity()), cs[0].Identity() == cs[1].Identity())
	// Output: 64 false
}
