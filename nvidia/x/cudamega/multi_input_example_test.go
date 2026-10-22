package cudamega_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func ExampleBody() {
	// Source and Evaluate belong to the same trusted, immutable body.
	b := pairBody{}
	source, _ := b.Source([]string{"left", "right"}, "out")
	fmt.Println(source)
	fmt.Println(b.Evaluate([]uint32{3, 2}))
	// Output:
	// out = left*7u + right;
	// 23
}

func ExampleProgram_Storage() {
	g, _ := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "a", Extent: 5, Scope: cudataskgraph.ScopeDevice},
		{ID: "b", Extent: 2, Scope: cudataskgraph.ScopeDevice},
	})
	p, _ := cudamega.NewProgram(g, map[string]cudamega.Operation{
		"a": {Kind: cudamega.Add, Scalar: 1},
		"b": {Kind: cudamega.Multiply, Scalar: 2},
	}, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32, MaxElements: 5}, cudamega.SM80)
	layout, _ := p.Storage(5)
	fmt.Println(layout)
	// Output: [{0 5} {5 2}]
}
