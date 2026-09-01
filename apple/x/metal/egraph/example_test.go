package egraph_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/apple/x/metal/egraph"
)

func ExampleEGraph_ToDOT() {
	g := egraph.New()
	x := g.Add(egraph.NewLeaf("x"))
	g.Add(egraph.NewNode("rmsnorm", x))

	dot := g.ToDOT(egraph.GraphvizOptions{})
	fmt.Println(strings.HasPrefix(dot, "digraph egraph {"))

	// Output:
	// true
}

func ExampleEGraph_ToMermaid() {
	g := egraph.New()
	x := g.Add(egraph.NewLeaf("x"))
	g.Add(egraph.NewNode("rmsnorm", x))

	mermaid := g.ToMermaid(egraph.GraphvizOptions{})
	fmt.Println(strings.HasPrefix(mermaid, "flowchart TD"))

	// Output:
	// true
}
