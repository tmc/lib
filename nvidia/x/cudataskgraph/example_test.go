package cudataskgraph_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func Example() {
	g, _ := cudataskgraph.New([]cudataskgraph.Task{{ID: "p", Rank: 0, Scope: cudataskgraph.ScopeCTA}})
	fmt.Println(len(g.Tasks()))
	// Output: 1
}
