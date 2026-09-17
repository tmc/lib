package cudair_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func ExampleGraph_Snapshot() {
	g := cudair.NewGraph()
	a := g.AddInput("A", []int{32, 64}, cudair.Float32)
	b := g.AddInput("B", []int{64, 128}, cudair.Float32)
	c := g.AddMatMul(a, b) // 32x128
	d := g.AddUnary("relu", c)

	if err := g.SetOutputs(d); err != nil {
		panic(err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		panic(err)
	}

	fmt.Printf("Nodes: %d, Outputs: %d, Order: %d\n",
		len(snap.Nodes()), len(snap.Outputs()), len(snap.Order()))
	// Output:
	// Nodes: 4, Outputs: 1, Order: 4
}
