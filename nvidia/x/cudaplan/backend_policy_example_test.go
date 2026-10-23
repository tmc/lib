package cudaplan_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

func ExampleExecutablePlan_BackendDecisions() {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{4}, cudair.Float32)
	y := g.AddUnary("neg", x)
	if err := g.SetOutputs(y); err != nil {
		panic(err)
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		panic(err)
	}
	p, err := cudaplan.Compile(snapshot)
	if err != nil {
		panic(err)
	}
	defer p.Close()
	d := p.BackendDecisions()[0]
	fmt.Println(d.Backend, d.Region.Start, d.Region.End, d.TargetChecked)
	// Output: Launch 0 1 false
}
