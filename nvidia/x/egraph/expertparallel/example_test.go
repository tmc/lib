package expertparallel_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/egraph/expertparallel"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func Example() {
	cfg := expertparallel.Config{
		Ranks: 4, Tokens: 64, Hidden: 256, Inter: 128,
		Experts: 32, TopK: 8, SharedInter: 128,
		Route: hopper.RowTopK{Normalize: true},
	}
	p, err := expertparallel.New(cfg, "moe")
	if err != nil {
		log.Fatal(err)
	}
	for _, s := range p.Stages() {
		fmt.Println(s.Name, s.Side, s.Collective, s.Kernels)
	}
	// Output:
	// route false false 1
	// layout false false 1
	// shared true false 2
	// dispatch false true 1
	// expert false false 4
	// combine false true 1
}
