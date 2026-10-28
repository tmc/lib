package cudagreen_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudagreen"
)

func ExampleComputePlan() {
	g, err := cudagreen.DefaultGranularity("sm_100a")
	if err != nil {
		panic(err)
	}
	plan, err := cudagreen.ComputePlan("sm_100a", 160, g, cudagreen.DecodeHeavy())
	if err != nil {
		panic(err)
	}
	if err := cudagreen.VerifyPlan(plan); err != nil {
		panic(err)
	}
	for _, p := range plan.Partitions {
		fmt.Printf("%-11v SMs %3d-%3d\n", p.Spec.Role, p.First, p.First+p.SMs-1)
	}
	// Output:
	// compute     SMs   0- 31
	// decode      SMs  32-143
	// fabric      SMs 144-151
	// maintenance SMs 152-159
}
