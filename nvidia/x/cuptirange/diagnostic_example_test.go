package cuptirange_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func ExampleLaunchTrace() {
	trace := cuptirange.LaunchTrace{Launches: []cuptirange.ObservedLaunch{{API: "cuLaunchKernel", Symbol: "vector_add"}}}
	fmt.Println(len(trace.Launches), trace.CompletedPasses)
	// Output: 1 0
}
