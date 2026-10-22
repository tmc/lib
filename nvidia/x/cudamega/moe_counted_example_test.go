package cudamega_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudamega"
)

func ExampleMoECountedWeightSource() {
	source, err := cudamega.MoECountedWeightSource(cudamega.MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 8, Hidden: 17}, "sm_121")
	fmt.Println(err)
	fmt.Println(strings.Contains(source, "task_count=2, grid_ctas=2"))
	fmt.Println(strings.Contains(source, "moe_weight_slab"))
	// Output:
	// <nil>
	// true
	// true
}

func ExampleLoadMoECountedWeightConsumer() {
	// Supply a source-bound CUBIN and actual device. This example only
	// demonstrates that missing metadata refuses before device admission.
	c, err := cudamega.LoadMoECountedWeightConsumer(nil, cudamega.MoEWeightConfig{}, nil)
	fmt.Println(c == nil, err != nil)
	// Output: true true
}
