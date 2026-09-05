package cudagraph_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudagraph"
)

func ExampleInstantiateOptions() {
	options := cudagraph.InstantiateOptions{UseNodePriority: true}
	fmt.Println(options.DeviceLaunch, options.UseNodePriority)
	// Output: false true
}

func ExampleGraph_InstantiateWithOptions() {
	var graph cudagraph.Graph
	_, err := graph.InstantiateWithOptions(cudagraph.InstantiateOptions{})
	fmt.Println(err)
	// Output: cudagraph: graph is closed
}
