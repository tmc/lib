package metalgraphexec_test

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal/metalgraphexec"
)

func Example() {
	graph := metalgraphexec.New()
	graph.AddDispatch("LayerNorm", "NormPipeline", []int{0, 1, 2}, [3]int{1, 1, 1}, [3]int{128, 1, 1})
	graph.AddDispatch("Megakernel", "LayerPipeline", []int{2, 3, 4, 5}, [3]int{64, 1, 1}, [3]int{128, 1, 1})

	fmt.Printf("Graph Node Count: %d\n", graph.NodeCount())
	fmt.Printf("Graph Valid: %t\n", graph.Validate() == nil)

	// Output:
	// Graph Node Count: 2
	// Graph Valid: true
}
