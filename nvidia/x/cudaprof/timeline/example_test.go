package timeline_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func ExampleUnion() {
	// Two kernels on concurrent streams overlap for 5ns; the device was busy
	// for 20ns, not 25.
	busy := []timeline.Interval{{Start: 10, End: 20}, {Start: 15, End: 30}}
	fmt.Println(timeline.Union(busy))
	// Output: 20
}
