package counters_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprof/counters"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
)

func ExampleSelect() {
	_, err := counters.Select(shape.Report{Version: 1}, shape.Identity{Operation: "gemv"})
	fmt.Println(err)
	// Output: counters: shape qualification failed
}

func ExampleCompare() {
	report := counters.Compare(counters.Receipt{}, counters.Receipt{})
	fmt.Println(report.Compatible, report.Reason)
	// Output: false shape identity unavailable
}

func ExampleCollect() {
	report, err := counters.Collect(nil, counters.CollectOptions{}, counters.Replay{})
	fmt.Println(report.LockDeclared, err)
	// Output: false counters: an existing GPU lock declaration is required
}
