package overview_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func Example() {
	r := overview.Analyze(timeline.Records{}, "driver", nil)
	fmt.Println(r.Version, len(r.Tokens), len(r.Unmatched))
	// Output: 1 0 0
}
func ExampleAnalyze() {
	records := timeline.Records{Activities: []timeline.Activity{{Journal: "j", Subtype: "kernel", Name: "raw_kernel", Start: 10, End: 20}}}
	r := overview.Analyze(records, "driver", nil)
	fmt.Println(r.Hierarchy.Metrics.KernelCount, len(r.Unmatched))
	// Output: 1 1
}

func ExampleReference() {
	r := overview.Reference{Journal: "capture.journal", Kind: "activity", Index: 0}
	fmt.Println(r.Kind, r.Index)
	// Output: activity 0
}
func ExampleInterval() {
	i := overview.Interval{Journal: "j", Start: 10, End: 20}
	fmt.Println(i.End - i.Start)
	// Output: 10
}
func ExampleMetrics() {
	m := overview.Metrics{KernelCount: 2, KernelSumNanos: 30}
	fmt.Println(m.KernelCount, m.KernelSumNanos)
	// Output: 2 30
}
func ExampleNode() {
	n := overview.Node{Kind: "operation", Label: "gemv", Provenance: "observed"}
	fmt.Println(n.Kind, n.Provenance)
	// Output: operation observed
}
func ExampleToken() {
	t := overview.Token{Index: 0, Provenance: "observed"}
	fmt.Println(t.Index, t.Delivery == nil)
	// Output: 0 true
}
func ExampleReport() {
	r := overview.Analyze(timeline.Records{}, "driver", nil)
	fmt.Println(r.Version, len(r.Inferred))
	// Output: 1 0
}
