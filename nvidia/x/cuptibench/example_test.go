package cuptibench_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cuptibench"
)

func Example() {
	result := cuptibench.Result{Operations: 4, Ranges: []cuptibench.Range{{ID: "operation#1", Name: "operation", Kind: "operation", Metrics: []cuptibench.Metric{{Name: "smsp__inst_executed.sum", Unit: "inst", Value: 120, Additive: true}}}}}
	fmt.Println(result.Ranges[0].Metrics[0].Value/float64(result.Operations), "inst/op")
	// Output: 30 inst/op
}
func ExampleNew() {
	_, err := cuptibench.New(nil, cuptibench.Config{})
	fmt.Println(err != nil)
	// Output: true
}
func ExamplePlan_Collect() {
	var plan *cuptibench.Plan
	_, err := plan.Collect(cuptibench.Workload{})
	fmt.Println(err)
	// Output: cuptibench: nil plan
}
func ExamplePlan_Close() {
	var plan *cuptibench.Plan
	fmt.Println(plan.Close())
	// Output: <nil>
}
func ExampleScope_Region() {
	workload := cuptibench.Workload{Run: func(scope *cuptibench.Scope) error {
		return scope.Region("attention", func() error { return nil })
	}}
	fmt.Println(workload.Run != nil)
	// Output: true
}
func ExampleResolve() {
	// A selector is a curated set name or a native metric name, so a caller
	// can mix the two and name metrics no set covers.
	names, err := cuptibench.Resolve("launches", "l1tex__t_sectors.sum")
	fmt.Println(names, err)
	_, err = cuptibench.Resolve("unknown")
	fmt.Println(err)
	// Output:
	// [sm__ctas_launched.sum sm__warps_launched.sum l1tex__t_sectors.sum] <nil>
	// cuptibench: unknown metric set "unknown", have default, instructions, l2, l2-read, launches
}
