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
	_, err := cuptibench.Resolve("unknown")
	fmt.Println(err)
	// Output: cuptibench: unknown metric set "unknown"
}
