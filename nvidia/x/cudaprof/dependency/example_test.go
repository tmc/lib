package dependency_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprof/dependency"
)

func Example() {
	report, err := dependency.Analyze(dependency.Graph{})
	fmt.Println(report.State, err)
	// Output: partial <nil>
}
func ExampleAnalyze() {
	report, err := dependency.Analyze(dependency.Graph{})
	fmt.Println(len(report.CriticalPath), err)
	// Output: 0 <nil>
}
func ExampleGraph() {
	g := dependency.Graph{Complete: false}
	fmt.Println(g.Complete)
	// Output: false
}
func ExampleNode() {
	n := dependency.Node{ID: "kernel", Record: "journal:activity:0"}
	fmt.Println(n.Record)
	// Output: journal:activity:0
}
func ExampleEdge() {
	e := dependency.Edge{Kind: "event_wait", Record: "journal:event:4"}
	fmt.Println(e.Kind)
	// Output: event_wait
}
func ExampleReport() {
	r := dependency.Report{State: "partial"}
	fmt.Println(r.State)
	// Output: partial
}
