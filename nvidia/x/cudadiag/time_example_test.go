package cudadiag_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

func ExampleSumDuration() {
	d, err := cudadiag.SumDuration([]tracecontract.Interval{{Start: 0, End: 10}, {Start: 5, End: 15}})
	fmt.Println(d, err)
	// Output: 20ns <nil>
}

func ExampleUnionDuration() {
	d, err := cudadiag.UnionDuration([]tracecontract.Interval{{Start: 0, End: 10}, {Start: 5, End: 15}})
	fmt.Println(d, err)
	// Output: 15ns <nil>
}

func ExampleComputeTimeAccounting() {
	ta, err := cudadiag.ComputeTimeAccounting(tracecontract.Interval{End: 20}, nil, []cudadiag.KernelEvent{{Span: tracecontract.Interval{End: 10}}}, cudadiag.SchedEvidence{})
	fmt.Println(ta.SummedKernelDuration, ta.BusyDuration, err)
	// Output: 10ns 10ns <nil>
}

func ExampleSchedEvidence_Durations() {
	s := cudadiag.SchedEvidence{Intervals: []cudadiag.SchedInterval{{State: cudadiag.SchedRunning, Span: tracecontract.Interval{End: 10}}}}
	runnable, scheduled, wait, err := s.Durations()
	fmt.Println(runnable, scheduled, wait, err)
	// Output: 0s 10ns 0s <nil>
}
