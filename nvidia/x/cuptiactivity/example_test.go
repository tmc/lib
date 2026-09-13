package cuptiactivity_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

func ExampleRecord() {
	r := cuptiactivity.Record{Start: 10, End: 25, Complete: true}
	if r.Complete {
		fmt.Println(r.End - r.Start)
	}
	// Output: 15
}

func ExampleRecord_QueuedToSubmitted() {
	r := cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Queued: 10, Submitted: 25}
	ns, ok := r.QueuedToSubmitted()
	fmt.Println(ns, ok)
	// Output: 15 true
}

func ExampleRecord_SubmittedToStart() {
	r := cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Submitted: 25, Start: 30}
	ns, ok := r.SubmittedToStart()
	fmt.Println(ns, ok)
	// Output: 5 true
}

func ExampleAttributePMSamples() {
	records := []cuptiactivity.Record{
		{
			Kind:     cupti.ActivityKindConcurrentKernel,
			Name:     "gemm",
			Start:    1000,
			End:      2000,
			Complete: true,
		},
	}
	samples := []cuptipmsampling.Sample{
		{
			Index:          0,
			StartTimestamp: 1000,
			EndTimestamp:   2000,
			Values:         map[string]float64{"sm_busy": 85.0},
		},
	}
	attributed := cuptiactivity.AttributePMSamples(records, samples)
	if len(attributed) > 0 {
		fmt.Printf("%s sm_busy=%.1f\n", attributed[0].KernelName, attributed[0].PMAverages["sm_busy"])
	}
	// Output:
	// gemm sm_busy=85.0
}
