package cuptiactivity_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
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
