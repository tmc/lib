package timeline_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// The records are supplied data, not a capture: a launch API call, a GEMM, an
// attention kernel that overlaps its tail, an idle gap, and a second attention
// kernel. Every nanosecond from the first kernel start to the last kernel end
// is in exactly one part.
func ExampleAccountDeviceSpan() {
	records := []cuptiactivity.Record{
		{Kind: cupti.ActivityKindRuntime, Name: "cudaLaunchKernel", Start: 0, End: 50, Complete: true},
		{Kind: cupti.ActivityKindConcurrentKernel, Name: "gemm", Start: 100, End: 800, Complete: true},
		{Kind: cupti.ActivityKindConcurrentKernel, Name: "attn_softmax", Start: 750, End: 900, Complete: true},
		{Kind: cupti.ActivityKindConcurrentKernel, Name: "attn_merge", Start: 1000, End: 1100, Complete: true},
	}
	attention := func(name string) bool { return strings.HasPrefix(name, "attn_") }
	kernels, err := timeline.KernelIntervalsFromActivity(records, attention)
	if err != nil {
		fmt.Println(err)
		return
	}
	a, err := timeline.AccountDeviceSpan(kernels)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%+v\n", a)
	f, err := a.Fractions()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("selected launches %.3f, selected span %.3f, idle span %.3f\n", f.SelectedLaunches, f.SelectedSpan, f.IdleSpan)
	_, err = timeline.DeviceSpan{}.Fractions()
	fmt.Println(err)
	// Output:
	// {Launches:3 SelectedLaunches:2 OtherLaunches:1 SpanNs:1000 SelectedNs:200 OtherNs:650 OverlapNs:50 IdleNs:100}
	// selected launches 0.667, selected span 0.200, idle span 0.100
	// region has no kernel launches; no fraction has a denominator
}
