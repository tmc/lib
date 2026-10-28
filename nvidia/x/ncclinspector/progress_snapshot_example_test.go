package ncclinspector_test

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

func ExampleReadProgressSnapshots() {
	// Constructed output of the pure C++ header, not a native NCCL capture.
	raw := []byte(progressCPPJSONL)
	trace, err := ncclinspector.ReadProgressSnapshots(context.Background(), strings.NewReader(string(raw)), progressCapture(raw), ncclinspector.Limits{})
	if err != nil {
		panic(err)
	}
	fmt.Println(len(trace.Records()), "reported diagnostic rows")
	// Output: 4 reported diagnostic rows
}

func ExampleSummarizeProgress() {
	// Source assertions and independent association are explicit caller inputs.
	raw := []byte(progressCPPJSONL)
	trace, err := ncclinspector.ReadProgressSnapshots(context.Background(), strings.NewReader(string(raw)), progressCapture(raw), ncclinspector.Limits{})
	if err != nil {
		panic(err)
	}
	records, err := ncclinspector.Read(strings.NewReader(progressInspectorJSONL), ncclinspector.Limits{})
	if err != nil {
		panic(err)
	}
	summary, err := ncclinspector.SummarizeProgress(trace, progressBinding(raw), records[0])
	if err != nil {
		panic(err)
	}
	fresh := 0
	for _, observation := range summary.Observations {
		if observation.Fresh {
			fresh++
		}
	}
	fmt.Println(fresh, "successful diagnostic mirror")
	fmt.Println("GPU completion withheld:", slices.Contains(summary.Withheld, "GPU completion"))
	// Output:
	// 1 successful diagnostic mirror
	// GPU completion withheld: true
}

func ExampleProgressTrace_Records() {
	raw := []byte(progressCPPJSONL)
	trace, err := ncclinspector.ReadProgressSnapshots(context.Background(), strings.NewReader(string(raw)), progressCapture(raw), ncclinspector.Limits{})
	if err != nil {
		panic(err)
	}
	rows := trace.Records()
	rows[2].Mirror.Counts[0] = 0
	fmt.Println(trace.Records()[2].Mirror.Counts[0])
	// Output: 9007199254740993
}
