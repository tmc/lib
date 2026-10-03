package cudaprof_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

func exampleOwnerReceipt() cudaprof.CaptureProviderResult {
	// These are synthetic provenance fields for this offline example.
	return cudaprof.CaptureProviderResult{PassID: "pass-1", Replay: "replay-1", Process: 42, Generation: 1, Context: "target-context", Start: 10, Stop: 20, CleanupComplete: true,
		Identity:    cudaevidence.Identity{ArtifactDigest: strings.Repeat("1", 64), Entry: "kernel", Target: "sm_70", CompilerKey: strings.Repeat("2", 64), WorkloadKey: strings.Repeat("3", 64)},
		Environment: cudaprof.CaptureEnvironment{Device: "example", Driver: "example", CUDA: "example", CUPTI: "example", ClockMode: "unchanged", CorrelationDomain: "example-clock"}}
}
func ExampleRangeCaptureProviderResult() {
	// A production caller supplies its actual owner's capture and Close receipt.
	capture := &cuptirange.Capture{Image: []byte{1}, Ranges: 1, Quality: cuptirange.CollectionQuality{PassesStarted: 1, PassesStopped: 1, AllPassesSubmitted: true, Decoded: true}}
	result, err := cudaprof.RangeCaptureProviderResult(exampleOwnerReceipt(), capture, []byte{7}, []string{"metric"}, nil)
	fmt.Println(result.Provider, result.Scope, *result.Counts.Retained, err)
	// Output: range target-context 1 <nil>
}
func ExamplePMCaptureProviderResult() {
	// CounterData and SampleInfo must be copied before the caller closes Session.
	decoded := cuptipmsampling.DecodeResult{StopReason: cuptipmsampling.DecodeStopReasonEndOfRecords, Info: cuptipmsampling.CounterDataInfo{NumTotalSamples: 1, NumPopulatedSamples: 1, NumCompletedSamples: 1}}
	samples := []cuptipmsampling.SampleInfo{{Index: 0, StartTimestamp: 11, EndTimestamp: 19}}
	result, err := cudaprof.PMCaptureProviderResult(exampleOwnerReceipt(), decoded, samples, []byte{1}, cuptipmsampling.Config{ConfigImage: []byte{7}, MetricNames: []string{"metric"}, SamplingInterval: 1000, MaxSamples: 8}, nil)
	fmt.Println(result.Provider, result.Scope, *result.Counts.Retained, err)
	// Output: pm device-wide 1 <nil>
}

func ExampleSASSCaptureProviderResult() {
	// Synthetic owner structures demonstrate loss preservation without a GPU.
	config := cuptisass.Config{Metrics: []cuptisass.MetricConfig{{MetricID: 7, Granularity: cuptisass.GranularityGPU}}}
	records := []cuptisass.Record{{FunctionName: "kernel", Values: []cuptisass.InstanceValue{{MetricID: 7, Value: 123}}}}
	result, err := cudaprof.SASSCaptureProviderResult(exampleOwnerReceipt(), config, records, nil, nil)
	fmt.Println(result.Provider, result.Status, result.Counts.Dropped == nil, err)
	// Output: sass partial true <nil>
}
