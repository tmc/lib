package nvtelemetry_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	n "github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleAggregateHistogramWindow_delta() {
	// Constructed source assertions, not a qualified producer capture.
	samples := hwSamples(hwReading(10, 20, []uint64{1, 2}, "delta"), hwReading(20, 30, []uint64{2, 3}, "delta"))
	manifest := hwManifest(samples, n.SchemaVersionV5)
	manifest.LostRecordsByFamily = map[string]tracecontract.Loss{"histograms": {Total: 2}}
	summaries, err := n.AggregateHistogramWindow("work", 10, 30, samples, hwOptions(manifest, n.HistogramWindowDelta))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(summaries[0].Status, summaries[0].Histogram.Counts, summaries[0].ObserverCoverage.Status)
	// Output: available [3 5] unknown
}

func ExampleAggregateHistogramWindow_cumulativeDifference() {
	samples := hwSamples(hwReading(1, 10, []uint64{1, 2}, "cumulative"), hwReading(1, 30, []uint64{3, 5}, "cumulative"))
	manifest := hwManifest(samples, n.SchemaVersionV5)
	manifest.LostRecordsByFamily = map[string]tracecontract.Loss{"histograms": {Total: 2}}
	summaries, err := n.AggregateHistogramWindow("work", 10, 30, samples, hwOptions(manifest, n.HistogramWindowCumulativeDifference))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(summaries[0].Histogram.Counts, summaries[0].MinStatus)
	// Output: [2 3] not_derivable
}

func ExampleAggregateHistogramWindow_unsupported() {
	reading := hwReading(10, 30, []uint64{1, 2}, "")
	reading.Observation.Histogram.Count = nil
	reading.Observation.Observer = n.Observer{}
	samples := hwSamples(reading)
	summaries, err := n.AggregateHistogramWindow("work", 10, 30, samples, hwOptions(hwManifest(samples, n.SchemaVersionV3), n.HistogramWindowDelta))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(summaries[0].Status, summaries[0].Reason)
	// Output: unsupported temporal_unknown
}
