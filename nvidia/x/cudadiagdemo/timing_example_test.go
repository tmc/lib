package cudadiagdemo_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadiagdemo"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

func ExampleSubmissionTiming() {
	// Constructed replay intervals, not physical observations.
	timing := cudadiagdemo.SubmissionTiming{PreEntry: tracecontract.Interval{Start: 10, End: 400, Clock: "replay"}}
	fmt.Println(timing.PreEntry.Duration())
	// Output: 390
}

func ExampleTimingSource() {
	source := cudadiagdemo.TimingSource{Version: 1, SourceClass: "synthetic", ClockEpoch: "replay-one"}
	fmt.Println(source.Version, source.SourceClass, source.ClockEpoch)
	// Output: 1 synthetic replay-one
}
