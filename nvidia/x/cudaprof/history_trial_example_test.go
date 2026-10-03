package cudaprof_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func ExampleHistoryTrial() {
	// A completion flag alone cannot supply missing artifact/workload identity.
	_, err := cudaprof.HistoryTrial(cudaprofilehistory.Record{}, cudatrial.Receipt{Complete: true}, cudaprofilehistory.Correctness{Verdict: cudaprofilehistory.Unknown}, cudaprof.HistoryTrialQualification{})
	fmt.Println(err != nil)
	// Output: true
}
