package cudaworkload_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func ExampleJoinActivities() {
	p := cudaworkload.ProcessGenID{PID: 1, Generation: 2}
	sub := cudaworkload.HostSubmission{Process: p, OpID: 9, Symbol: "native", Workload: cudaworkload.WorkloadIdentity{Process: p, RunID: "run"}, Provenance: cudaworkload.ProvenanceCUPTI, EvidenceRef: "annotation", Context: cudaworkload.ContextStreamID{Context: 3, Stream: 4}}
	b := cudaworkload.ActivityBatch{Session: "session", Run: "run", Process: p, ClockEpoch: "producer-epoch", EvidenceRef: "capture", ExternalKind: cupti.ExternalCorrelationKindCustom0, Records: []cuptiactivity.Record{
		{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 7, ContextID: 3, StreamID: 4},
		{Kind: cupti.ActivityKindExternalCorrelation, CorrelationID: 7, ExternalKind: cupti.ExternalCorrelationKindCustom0, ExternalID: 9},
	}}
	joins, err := cudaworkload.JoinActivities(b, []cudaworkload.HostSubmission{sub})
	fmt.Println(joins[0].Join.Outcome, err)
	// Output: exact <nil>
}
