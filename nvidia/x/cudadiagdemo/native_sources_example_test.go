package cudadiagdemo

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func ExampleNativeActivitySource() {
	// Constructed bytes in the activity.jsonl encoding, a synthetic host
	// contract rather than a native capture: one annotated launch and one
	// unannotated kernel. Binding them admits no diagnosis.
	process := cudaworkload.ProcessGenID{PID: 4242, Generation: 1}
	workload := cudaworkload.WorkloadIdentity{Process: process, WorkloadName: "transform", RunID: "run-1"}
	result := &NativeResult{
		Synthetic:   true,
		Requests:    []NativeRequestLog{{Attempt: "a", Complete: true, GroundTruth: GroundTruthLog{RequestID: "request-1", WorkloadName: "transform", OpID: 7, ProcessGen: process, Workload: workload, TID: 11}}},
		Annotations: []cudaworkload.HostSubmission{{Process: process, OpID: 7, Workload: workload, Provenance: cudaworkload.ProvenanceCUPTI, TID: 11, Symbol: "cuLaunchKernel", EvidenceRef: "example"}},
	}
	var raw bytes.Buffer
	encoder := json.NewEncoder(&raw)
	for _, record := range []cuptiactivity.Record{
		{Kind: cupti.ActivityKindExternalCorrelation, ExternalKind: cupti.ExternalCorrelationKindCustom0, CorrelationID: 1, ExternalID: 7},
		{Kind: cupti.ActivityKindDriver, Name: "cuLaunchKernel", CorrelationID: 1, Start: 10, End: 20, Complete: true},
		{Kind: cupti.ActivityKindConcurrentKernel, Name: "transform", CorrelationID: 2, Start: 30, End: 50, Complete: true},
	} {
		if err := encoder.Encode(record); err != nil {
			fmt.Println(err)
			return
		}
	}
	sum := sha256.Sum256(raw.Bytes())
	source := NativeActivitySource{
		Ref:    cudaregress.ArtifactRef{Name: "activity.jsonl", Digest: hex.EncodeToString(sum[:]), SizeBytes: int64(raw.Len()), Role: "trace"},
		Reader: &raw,
		Batch:  cudaworkload.ActivityBatch{Session: "run-1", Run: "run-1", ClockEpoch: "run-1", Process: process, ExternalKind: cupti.ExternalCorrelationKindCustom0},
	}
	out, err := ComposeNativeEvidence(context.Background(), result, NativeCompositionOptions{Activity: &source})
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, join := range out.ActivityAnalysis.Joins {
		fmt.Println(join.Record.Name, join.Join.Outcome, uint64(join.Join.OpID))
	}
	fmt.Println(out.Requests[0].Diagnosis.Admitted)
	// Output:
	// cuLaunchKernel exact 7
	// transform missing 0
	// false
}
