package cudadiagdemo

import (
	"context"
	"fmt"
)

func ExampleComposeNativeEvidence() {
	// An actual incomplete native request stays visible. There is no supplied
	// host/device mapping or scheduler/loss proof to manufacture admission.
	result := &NativeResult{Qualification: "partial", Requests: []NativeRequestLog{{Attempt: "baseline", GroundTruth: GroundTruthLog{RequestID: "request-1", WorkloadName: "transform"}}}}
	summary, err := ComposeNativeEvidence(context.Background(), result, NativeCompositionOptions{})
	fmt.Println(summary.Synthetic, len(summary.Requests), summary.Requests[0].Diagnosis.Admitted, err)
	// Output: false 1 false <nil>
}
