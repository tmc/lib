package cudaworkload_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

type exampleHook struct{ id uint64 }

func (h *exampleHook) Push(kind cupti.ExternalCorrelationKind, id uint64) error {
	h.id = id
	return nil
}

func (h *exampleHook) Pop(kind cupti.ExternalCorrelationKind) (uint64, error) {
	return h.id, nil
}

func Example() {
	// Initialize workload annotator for process PID 1234, generation 1
	proc := cudaworkload.ProcessGenID{PID: 1234, Generation: 1}
	annotator := cudaworkload.NewAnnotator(proc, "inference-pipeline",
		cudaworkload.WithCorrelationHook(&exampleHook{}),
	)

	ctx := context.Background()
	_, op := annotator.Begin(ctx, "prompt_prefill")
	defer op.End()

	// Perform scoped native CUDA call
	var nativeCorrID uint32 = 1001
	stream := cudaworkload.ContextStreamID{Context: 0xCAFE, Stream: 0x1}
	sub, err := op.NativeCallWithSubmission(stream, cudaworkload.GraphReplayGeneration{}, "cudaLaunchKernel", "kernel_ref", func() (uint32, error) {
		// Native CUDA driver/runtime launch occurs here on pinned OS thread.
		return nativeCorrID, nil
	})
	if err != nil {
		fmt.Printf("NativeCall error: %v\n", err)
		return
	}

	// Joiner correlates host submissions with incoming device activity records
	joiner := cudaworkload.NewJoiner(cudaworkload.JoinConfig{})
	joiner.SubmitHost(sub)

	// Simulate device activity record collected from CUPTI
	act := cudaworkload.DeviceActivity{
		Process:               proc,
		NativeCorrelationID:   nativeCorrID,
		ExternalCorrelationID: uint64(op.ID()),
		Kind:                  "kernel",
		Name:                  "gemm_prefill_sm121",
		Context:               stream,
		StartNs:               10000,
		EndNs:                 15000,
	}

	results := joiner.SubmitDevice(act)
	for _, res := range results {
		fmt.Printf("Outcome: %s, Reason: %s, Workload: %s, operation matches: %t\n",
			res.Outcome, res.Reason, res.Workload.WorkloadName, res.OpID == op.ID())
	}

	// Output:
	// Outcome: exact, Reason: explicit_external_correlation_match, Workload: inference-pipeline, operation matches: true
}
