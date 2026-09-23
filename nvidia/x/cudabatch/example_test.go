package cudabatch_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudabatch"
)

func ExampleBatchScheduler() {
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    4,
		MaxBudgetTokens: 64,
		ChunkSize:       16,
	}, nil)

	req := &cudabatch.Request{
		ID:           "req-1",
		PromptTokens: []int32{10, 20, 30, 40},
		MaxTokens:    2,
	}
	if err := sched.Submit(req); err != nil {
		panic(err)
	}

	iter, err := sched.Schedule()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Scheduled %d prefill chunks, %d total tokens\n", len(iter.PrefillChunks), iter.TotalTokens)

	// Complete prefill step
	_ = sched.CompleteStep(iter, nil, nil)
	r, _ := sched.Request("req-1")
	fmt.Printf("Request state after prefill: %v\n", r.State)

	// Output:
	// Scheduled 1 prefill chunks, 4 total tokens
	// Request state after prefill: decoding
}
