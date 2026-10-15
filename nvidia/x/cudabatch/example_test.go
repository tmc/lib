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

	// Complete prefill with the first token sampled from its final logits.
	_ = sched.CompleteStep(iter, map[string]int32{"req-1": 41}, nil)
	r, _ := sched.Request("req-1")
	fmt.Printf("Request state after prefill: %v\n", r.State)

	// Output:
	// Scheduled 1 prefill chunks, 4 total tokens
	// Request state after prefill: decoding
}

func ExampleValidateAndBindRows() {
	rows := []cudabatch.BatchRowBinding{
		{RowIndex: 0, SeqID: "prefill", IsPrefill: true, Tokens: []int32{7, 8}, SequenceLength: 2, BlockTable: []int{1}},
		{RowIndex: 1, SeqID: "decode", Tokens: []int32{9}, StartPos: 2, SequenceLength: 3, BlockTable: []int{2, 3}},
	}
	metadata, err := cudabatch.ValidateAndBindRows(rows, cudabatch.ShapeBucket{BatchSize: 3, MaxSeqLen: 5, MaxTokens: 3}, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(metadata.TotalTokens, metadata.SequenceLens)
	// This is one prefix-validity mask per row, not a per-query causal mask.
	fmt.Println(metadata.AttentionMask)
	// Output:
	// 3 [2 3 0]
	// [1 1 0 0 0 1 1 1 0 0 0 0 0 0 0]
}
