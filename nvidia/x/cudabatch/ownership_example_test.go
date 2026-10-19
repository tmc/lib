package cudabatch_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

func ExampleRequest_Err() {
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{}, nil)
	var finished *cudabatch.Request
	req := &cudabatch.Request{
		ID:           "req-1",
		PromptTokens: []int32{10, 20},
		MaxTokens:    2,
		OnFinish:     func(r *cudabatch.Request) { finished = r },
	}
	_ = sched.Submit(req)
	req.PromptTokens[0] = 99
	_ = sched.Cancel("req-1")
	snap, _ := sched.Request("req-1")
	fmt.Println(snap.State, snap.Err(), finished.Err(), snap.PromptTokens[0])
	// Output: canceled context canceled context canceled 10
}

func ExampleBindAllocatedRows() {
	pool := cudakv.NewBlockAllocator(2, 2)
	_ = pool.AllocSequence(11, nil, 3)
	row := cudabatch.BatchRowBinding{
		RowIndex:       0,
		SeqID:          "req-1",
		IsPrefill:      true,
		Tokens:         []int32{10, 20, 30},
		SequenceLength: 3,
		BlockTable:     pool.Table(11),
	}
	bucket := cudabatch.ShapeBucket{BatchSize: 1, MaxSeqLen: 4, MaxTokens: 3}
	meta, err := cudabatch.BindAllocatedRows([]cudabatch.BatchRowBinding{row}, bucket, pool, []int64{11})
	_, errOwner := cudabatch.BindAllocatedRows([]cudabatch.BatchRowBinding{row}, bucket, pool, []int64{99})
	fmt.Println(err, meta.TotalTokens, meta.SequenceLens, meta.BlockTableFlat, errors.Is(errOwner, cudabatch.ErrCorruptBlockTable))
	// Output: <nil> 3 [3] [1 0] true
}

func ExampleBatchScheduler_BindIteration() {
	pool := cudakv.NewBlockAllocator(2, 4)
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    2,
		MaxBudgetTokens: 8,
		ChunkSize:       4,
	}, cudabatch.NewMemoryBudgetTracker(pool))
	_ = sched.Submit(&cudabatch.Request{ID: "req-1", PromptTokens: []int32{10, 20, 30}, MaxTokens: 2})
	iter, _ := sched.Schedule()
	bucket := cudabatch.ShapeBucket{BatchSize: 1, MaxSeqLen: 4, MaxTokens: 4}
	meta, err := sched.BindIteration(iter, bucket)
	iter.AllocatedRows[0].BlockTable[0] = 99
	_, errTampered := sched.BindIteration(iter, bucket)
	fmt.Println(err, meta.TotalTokens, meta.SequenceLens, errors.Is(errTampered, cudabatch.ErrInvalidRequest))
	// Output: <nil> 3 [3] true
}
