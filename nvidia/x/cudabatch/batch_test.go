package cudabatch_test

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

// TestBatchTokenBudget verifies that the scheduler strictly respects MaxBudgetTokens
// across iterations with mixed chunk sizes.
func TestBatchTokenBudget(t *testing.T) {
	budget := 64
	chunkSize := 16
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    8,
		MaxBudgetTokens: budget,
		ChunkSize:       chunkSize,
	}, nil)

	// Submit 3 requests with large prompts (50 tokens each)
	for i := 1; i <= 3; i++ {
		prompt := make([]int32, 50)
		for p := range prompt {
			prompt[p] = int32(p + 1)
		}
		if err := sched.Submit(&cudabatch.Request{
			ID:           fmt.Sprintf("req-%d", i),
			PromptTokens: prompt,
			MaxTokens:    10,
		}); err != nil {
			t.Fatalf("Submit req-%d: %v", i, err)
		}
	}

	iter, err := sched.Schedule()
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	if iter.TotalTokens > budget {
		t.Fatalf("iteration tokens %d exceeded budget %d", iter.TotalTokens, budget)
	}
	// 3 requests admitted, each takes min(16, 50) = 16 tokens -> total 48 <= 64
	if iter.TotalTokens != 48 {
		t.Fatalf("got %d tokens, want 48", iter.TotalTokens)
	}
}

// TestBatchFairness verifies that long prefills make progress and are not starved
// by shorter or later-arriving prefills under PolicyFairShare.
func TestBatchFairness(t *testing.T) {
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    4,
		MaxBudgetTokens: 32,
		ChunkSize:       16,
		FairnessPolicy:  cudabatch.PolicyFairShare,
	}, nil)

	// Submit req1: long prefill (64 tokens)
	longPrompt := make([]int32, 64)
	if err := sched.Submit(&cudabatch.Request{
		ID:           "long-prefill",
		PromptTokens: longPrompt,
		MaxTokens:    10,
	}); err != nil {
		t.Fatal(err)
	}

	// Submit req2: short prefill (16 tokens)
	shortPrompt := make([]int32, 16)
	if err := sched.Submit(&cudabatch.Request{
		ID:           "short-prefill",
		PromptTokens: shortPrompt,
		MaxTokens:    10,
	}); err != nil {
		t.Fatal(err)
	}

	// Iteration 1: budget 32, both reqs take 16 tokens each -> total 32
	iter1, err := sched.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	if len(iter1.PrefillChunks) != 2 {
		t.Fatalf("expected 2 prefill chunks, got %d", len(iter1.PrefillChunks))
	}
	_ = sched.CompleteStep(iter1, nil, nil)

	rShort, _ := sched.Request("short-prefill")
	if rShort.State != cudabatch.StateDecoding {
		t.Fatalf("short-prefill state = %v, want decoding", rShort.State)
	}

	rLong, _ := sched.Request("long-prefill")
	if rLong.PrefilledTokens != 16 {
		t.Fatalf("long-prefill prefilled %d, want 16", rLong.PrefilledTokens)
	}

	// Iteration 2: long-prefill should continue progressing
	iter2, err := sched.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	_ = sched.CompleteStep(iter2, map[string]int32{"short-prefill": 999}, nil)

	rLongAfter, _ := sched.Request("long-prefill")
	if rLongAfter.PrefilledTokens <= 16 {
		t.Fatalf("long-prefill starved: prefilled remained %d", rLongAfter.PrefilledTokens)
	}
}

// TestCancelRetainsKVUntilCompletion verifies that canceling a request during active
// execution retains its KV blocks and release callbacks until execution completes.
func TestCancelRetainsKVUntilCompletion(t *testing.T) {
	alloc := cudakv.NewBlockAllocator(16, 10)
	tracker := cudabatch.NewMemoryBudgetTracker(alloc)
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    4,
		MaxBudgetTokens: 64,
		ChunkSize:       16,
	}, tracker)

	prompt := make([]int32, 32)
	reqID := "cancel-test"
	if err := sched.Submit(&cudabatch.Request{
		ID:           reqID,
		PromptTokens: prompt,
		MaxTokens:    5,
	}); err != nil {
		t.Fatal(err)
	}

	// Schedule iteration
	iter, err := sched.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	if len(iter.AllocatedRows) != 1 {
		t.Fatalf("expected 1 row scheduled, got %d", len(iter.AllocatedRows))
	}

	// Verify blocks were allocated
	initialFree := alloc.NumFreeBlocks()
	if initialFree != 8 { // 10 total - 2 blocks for 32 tokens
		t.Fatalf("free blocks = %d, want 8", initialFree)
	}

	var releaseCalled int32
	sched.RegisterResourceRelease(reqID, func() {
		atomic.AddInt32(&releaseCalled, 1)
	})

	// Cancel while actively executing
	if err := sched.Cancel(reqID); err != nil {
		t.Fatalf("Cancel error: %v", err)
	}

	// KV blocks and release callback MUST NOT be released yet
	if alloc.NumFreeBlocks() != 8 {
		t.Fatalf("premature KV block release upon Cancel: free = %d, want 8", alloc.NumFreeBlocks())
	}
	if atomic.LoadInt32(&releaseCalled) != 0 {
		t.Fatalf("premature resource callback execution upon Cancel")
	}

	// Complete the step (execution completes)
	if err := sched.CompleteStep(iter, nil, nil); err != nil {
		t.Fatal(err)
	}

	// NOW the request is canceled, KV blocks returned to pool, and callbacks invoked
	if alloc.NumFreeBlocks() != 10 {
		t.Fatalf("KV blocks not freed after step completion: free = %d, want 10", alloc.NumFreeBlocks())
	}
	if atomic.LoadInt32(&releaseCalled) != 1 {
		t.Fatalf("release callback was not called upon step completion")
	}

	r, ok := sched.Request(reqID)
	if !ok || r.State != cudabatch.StateCanceled {
		t.Fatalf("request state = %v, want canceled", r.State)
	}
}

// TestBatchSequenceIsolation verifies that rows in a batch maintain isolated sequence
// metadata, offsets, and block tables without cross-talk or overlap.
func TestBatchSequenceIsolation(t *testing.T) {
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    4,
		MaxBudgetTokens: 128,
		ChunkSize:       32,
	}, nil)

	p1 := []int32{1, 2, 3, 4}
	p2 := []int32{10, 20, 30, 40, 50, 60}
	_ = sched.Submit(&cudabatch.Request{ID: "seq-1", PromptTokens: p1, MaxTokens: 2})
	_ = sched.Submit(&cudabatch.Request{ID: "seq-2", PromptTokens: p2, MaxTokens: 2})

	iter, err := sched.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	if len(iter.AllocatedRows) != 2 {
		t.Fatalf("allocated rows = %d, want 2", len(iter.AllocatedRows))
	}

	row1 := iter.AllocatedRows[0]
	row2 := iter.AllocatedRows[1]

	if row1.SeqID != "seq-1" || !reflect.DeepEqual(row1.Tokens, p1) {
		t.Fatalf("row1 corrupted: %+v", row1)
	}
	if row2.SeqID != "seq-2" || !reflect.DeepEqual(row2.Tokens, p2) {
		t.Fatalf("row2 corrupted: %+v", row2)
	}
	if row1.RowIndex == row2.RowIndex {
		t.Fatalf("row indices collided: %d vs %d", row1.RowIndex, row2.RowIndex)
	}
}

// TestMixedPrefillDecode verifies co-scheduling of chunked prefill and active decode rows.
func TestMixedPrefillDecode(t *testing.T) {
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    4,
		MaxBudgetTokens: 32,
		ChunkSize:       16,
	}, nil)

	// Submit decode request (1-token prompt to enter decode immediately)
	_ = sched.Submit(&cudabatch.Request{ID: "dec-1", PromptTokens: []int32{10}, MaxTokens: 5})
	iter1, _ := sched.Schedule()
	_ = sched.CompleteStep(iter1, nil, nil) // prefill finished -> state Decoding

	// Submit long prefill request
	_ = sched.Submit(&cudabatch.Request{ID: "pref-1", PromptTokens: make([]int32, 40), MaxTokens: 5})

	// Schedule iteration 2: should contain 1 decode row and 1 prefill chunk
	iter2, err := sched.Schedule()
	if err != nil {
		t.Fatal(err)
	}

	if len(iter2.DecodeSeqIDs) != 1 || iter2.DecodeSeqIDs[0] != "dec-1" {
		t.Fatalf("expected dec-1 in decodes, got %v", iter2.DecodeSeqIDs)
	}
	if len(iter2.PrefillChunks) != 1 || iter2.PrefillChunks[0].SeqID != "pref-1" {
		t.Fatalf("expected pref-1 in prefills, got %v", iter2.PrefillChunks)
	}
	if iter2.TotalTokens != 1+16 { // 1 decode token + 16 prefill chunk tokens
		t.Fatalf("total tokens = %d, want 17", iter2.TotalTokens)
	}
}

// TestShapeBucketBinding verifies binding batch rows to shape buckets, causal attention masks,
// and bounds checking.
func TestShapeBucketBinding(t *testing.T) {
	buckets := []cudabatch.ShapeBucket{
		{BatchSize: 2, MaxSeqLen: 64, MaxTokens: 128},
		{BatchSize: 4, MaxSeqLen: 128, MaxTokens: 512},
		{BatchSize: 8, MaxSeqLen: 256, MaxTokens: 2048},
	}

	bucket, err := cudabatch.SelectBucket(3, 100, buckets)
	if err != nil {
		t.Fatalf("SelectBucket: %v", err)
	}
	if bucket.BatchSize != 4 || bucket.MaxSeqLen != 128 {
		t.Fatalf("selected bucket %+v, want BatchSize=4 MaxSeqLen=128", bucket)
	}

	rows := []cudabatch.BatchRowBinding{
		{
			RowIndex:       0,
			SeqID:          "row-0",
			IsPrefill:      true,
			Tokens:         []int32{1, 2, 3},
			StartPos:       0,
			SequenceLength: 3,
			BlockTable:     []int{0},
		},
		{
			RowIndex:       1,
			SeqID:          "row-1",
			IsPrefill:      false,
			Tokens:         nil,
			StartPos:       10,
			SequenceLength: 11,
			BlockTable:     []int{1},
		},
	}

	meta, err := cudabatch.ValidateAndBindRows(rows, bucket, 16)
	if err != nil {
		t.Fatalf("ValidateAndBindRows: %v", err)
	}

	if meta.BatchShape.Dim(0) != 4 || meta.BatchShape.Dim(1) != 128 {
		t.Fatalf("unexpected shape: %v", meta.BatchShape.Dims())
	}

	// Verify mask for row 0: first 3 entries 1.0, rest 0.0
	for pos := 0; pos < 128; pos++ {
		val := meta.AttentionMask[pos]
		if pos < 3 && val != 1.0 {
			t.Fatalf("row 0 pos %d mask = %f, want 1.0", pos, val)
		} else if pos >= 3 && val != 0.0 {
			t.Fatalf("row 0 pos %d mask = %f, want 0.0", pos, val)
		}
	}
}

// TestServingEngineIntegration tests deterministic end-to-end continuous batch execution
// with HostDeterministicEngine.
func TestServingEngineIntegration(t *testing.T) {
	alloc := cudakv.NewBlockAllocator(16, 20)
	tracker := cudabatch.NewMemoryBudgetTracker(alloc)
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    4,
		MaxBudgetTokens: 64,
		ChunkSize:       16,
	}, tracker)

	engine := cudabatch.NewHostDeterministicEngine(0, []int32{42, 43, 44})
	bucket := cudabatch.ShapeBucket{BatchSize: 4, MaxSeqLen: 64, MaxTokens: 256}

	_ = sched.Submit(&cudabatch.Request{ID: "client-1", PromptTokens: []int32{1, 2}, MaxTokens: 2})
	_ = sched.Submit(&cudabatch.Request{ID: "client-2", PromptTokens: []int32{3, 4}, MaxTokens: 2})

	ctx := context.Background()

	// Step 1: Prefill iteration
	iter1, err := sched.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	meta1, err := cudabatch.ValidateAndBindRows(iter1.AllocatedRows, bucket, 16)
	if err != nil {
		t.Fatal(err)
	}
	gen1, rowErr1, err := engine.ExecuteForward(ctx, &cudabatch.ExecutionStep{
		StepID:    iter1.StepID,
		Iteration: iter1,
		Metadata:  meta1,
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = sched.CompleteStep(iter1, gen1, rowErr1)

	// Step 2: Decode iteration 1
	iter2, _ := sched.Schedule()
	meta2, _ := cudabatch.ValidateAndBindRows(iter2.AllocatedRows, bucket, 16)
	gen2, rowErr2, _ := engine.ExecuteForward(ctx, &cudabatch.ExecutionStep{
		StepID:    iter2.StepID,
		Iteration: iter2,
		Metadata:  meta2,
	})
	_ = sched.CompleteStep(iter2, gen2, rowErr2)

	// Step 3: Decode iteration 2 (reaches max_tokens=2)
	iter3, _ := sched.Schedule()
	meta3, _ := cudabatch.ValidateAndBindRows(iter3.AllocatedRows, bucket, 16)
	gen3, rowErr3, _ := engine.ExecuteForward(ctx, &cudabatch.ExecutionStep{
		StepID:    iter3.StepID,
		Iteration: iter3,
		Metadata:  meta3,
	})
	_ = sched.CompleteStep(iter3, gen3, rowErr3)

	r1, _ := sched.Request("client-1")
	r2, _ := sched.Request("client-2")

	if r1.State != cudabatch.StateCompleted || r2.State != cudabatch.StateCompleted {
		t.Fatalf("requests did not complete: r1=%v r2=%v", r1.State, r2.State)
	}
	if len(r1.GeneratedTokens) != 2 || len(r2.GeneratedTokens) != 2 {
		t.Fatalf("generated token count mismatch: r1=%d r2=%d", len(r1.GeneratedTokens), len(r2.GeneratedTokens))
	}
	// Blocks must be completely returned to pool
	if alloc.NumFreeBlocks() != 20 {
		t.Fatalf("blocks leaked: free = %d, want 20", alloc.NumFreeBlocks())
	}
}

// Negative Controls

// Negative control 1: Exhaust KV blocks rejects admission or keeps request waiting
func TestNegativeControl_ExhaustKVBlocks(t *testing.T) {
	alloc := cudakv.NewBlockAllocator(16, 2) // only 2 blocks total
	tracker := cudabatch.NewMemoryBudgetTracker(alloc)
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    4,
		MaxBudgetTokens: 64,
		ChunkSize:       16,
	}, tracker)

	// Request 1 takes 2 blocks (32 tokens) -> exhausts pool
	_ = sched.Submit(&cudabatch.Request{ID: "req-fill", PromptTokens: make([]int32, 32), MaxTokens: 2})
	iter1, err := sched.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	if len(iter1.AllocatedRows) != 1 {
		t.Fatalf("expected 1 row admitted, got %d", len(iter1.AllocatedRows))
	}
	if alloc.NumFreeBlocks() != 0 {
		t.Fatalf("expected 0 free blocks, got %d", alloc.NumFreeBlocks())
	}

	// Request 2 requires 1 block -> cannot be admitted, must stay in waiting queue
	_ = sched.Submit(&cudabatch.Request{ID: "req-starved", PromptTokens: make([]int32, 16), MaxTokens: 2})
	iter2, err := sched.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range iter2.AllocatedRows {
		if row.SeqID == "req-starved" {
			t.Fatalf("starved request was admitted despite zero free blocks!")
		}
	}
	rStarved, _ := sched.Request("req-starved")
	if rStarved.State != cudabatch.StateWaiting {
		t.Fatalf("req-starved state = %v, want waiting", rStarved.State)
	}
}

// Negative control 2: Cancel during active invocation does not cause panic or corruption
func TestNegativeControl_CancelDuringActiveInvocation(t *testing.T) {
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    2,
		MaxBudgetTokens: 32,
		ChunkSize:       16,
	}, nil)

	_ = sched.Submit(&cudabatch.Request{ID: "cancel-me", PromptTokens: []int32{1, 2}, MaxTokens: 5})
	iter, _ := sched.Schedule()

	// Concurrent cancel during active forward step
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(2 * time.Millisecond)
		_ = sched.Cancel("cancel-me")
	}()

	time.Sleep(5 * time.Millisecond)
	_ = sched.CompleteStep(iter, map[string]int32{"cancel-me": 99}, nil)
	wg.Wait()

	r, ok := sched.Request("cancel-me")
	if !ok || r.State != cudabatch.StateCanceled {
		t.Fatalf("expected canceled state, got %+v", r)
	}
}

// Negative control 3: Corrupt row block table is detected and rejected with ErrCorruptBlockTable
func TestNegativeControl_CorruptRowBlockTable(t *testing.T) {
	bucket := cudabatch.ShapeBucket{BatchSize: 2, MaxSeqLen: 64, MaxTokens: 128}

	corruptRows := []cudabatch.BatchRowBinding{
		{
			RowIndex:       0,
			SeqID:          "corrupt-seq",
			IsPrefill:      true,
			Tokens:         []int32{1, 2, 3},
			StartPos:       0,
			SequenceLength: 32,
			BlockTable:     []int{0}, // needs 2 blocks (blockSize=16), provided only 1!
		},
	}

	_, err := cudabatch.ValidateAndBindRows(corruptRows, bucket, 16)
	if err == nil {
		t.Fatal("expected error on insufficient block table, got nil")
	}

	// Corrupt with negative block index
	negativeBlockRows := []cudabatch.BatchRowBinding{
		{
			RowIndex:       0,
			SeqID:          "neg-seq",
			IsPrefill:      true,
			Tokens:         []int32{1, 2, 3},
			StartPos:       0,
			SequenceLength: 16,
			BlockTable:     []int{-1}, // invalid negative block index!
		},
	}
	_, err = cudabatch.ValidateAndBindRows(negativeBlockRows, bucket, 16)
	if err == nil {
		t.Fatal("expected error on negative block table index, got nil")
	}
}

// Negative control 4: Starve queued long prefill under steady decode traffic
// Tests that PolicyFairShare prevents decode traffic from starving long prefill
func TestNegativeControl_StarveQueuedLongPrefillUnderSteadyDecode(t *testing.T) {
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    18,
		MaxBudgetTokens: 16,
		ChunkSize:       8,
		FairnessPolicy:  cudabatch.PolicyFairShare,
	}, nil)

	// Add 16 active decode requests (each requests 1 token, total 16 tokens == budget)
	for i := 1; i <= 16; i++ {
		id := fmt.Sprintf("dec-%d", i)
		_ = sched.Submit(&cudabatch.Request{ID: id, PromptTokens: []int32{int32(i)}, MaxTokens: 100})
	}
	// Run first schedule to admit and transition all to decode
	iter1, _ := sched.Schedule()
	_ = sched.CompleteStep(iter1, nil, nil)

	// Now submit long prefill (100 tokens)
	_ = sched.Submit(&cudabatch.Request{ID: "long-prefill", PromptTokens: make([]int32, 100), MaxTokens: 10})

	// Run schedule: with PolicyFairShare, long-prefill MUST receive tokens despite heavy decode traffic
	iter2, err := sched.Schedule()
	if err != nil {
		t.Fatal(err)
	}

	prefillFound := false
	for _, chunk := range iter2.PrefillChunks {
		if chunk.SeqID == "long-prefill" {
			prefillFound = true
			break
		}
	}
	if !prefillFound {
		t.Fatalf("long-prefill was completely starved under steady decode traffic under PolicyFairShare!")
	}
}
