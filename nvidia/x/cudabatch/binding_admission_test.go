package cudabatch_test

import (
	"errors"
	"math"
	"reflect"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

func bindingRow() cudabatch.BatchRowBinding {
	return cudabatch.BatchRowBinding{RowIndex: 0, SeqID: "row", IsPrefill: true, Tokens: []int32{7, 8}, SequenceLength: 2, BlockTable: []int{3}}
}

func TestBindingAdmissionBudget(t *testing.T) {
	meta, err := cudabatch.ValidateAndBindRows([]cudabatch.BatchRowBinding{bindingRow()}, cudabatch.ShapeBucket{BatchSize: 1, MaxSeqLen: 2, MaxTokens: 1}, 4)
	if meta != nil || !errors.Is(err, cudabatch.ErrCapacityExhausted) {
		t.Fatalf("over-budget row: metadata=%+v err=%v", meta, err)
	}
}

func TestBindingAdmissionOwnership(t *testing.T) {
	rows := []cudabatch.BatchRowBinding{bindingRow()}
	meta, err := cudabatch.ValidateAndBindRows(rows, cudabatch.ShapeBucket{BatchSize: 2, MaxSeqLen: 5, MaxTokens: 2}, 4)
	if err != nil {
		t.Fatal(err)
	}
	if meta.TotalTokens != 2 || !reflect.DeepEqual(meta.SequenceLens, []int{2, 0}) || !reflect.DeepEqual(meta.AttentionMask, []float32{1, 1, 0, 0, 0, 0, 0, 0, 0, 0}) || !reflect.DeepEqual(meta.BlockTableFlat, []int32{3, 0, 0, 0}) {
		t.Fatalf("prefix/padding metadata changed: %+v", meta)
	}
	rows[0].Tokens[0] = 99
	rows[0].BlockTable[0] = 12
	rows[0].StartPos = 1
	if meta.RowBindings[0].Tokens[0] != 7 || meta.RowBindings[0].BlockTable[0] != 3 || meta.RowBindings[0].StartPos != 0 {
		t.Fatalf("metadata aliases caller rows: %+v", meta.RowBindings)
	}
	meta.RowBindings[0].Tokens[1] = 44
	meta.RowBindings[0].BlockTable[0] = 77
	if rows[0].Tokens[1] != 8 || rows[0].BlockTable[0] != 12 {
		t.Fatal("caller slices alias metadata")
	}
}

func TestBindingAdmissionBounds(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tt := range []struct {
		name   string
		bucket cudabatch.ShapeBucket
		block  int
	}{
		{"zero batch", cudabatch.ShapeBucket{0, 2, 2}, 4},
		{"negative sequence", cudabatch.ShapeBucket{1, -1, 2}, 4},
		{"zero budget", cudabatch.ShapeBucket{1, 2, 0}, 4},
		{"negative budget", cudabatch.ShapeBucket{1, 2, -1}, 4},
		{"zero block", cudabatch.ShapeBucket{1, 2, 2}, 0},
		// Without admission these values overflow element or byte arithmetic;
		// they are not representable large allocations or memory-pressure probes.
		{"mask element overflow", cudabatch.ShapeBucket{2, maxInt, 2}, 1},
		{"mask byte overflow", cudabatch.ShapeBucket{1, maxInt, 2}, maxInt},
		{"sequence byte overflow", cudabatch.ShapeBucket{maxInt, 1, 2}, 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			meta, err := cudabatch.ValidateAndBindRows([]cudabatch.BatchRowBinding{bindingRow()}, tt.bucket, tt.block)
			if meta != nil || !errors.Is(err, cudabatch.ErrRowBoundsViolation) {
				t.Fatalf("invalid bucket admitted: %+v, %v", meta, err)
			}
		})
	}
	// A maximum-size block is safe for a small sequence: ceil(2/block)=1.
	meta, err := cudabatch.ValidateAndBindRows([]cudabatch.BatchRowBinding{bindingRow()}, cudabatch.ShapeBucket{1, 2, 2}, maxInt)
	if err != nil || len(meta.BlockTableFlat) != 1 || meta.BlockTableFlat[0] != 3 {
		t.Fatalf("overflow-free block ceiling: %+v, %v", meta, err)
	}
	if meta, err := cudabatch.ValidateAndBindRows(nil, cudabatch.ShapeBucket{1, 2, 2}, 4); meta != nil || !errors.Is(err, cudabatch.ErrRowBoundsViolation) {
		t.Fatalf("empty batch admitted: %+v, %v", meta, err)
	}
}

func TestBindingAdmissionRows(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*cudabatch.BatchRowBinding)
		want   error
	}{
		{"row index", func(r *cudabatch.BatchRowBinding) { r.RowIndex = 1 }, cudabatch.ErrRowBoundsViolation},
		{"missing sequence", func(r *cudabatch.BatchRowBinding) { r.SeqID = "" }, cudabatch.ErrRowBoundsViolation},
		{"negative start", func(r *cudabatch.BatchRowBinding) { r.StartPos = -1 }, cudabatch.ErrRowBoundsViolation},
		{"empty extent", func(r *cudabatch.BatchRowBinding) { r.StartPos = 2 }, cudabatch.ErrRowBoundsViolation},
		{"token extent", func(r *cudabatch.BatchRowBinding) { r.SequenceLength = 3 }, cudabatch.ErrRowBoundsViolation},
		{"empty prefill", func(r *cudabatch.BatchRowBinding) { r.Tokens = nil }, cudabatch.ErrRowBoundsViolation},
		{"empty decode", func(r *cudabatch.BatchRowBinding) { r.IsPrefill = false; r.Tokens = nil }, cudabatch.ErrRowBoundsViolation},
		{"two decode tokens", func(r *cudabatch.BatchRowBinding) { r.IsPrefill = false }, cudabatch.ErrRowBoundsViolation},
		{"missing blocks", func(r *cudabatch.BatchRowBinding) { r.BlockTable = nil }, cudabatch.ErrCorruptBlockTable},
		{"negative unused block", func(r *cudabatch.BatchRowBinding) { r.BlockTable = []int{3, -1} }, cudabatch.ErrCorruptBlockTable},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := bindingRow()
			tt.change(&r)
			meta, err := cudabatch.ValidateAndBindRows([]cudabatch.BatchRowBinding{r}, cudabatch.ShapeBucket{1, 4, 4}, 4)
			if meta != nil || !errors.Is(err, tt.want) {
				t.Fatalf("invalid row admitted: %+v, %v", meta, err)
			}
		})
	}
	rows := []cudabatch.BatchRowBinding{bindingRow(), bindingRow()}
	rows[1].RowIndex = 1
	if meta, err := cudabatch.ValidateAndBindRows(rows, cudabatch.ShapeBucket{2, 2, 4}, 4); meta != nil || !errors.Is(err, cudabatch.ErrRowBoundsViolation) {
		t.Fatalf("duplicate sequence admitted: %+v, %v", meta, err)
	}
	if strconv.IntSize == 64 {
		r := bindingRow()
		large := int64(math.MaxInt32) + 1
		r.BlockTable[0] = int(large)
		if meta, err := cudabatch.ValidateAndBindRows([]cudabatch.BatchRowBinding{r}, cudabatch.ShapeBucket{1, 2, 2}, 4); meta != nil || !errors.Is(err, cudabatch.ErrCorruptBlockTable) {
			t.Fatalf("lossy block conversion admitted: %+v, %v", meta, err)
		}
	}
}

func TestBindingAdmissionDecodeProducer(t *testing.T) {
	allocator := cudakv.NewBlockAllocator(4, 8)
	scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 1, MaxBudgetTokens: 2, ChunkSize: 2}, cudabatch.NewMemoryBudgetTracker(allocator))
	req := &cudabatch.Request{ID: "decode", PromptTokens: []int32{10, 11}, MaxTokens: 3}
	if err := scheduler.Submit(req); err != nil {
		t.Fatal(err)
	}
	prefill, err := scheduler.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	if err := scheduler.CompleteStep(prefill, nil, nil); err != nil {
		t.Fatal(err)
	}
	for i, want := range []int32{11, 42} {
		iteration, err := scheduler.Schedule()
		if err != nil {
			t.Fatal(err)
		}
		row := iteration.AllocatedRows[0]
		if !reflect.DeepEqual(row.Tokens, []int32{want}) || row.IsPrefill || row.StartPos != 2+i || row.SequenceLength != 3+i {
			t.Fatalf("retained token/extent: %+v, want token %d", row, want)
		}
		metadata, err := cudabatch.ValidateAndBindRows(iteration.AllocatedRows, cudabatch.ShapeBucket{1, 8, 2}, 4)
		if err != nil || metadata.TotalTokens != 1 || metadata.SequenceLens[0] != 3+i {
			t.Fatalf("producer-to-binder: %+v, %v", metadata, err)
		}
		// A detached row token must not overwrite either retained source slice.
		row.Tokens[0] = 99
		if req.PromptTokens[1] != 11 || i > 0 && req.GeneratedTokens[0] != 42 {
			t.Fatal("decode row aliases retained tokens")
		}
		if err := scheduler.CompleteStep(iteration, map[string]int32{"decode": 42}, nil); err != nil {
			t.Fatal(err)
		}
	}
	// Submit currently retains req. A caller can erase its tokens; Schedule
	// must refuse before changing KV allocation, rather than indexing nil.
	req.PromptTokens, req.GeneratedTokens = nil, nil
	free := allocator.NumFreeBlocks()
	if iteration, err := scheduler.Schedule(); iteration != nil || !errors.Is(err, cudabatch.ErrInvalidRequest) || allocator.NumFreeBlocks() != free {
		t.Fatalf("empty retained tokens: %+v, %v", iteration, err)
	}
}

func TestSelectBucketAdmission(t *testing.T) {
	valid := cudabatch.ShapeBucket{2, 8, 4}
	if got, err := cudabatch.SelectBucket(1, 5, []cudabatch.ShapeBucket{valid}); err != nil || got != valid {
		t.Fatalf("valid bucket: %+v, %v", got, err)
	}
	for _, invalid := range []cudabatch.ShapeBucket{{2, 8, 0}, {2, -8, 4}, {2, int(^uint(0) >> 1), 4}} {
		if _, err := cudabatch.SelectBucket(1, 5, []cudabatch.ShapeBucket{invalid, valid}); !errors.Is(err, cudabatch.ErrRowBoundsViolation) {
			t.Fatalf("invalid candidate %+v: %v", invalid, err)
		}
	}
}
