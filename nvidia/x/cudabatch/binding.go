package cudabatch

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"strconv"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

// BatchRowBinding binds a single sequence row in a batched forward pass to validated
// tensor shapes, attention masks, sequence lengths, and physical KV block tables.
type BatchRowBinding struct {
	RowIndex        int
	SeqID           string
	IsPrefill       bool
	IsLastPrefill   bool    // Final prompt chunk: its logits supply the first generated token
	Tokens          []int32 // Input tokens for this iteration (chunk tokens for prefill, 1 token for decode)
	StartPos        int     // Offset within full sequence
	SequenceLength  int     // Total active length (StartPos + len(Tokens))
	BlockTable      []int   // Physical block indices in KV pool
	ContinuationVis ContinuationVisibility
}

// ShapeBucket defines a discretized batch shape to reuse compiled plans.
type ShapeBucket struct {
	BatchSize int // Number of rows in bucket
	MaxSeqLen int // Maximum sequence length dimension
	MaxTokens int // Maximum tokens flattened
}

// SelectBucket picks the smallest ShapeBucket capable of containing batchSize and maxSeqLen.
func SelectBucket(batchSize, maxSeqLen int, buckets []ShapeBucket) (ShapeBucket, error) {
	if batchSize <= 0 || maxSeqLen <= 0 {
		return ShapeBucket{}, fmt.Errorf("%w: non-positive dimensions (batch=%d, maxSeqLen=%d)", ErrRowBoundsViolation, batchSize, maxSeqLen)
	}
	var best ShapeBucket
	found := false
	for _, b := range buckets {
		if _, err := bindingBucketSize(b); err != nil {
			return ShapeBucket{}, err
		}
		if b.BatchSize >= batchSize && b.MaxSeqLen >= maxSeqLen {
			if !found || (b.BatchSize <= best.BatchSize && b.MaxSeqLen <= best.MaxSeqLen) {
				best = b
				found = true
			}
		}
	}
	if !found {
		return ShapeBucket{}, fmt.Errorf("%w: no bucket accommodates batchSize=%d maxSeqLen=%d", ErrRowBoundsViolation, batchSize, maxSeqLen)
	}
	return best, nil
}

// BatchMetadata holds validated layout, attention mask, and sequence offsets for an iteration.
type BatchMetadata struct {
	BatchShape     cudashape.Shape
	TotalTokens    int
	MaxTokens      int
	BlockSize      int
	RowBindings    []BatchRowBinding
	SequenceLens   []int
	AttentionMask  []float32 // Flattened [BatchSize, MaxSeqLen] prefix-validity mask, not per-query causal masking
	BlockTableFlat []int32   // Flattened [BatchSize, MaxBlocksPerRow]
	allocation     *batchAllocation
}

type batchAllocation struct {
	pool        *cudakv.BlockAllocator
	sequenceIDs []int64
	generations []uint64
	rowIDs      []string
}

// BindAllocatedRows validates rows against their actual pool allocations before
// constructing metadata. sequenceIDs maps rows to numeric allocator owners; it
// is not inferred from block indices or user-facing sequence names. The caller
// must serialize pool access and retain these allocations through execution.
// This checks pool bookkeeping, not the device storage or the model's KV ABI.
func BindAllocatedRows(rows []BatchRowBinding, bucket ShapeBucket, pool *cudakv.BlockAllocator, sequenceIDs []int64) (*BatchMetadata, error) {
	if pool == nil || len(rows) == 0 || len(sequenceIDs) != len(rows) {
		return nil, fmt.Errorf("%w: actual pool and row owners required", ErrCorruptBlockTable)
	}
	owner := &batchAllocation{pool: pool, sequenceIDs: append([]int64(nil), sequenceIDs...), generations: make([]uint64, len(rows)), rowIDs: make([]string, len(rows))}
	for i, row := range rows {
		owner.rowIDs[i] = row.SeqID
		for j := 0; j < i; j++ {
			if sequenceIDs[j] == sequenceIDs[i] {
				return nil, fmt.Errorf("%w: duplicate row owner", ErrCorruptBlockTable)
			}
		}
		generation, ok := pool.AllocationGeneration(sequenceIDs[i])
		if !ok {
			return nil, fmt.Errorf("%w: row %d has no allocation", ErrCorruptBlockTable, i)
		}
		owner.generations[i] = generation
		if err := pool.ValidateSequence(sequenceIDs[i], generation, row.BlockTable, row.SequenceLength); err != nil {
			return nil, fmt.Errorf("%w: row %d: %v", ErrCorruptBlockTable, i, err)
		}
	}
	metadata, err := ValidateAndBindRows(rows, bucket, pool.BlockSize())
	if err != nil {
		return nil, err
	}
	metadata.allocation = owner
	return metadata, nil
}

func (m *BatchMetadata) validateAllocation() error {
	owner := m.allocation
	if owner == nil || owner.pool == nil || len(owner.sequenceIDs) != len(m.RowBindings) || len(owner.generations) != len(m.RowBindings) || len(owner.rowIDs) != len(m.RowBindings) || owner.pool.BlockSize() != m.BlockSize {
		return fmt.Errorf("%w: allocation-bound metadata required", ErrCorruptBlockTable)
	}
	for i, row := range m.RowBindings {
		if row.SeqID != owner.rowIDs[i] {
			return fmt.Errorf("%w: row owner identity changed", ErrCorruptBlockTable)
		}
		if err := owner.pool.ValidateSequence(owner.sequenceIDs[i], owner.generations[i], row.BlockTable, row.SequenceLength); err != nil {
			return fmt.Errorf("%w: row %d: %v", ErrCorruptBlockTable, i, err)
		}
	}
	return nil
}

// ValidateAndBindRows validates a nonempty batch before allocating its metadata.
// Rows must have distinct nonempty sequence IDs, consecutive row indices, and
// exactly SequenceLength-StartPos tokens (one token for decode). Returned token
// and block-table slices are detached from the caller.
//
// AttentionMask marks each row's valid prefix; the consumer must separately
// enforce per-query causal attention. Block indices must fit int32. This
// function has no KV pool ownership or capacity information and cannot prove
// that a positive block index belongs to a sequence or a physical allocation.
func ValidateAndBindRows(rows []BatchRowBinding, bucket ShapeBucket, blockSize int) (*BatchMetadata, error) {
	maskElements, err := bindingBucketSize(bucket)
	if err != nil {
		return nil, err
	}
	if blockSize <= 0 {
		return nil, fmt.Errorf("%w: non-positive blockSize %d", ErrRowBoundsViolation, blockSize)
	}
	if len(rows) == 0 || len(rows) > bucket.BatchSize {
		return nil, fmt.Errorf("%w: row count %d must be in [1, %d]", ErrRowBoundsViolation, len(rows), bucket.BatchSize)
	}
	maxBlocksPerRow := (bucket.MaxSeqLen-1)/blockSize + 1
	blockElements, err := bindingStorage(bucket.BatchSize, maxBlocksPerRow, 4)
	if err != nil {
		return nil, err
	}

	totalTokens := 0
	for i, row := range rows {
		if row.ContinuationVis < VisibilityDefault || row.ContinuationVis > VisibilityThinkingVisible {
			return nil, fmt.Errorf("%w: invalid row visibility", ErrRowBoundsViolation)
		}
		if row.RowIndex != i || row.SeqID == "" {
			return nil, fmt.Errorf("%w: row %d has invalid identity", ErrRowBoundsViolation, i)
		}
		for j := 0; j < i; j++ {
			if rows[j].SeqID == row.SeqID {
				return nil, fmt.Errorf("%w: duplicate sequence %q", ErrRowBoundsViolation, row.SeqID)
			}
		}
		if row.SequenceLength <= 0 || row.SequenceLength > bucket.MaxSeqLen {
			return nil, fmt.Errorf("%w: row %d (seq %s) length %d exceeds bucket max %d", ErrRowBoundsViolation, i, row.SeqID, row.SequenceLength, bucket.MaxSeqLen)
		}
		if row.StartPos < 0 || row.StartPos >= row.SequenceLength {
			return nil, fmt.Errorf("%w: row %d (seq %s) invalid startPos %d with seqLen %d", ErrRowBoundsViolation, i, row.SeqID, row.StartPos, row.SequenceLength)
		}

		if row.IsLastPrefill && !row.IsPrefill {
			return nil, fmt.Errorf("%w: row %d marks decode as last prefill", ErrRowBoundsViolation, i)
		}

		neededBlocks := (row.SequenceLength-1)/blockSize + 1
		if len(row.BlockTable) < neededBlocks {
			return nil, fmt.Errorf("%w: row %d has %d blocks, needs %d for length %d", ErrCorruptBlockTable, i, len(row.BlockTable), neededBlocks, row.SequenceLength)
		}

		for bi, b := range row.BlockTable {
			if b < 0 || uint64(b) > 1<<31-1 {
				return nil, fmt.Errorf("%w: row %d (seq %s) has invalid int32 block index %d at position %d", ErrCorruptBlockTable, i, row.SeqID, b, bi)
			}
		}
		numRowTokens := len(row.Tokens)
		if numRowTokens == 0 || numRowTokens != row.SequenceLength-row.StartPos || !row.IsPrefill && numRowTokens != 1 {
			return nil, fmt.Errorf("%w: row %d has inconsistent token extent", ErrRowBoundsViolation, i)
		}
		if numRowTokens > bucket.MaxTokens-totalTokens {
			return nil, fmt.Errorf("%w: row %d exceeds bucket token budget %d", ErrCapacityExhausted, i, bucket.MaxTokens)
		}
		totalTokens += numRowTokens
	}

	shape, err := cudashape.New(bucket.BatchSize, bucket.MaxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("cudabatch: construct shape: %w", err)
	}
	seqLens := make([]int, bucket.BatchSize)
	mask := make([]float32, maskElements)
	blockTableFlat := make([]int32, blockElements)
	boundRows := make([]BatchRowBinding, len(rows))
	for i, row := range rows {
		boundRows[i] = row
		boundRows[i].Tokens = append([]int32(nil), row.Tokens...)
		boundRows[i].BlockTable = append([]int(nil), row.BlockTable...)
		seqLens[i] = row.SequenceLength
		for pos := 0; pos < row.SequenceLength; pos++ {
			mask[i*bucket.MaxSeqLen+pos] = 1
		}
		neededBlocks := (row.SequenceLength-1)/blockSize + 1
		for bi, block := range row.BlockTable[:neededBlocks] {
			blockTableFlat[i*maxBlocksPerRow+bi] = int32(block)
		}
	}

	return &BatchMetadata{
		BatchShape:     shape,
		TotalTokens:    totalTokens,
		MaxTokens:      bucket.MaxTokens,
		BlockSize:      blockSize,
		RowBindings:    boundRows,
		SequenceLens:   seqLens,
		AttentionMask:  mask,
		BlockTableFlat: blockTableFlat,
	}, nil
}

func bindingBucketSize(bucket ShapeBucket) (int, error) {
	if bucket.BatchSize <= 0 || bucket.MaxSeqLen <= 0 || bucket.MaxTokens <= 0 {
		return 0, fmt.Errorf("%w: bucket dimensions and token budget must be positive", ErrRowBoundsViolation)
	}
	if _, err := bindingStorage(bucket.BatchSize, 1, strconv.IntSize/8); err != nil {
		return 0, err
	}
	return bindingStorage(bucket.BatchSize, bucket.MaxSeqLen, 4)
}

// bindingStorage checks element and byte extents before forming a slice length.
func bindingStorage(a, b, elementBytes int) (int, error) {
	maxInt := int(^uint(0) >> 1)
	if a <= 0 || b <= 0 || elementBytes <= 0 || a > maxInt/b {
		return 0, fmt.Errorf("%w: metadata element extent overflows int", ErrRowBoundsViolation)
	}
	elements := a * b
	if elements > maxInt/elementBytes || uint64(elements) > cudashape.MaxByteLimit/uint64(elementBytes) {
		return 0, fmt.Errorf("%w: metadata byte extent exceeds addressable storage", ErrRowBoundsViolation)
	}
	return elements, nil
}

// rowNeedsSample reports which completed rows supply a generated token.
func rowNeedsSample(row BatchRowBinding) bool {
	return !row.IsPrefill || row.IsLastPrefill
}
