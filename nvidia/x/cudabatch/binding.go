package cudabatch

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

// BatchRowBinding binds a single sequence row in a batched forward pass to validated
// tensor shapes, attention masks, sequence lengths, and physical KV block tables.
type BatchRowBinding struct {
	RowIndex        int
	SeqID           string
	IsPrefill       bool
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
	AttentionMask  []float32 // Flattened [BatchSize, MaxSeqLen] causal/prefix mask
	BlockTableFlat []int32   // Flattened [BatchSize, MaxBlocksPerRow]
}

// ValidateAndBindRows validates bounds, block table integrity, and attention mask layout
// for the given rows against bucket constraints and blockSize.
func ValidateAndBindRows(rows []BatchRowBinding, bucket ShapeBucket, blockSize int) (*BatchMetadata, error) {
	if blockSize <= 0 {
		return nil, fmt.Errorf("%w: non-positive blockSize %d", ErrRowBoundsViolation, blockSize)
	}
	if len(rows) > bucket.BatchSize {
		return nil, fmt.Errorf("%w: rows %d exceeds bucket batch size %d", ErrRowBoundsViolation, len(rows), bucket.BatchSize)
	}

	totalTokens := 0
	seqLens := make([]int, bucket.BatchSize)
	maxBlocksPerRow := (bucket.MaxSeqLen + blockSize - 1) / blockSize

	mask := make([]float32, bucket.BatchSize*bucket.MaxSeqLen)
	blockTableFlat := make([]int32, bucket.BatchSize*maxBlocksPerRow)

	for i, row := range rows {
		if row.SequenceLength < 0 || row.SequenceLength > bucket.MaxSeqLen {
			return nil, fmt.Errorf("%w: row %d (seq %s) length %d exceeds bucket max %d", ErrRowBoundsViolation, i, row.SeqID, row.SequenceLength, bucket.MaxSeqLen)
		}
		if row.StartPos < 0 || row.StartPos > row.SequenceLength {
			return nil, fmt.Errorf("%w: row %d (seq %s) invalid startPos %d with seqLen %d", ErrRowBoundsViolation, i, row.SeqID, row.StartPos, row.SequenceLength)
		}

		// Verify block table covers the entire sequence length
		neededBlocks := (row.SequenceLength + blockSize - 1) / blockSize
		if len(row.BlockTable) < neededBlocks {
			return nil, fmt.Errorf("%w: row %d has %d blocks, needs %d for length %d", ErrCorruptBlockTable, i, len(row.BlockTable), neededBlocks, row.SequenceLength)
		}

		// Validate block table indices are non-negative and not corrupted
		for bi, b := range row.BlockTable[:neededBlocks] {
			if b < 0 {
				return nil, fmt.Errorf("%w: row %d (seq %s) has negative block index %d at position %d", ErrCorruptBlockTable, i, row.SeqID, b, bi)
			}
			blockTableFlat[i*maxBlocksPerRow+bi] = int32(b)
		}

		numRowTokens := len(row.Tokens)
		if !row.IsPrefill {
			numRowTokens = 1
		}
		totalTokens += numRowTokens
		seqLens[i] = row.SequenceLength

		// Construct causal attention mask for this row: 1.0 for valid attended positions, 0.0 for future/padding
		rowOffset := i * bucket.MaxSeqLen
		for pos := 0; pos < bucket.MaxSeqLen; pos++ {
			if pos < row.SequenceLength {
				mask[rowOffset+pos] = 1.0
			} else {
				mask[rowOffset+pos] = 0.0
			}
		}
	}

	shape, err := cudashape.New(bucket.BatchSize, bucket.MaxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("cudabatch: construct shape: %w", err)
	}

	return &BatchMetadata{
		BatchShape:     shape,
		TotalTokens:    totalTokens,
		MaxTokens:      bucket.MaxTokens,
		BlockSize:      blockSize,
		RowBindings:    append([]BatchRowBinding(nil), rows...),
		SequenceLens:   seqLens,
		AttentionMask:  mask,
		BlockTableFlat: blockTableFlat,
	}, nil
}
