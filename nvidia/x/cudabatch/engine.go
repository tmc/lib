package cudabatch

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ExecutionStep represents a scheduled batched forward step ready for execution.
type ExecutionStep struct {
	StepID      int64
	Iteration   *ScheduledIteration
	Metadata    *BatchMetadata
	CancelWait  <-chan struct{}
	ReleaseHook func() // Releases temporary output/step buffers
}

// ExecutionEngine models the low-level model forward execution (host test double or device plan).
type ExecutionEngine interface {
	// ExecuteForward executes a forward pass on the scheduled batch.
	// It returns the newly sampled/generated token per row (for decode rows) and any row errors.
	ExecuteForward(ctx context.Context, step *ExecutionStep) (generatedTokens map[string]int32, rowErrors map[string]error, err error)
}

// HostDeterministicEngine is a deterministic host-only test engine for testing
// continuous batching, prefill chunking, decode loops, and cancellation without a GPU.
type HostDeterministicEngine struct {
	mu           sync.Mutex
	delayPerStep time.Duration
	stepCount    int
	// Token generation strategy: maps prompt token -> generated token sequence
	outputSequence []int32
}

// NewHostDeterministicEngine creates a host test engine with optional artificial delay.
func NewHostDeterministicEngine(delay time.Duration, outputSequence []int32) *HostDeterministicEngine {
	return &HostDeterministicEngine{
		delayPerStep:   delay,
		outputSequence: outputSequence,
	}
}

// ExecuteForward simulates executing a continuous batch iteration on host.
func (e *HostDeterministicEngine) ExecuteForward(ctx context.Context, step *ExecutionStep) (map[string]int32, map[string]error, error) {
	if step == nil || step.Metadata == nil {
		return nil, nil, fmt.Errorf("cudabatch: nil execution step or metadata")
	}

	// Validate bounds of all rows before processing
	for _, row := range step.Metadata.RowBindings {
		if row.SequenceLength > step.Metadata.BatchShape.Dim(1) {
			return nil, nil, fmt.Errorf("%w: row seqLen %d exceeds shape %d", ErrRowBoundsViolation, row.SequenceLength, step.Metadata.BatchShape.Dim(1))
		}
	}

	if e.delayPerStep > 0 {
		select {
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		case <-time.After(e.delayPerStep):
		}
	}

	e.mu.Lock()
	e.stepCount++
	stepNum := e.stepCount
	e.mu.Unlock()

	genTokens := make(map[string]int32)
	rowErrors := make(map[string]error)

	for _, row := range step.Metadata.RowBindings {
		if !row.IsPrefill {
			// Generate deterministic next token
			var tok int32 = int32(1000 + stepNum)
			if len(e.outputSequence) > 0 {
				tok = e.outputSequence[(stepNum-1)%len(e.outputSequence)]
			}
			genTokens[row.SeqID] = tok
		}
	}

	return genTokens, rowErrors, nil
}
