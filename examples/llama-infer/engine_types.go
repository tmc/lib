package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// SequenceStatus tracks the lifecycle phase of an ongoing generation sequence.
type SequenceStatus int

const (
	// SeqPendingPrefill indicates the sequence is waiting for prompt ingestion.
	SeqPendingPrefill SequenceStatus = iota
	// SeqDecoding indicates the sequence is actively generating autoregressive tokens.
	SeqDecoding
	// SeqFinished indicates the sequence has completed generation.
	SeqFinished
	// SeqCanceled indicates the sequence was canceled by the client.
	SeqCanceled
	// SeqFailed indicates the sequence encountered an unrecoverable error during execution.
	SeqFailed
)

func (s SequenceStatus) String() string {
	switch s {
	case SeqPendingPrefill:
		return "PendingPrefill"
	case SeqDecoding:
		return "Decoding"
	case SeqFinished:
		return "Finished"
	case SeqCanceled:
		return "Canceled"
	case SeqFailed:
		return "Failed"
	default:
		return fmt.Sprintf("Status(%d)", int(s))
	}
}

// Sequence represents an isolated generation stream containing its own KV cache state,
// sequence position, and token history.
type Sequence struct {
	ID              string
	Prompt          string
	PromptTokens    []int32
	GeneratedTokens []int32
	CurrentToken    int32
	Pos             int
	MaxTokens       int
	Status          SequenceStatus
	Error           error
	OnToken         func(tokenID int32, tokenText string) bool
	DoneCh          chan struct{}
	CancelCh        chan struct{}
	mu              sync.Mutex

	// Isolated KV cache state
	KVCache *KVCacheState

	// Metrics
	PrefillDuration time.Duration
	DecodeDuration  time.Duration
	CreatedTime     time.Time
}

// NewSequence initializes an isolated sequence stream.
func NewSequence(id string, prompt string, promptTokens []int32, maxTokens int, onToken func(tokenID int32, tokenText string) bool) *Sequence {
	if maxTokens <= 0 {
		maxTokens = 64
	}
	return &Sequence{
		ID:           id,
		Prompt:       prompt,
		PromptTokens: promptTokens,
		MaxTokens:    maxTokens,
		Status:       SeqPendingPrefill,
		OnToken:      onToken,
		DoneCh:       make(chan struct{}),
		CancelCh:     make(chan struct{}),
		CreatedTime:  time.Now(),
	}
}

// Cancel requests early termination of the sequence.
func (s *Sequence) Cancel() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.Status == SeqFinished || s.Status == SeqCanceled || s.Status == SeqFailed {
		return
	}
	s.Status = SeqCanceled
	select {
	case <-s.CancelCh:
	default:
		close(s.CancelCh)
	}
	select {
	case <-s.DoneCh:
	default:
		close(s.DoneCh)
	}
}

func (s *Sequence) appendTokenLocked(tokenID int32, tokenText string) bool {
	if s.Status != SeqDecoding {
		return false
	}

	s.GeneratedTokens = append(s.GeneratedTokens, tokenID)
	s.CurrentToken = tokenID

	if len(s.GeneratedTokens) >= s.MaxTokens {
		s.Status = SeqFinished
		select {
		case <-s.DoneCh:
		default:
			close(s.DoneCh)
		}
		return false
	}
	return true
}

// AppendToken records a newly generated token and updates sequence status.
func (s *Sequence) AppendToken(tokenID int32, tokenText string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.appendTokenLocked(tokenID, tokenText)
}

// Fail marks the sequence as failed with the given error.
func (s *Sequence) Fail(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.Error = err
	s.Status = SeqFailed
	select {
	case <-s.DoneCh:
	default:
		close(s.DoneCh)
	}
}

// Scheduler coordinates continuous batching and sequence execution steps.
type Scheduler struct {
	engine   *ModelEngine
	cfg      Config
	mu       sync.Mutex
	active   []*Sequence
	maxBatch int
}

// NewScheduler creates a scheduler for the given model engine and configuration.
func NewScheduler(eng *ModelEngine, cfg Config, maxBatch int) *Scheduler {
	if maxBatch <= 0 {
		maxBatch = 32
	}
	return &Scheduler{
		engine:   eng,
		cfg:      cfg,
		maxBatch: maxBatch,
		active:   make([]*Sequence, 0, maxBatch),
	}
}

// Submit enqueues a new sequence into the scheduler.
func (s *Scheduler) Submit(seq *Sequence) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.active) >= s.maxBatch {
		return fmt.Errorf("scheduler: batch capacity (%d) reached", s.maxBatch)
	}
	s.active = append(s.active, seq)
	return nil
}

// ActiveCount returns the number of active sequences in the scheduler.
func (s *Scheduler) ActiveCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.active)
}

type sequenceStepCallback struct {
	seq       *Sequence
	tokenID   int32
	tokenText string
}

// Step advances each active sequence by a single bounded step (prefill or decode token).
// Callbacks are invoked outside the scheduler lock to avoid contention.
func (s *Scheduler) Step(ctx context.Context) (int, error) {
	s.mu.Lock()
	if len(s.active) == 0 {
		s.mu.Unlock()
		return 0, nil
	}

	// Snapshot active sequences
	activeSeqs := make([]*Sequence, len(s.active))
	copy(activeSeqs, s.active)
	s.mu.Unlock()

	var callbacks []sequenceStepCallback

	for _, seq := range activeSeqs {
		select {
		case <-ctx.Done():
			seq.Cancel()
			continue
		case <-seq.CancelCh:
			continue
		default:
		}

		seq.mu.Lock()
		status := seq.Status
		seq.mu.Unlock()

		if status == SeqPendingPrefill {
			// Ingest prompt
			if s.engine != nil {
				start := time.Now()
				nextTok := GPUPrefillPass(s.engine, s.cfg, seq.PromptTokens)
				seq.mu.Lock()
				seq.PrefillDuration = time.Since(start)
				seq.CurrentToken = nextTok
				seq.Pos = len(seq.PromptTokens)
				seq.Status = SeqDecoding
				seq.GeneratedTokens = append(seq.GeneratedTokens, nextTok)
				seq.mu.Unlock()

				callbacks = append(callbacks, sequenceStepCallback{
					seq:       seq,
					tokenID:   nextTok,
					tokenText: fmt.Sprintf("%d", nextTok),
				})
			} else {
				seq.mu.Lock()
				seq.Status = SeqFinished
				select {
				case <-seq.DoneCh:
				default:
					close(seq.DoneCh)
				}
				seq.mu.Unlock()
			}
		} else if status == SeqDecoding {
			if s.engine != nil {
				seq.mu.Lock()
				curTok := seq.CurrentToken
				pos := seq.Pos
				seq.mu.Unlock()

				start := time.Now()
				nextTok := GPUForwardPass(s.engine, s.cfg, curTok, pos)
				dur := time.Since(start)

				seq.mu.Lock()
				seq.DecodeDuration += dur
				seq.Pos++
				cont := seq.appendTokenLocked(nextTok, fmt.Sprintf("%d", nextTok))
				seq.mu.Unlock()

				callbacks = append(callbacks, sequenceStepCallback{
					seq:       seq,
					tokenID:   nextTok,
					tokenText: fmt.Sprintf("%d", nextTok),
				})

				if !cont {
					// Sequence finished
				}
			}
		}
	}

	// Dispatch callbacks outside scheduler lock
	for _, cb := range callbacks {
		if cb.seq.OnToken != nil {
			cont := cb.seq.OnToken(cb.tokenID, cb.tokenText)
			if !cont {
				cb.seq.Cancel()
			}
		}
	}

	// Prune finished/canceled sequences
	s.mu.Lock()
	defer s.mu.Unlock()

	remaining := s.active[:0]
	for _, seq := range s.active {
		seq.mu.Lock()
		st := seq.Status
		seq.mu.Unlock()
		if st == SeqPendingPrefill || st == SeqDecoding {
			remaining = append(remaining, seq)
		}
	}
	s.active = remaining

	return len(callbacks), nil
}
