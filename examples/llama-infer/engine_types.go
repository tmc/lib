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
	// SeqPendingPrefill indicates the sequence is waiting for prompt ingestion and KV cache allocation.
	SeqPendingPrefill SequenceStatus = iota
	// SeqDecoding indicates the sequence is actively generating autoregressive tokens.
	SeqDecoding
	// SeqFinished indicates the sequence has completed generation.
	SeqFinished
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
	case SeqFailed:
		return "Failed"
	default:
		return fmt.Sprintf("Status(%d)", int(s))
	}
}

// Sequence represents an isolated generation stream containing its own KV cache state,
// sequence position, activation scratch references, and token history.
type Sequence struct {
	ID              string
	Prompt          string
	PromptTokens    []int32
	GeneratedTokens []int32
	Pos             int
	MaxTokens       int
	Status          SequenceStatus
	Error           error
	OnToken         func(tokenID int32, tokenText string) bool
	DoneCh          chan struct{}
	mu              sync.Mutex

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
		CreatedTime:  time.Now(),
	}
}

// AppendToken records a newly generated token, invokes the streaming callback,
// and updates the sequence status if generation has reached its limit.
func (s *Sequence) AppendToken(tokenID int32, tokenText string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.GeneratedTokens = append(s.GeneratedTokens, tokenID)
	s.Pos++

	shouldContinue := true
	if s.OnToken != nil {
		shouldContinue = s.OnToken(tokenID, tokenText)
	}

	if len(s.GeneratedTokens) >= s.MaxTokens || !shouldContinue {
		s.Status = SeqFinished
		close(s.DoneCh)
		return false
	}
	return true
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

// Scheduler coordinates continuous batching, sequence lifecycles, and plan dispatches.
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

// Step advances active sequences by one execution batch (prefill or decode step).
func (s *Scheduler) Step(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.active) == 0 {
		return nil
	}

	// 1. Prioritize any pending prefill sequence
	for i, seq := range s.active {
		if seq.Status == SeqPendingPrefill {
			if s.engine == nil {
				seq.Status = SeqFinished
				select {
				case <-seq.DoneCh:
				default:
					close(seq.DoneCh)
				}
				s.active = append(s.active[:i], s.active[i+1:]...)
				return nil
			}
			start := time.Now()
			err := s.engine.Generate(ctx, s.cfg, seq.Prompt, seq.MaxTokens, seq.OnToken)
			if err != nil {
				seq.Fail(err)
			} else {
				seq.Status = SeqFinished
				select {
				case <-seq.DoneCh:
				default:
					close(seq.DoneCh)
				}
			}
			seq.PrefillDuration = time.Since(start)
			// Remove from active list
			s.active = append(s.active[:i], s.active[i+1:]...)
			return err
		}
	}

	return nil
}
