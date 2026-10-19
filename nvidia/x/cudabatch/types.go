package cudabatch

import (
	"errors"
	"fmt"
	"time"
)

// RequestState defines deterministic lifecycle states of an inference request.
type RequestState int

const (
	StateWaiting RequestState = iota
	StatePrefilling
	StateDecoding
	StateCompleted
	StateCanceled
)

func (s RequestState) String() string {
	switch s {
	case StateWaiting:
		return "waiting"
	case StatePrefilling:
		return "prefilling"
	case StateDecoding:
		return "decoding"
	case StateCompleted:
		return "completed"
	case StateCanceled:
		return "canceled"
	default:
		return fmt.Sprintf("state(%d)", int(s))
	}
}

// Common errors returned by batch scheduling, admission, and binding.
var (
	ErrRequestNotFound      = errors.New("cudabatch: request not found")
	ErrRequestAlreadyExists = errors.New("cudabatch: request already exists")
	ErrInvalidRequest       = errors.New("cudabatch: invalid request parameters")
	ErrCapacityExhausted    = errors.New("cudabatch: batch capacity or memory exhausted")
	ErrCorruptBlockTable    = errors.New("cudabatch: corrupted block table")
	ErrRowBoundsViolation   = errors.New("cudabatch: row bounds or shape violation")
	ErrExecutionInFlight    = errors.New("cudabatch: execution step in flight")
	ErrWaitingTimeout       = errors.New("cudabatch: request waiting timeout")
)

// ContinuationVisibility defines how tokens are preserved for multi-turn protocols.
type ContinuationVisibility int

const (
	// VisibilityDefault indicates standard public user/assistant context.
	VisibilityDefault ContinuationVisibility = iota
	// VisibilityResponsesVisible preserves intermediate tool call responses and hidden outputs.
	VisibilityResponsesVisible
	// VisibilityThinkingVisible preserves chain-of-thought reasoning tokens across turns.
	VisibilityThinkingVisible
)

func (v ContinuationVisibility) String() string {
	switch v {
	case VisibilityResponsesVisible:
		return "responses-visible"
	case VisibilityThinkingVisible:
		return "thinking-visible"
	default:
		return "default"
	}
}

// Request represents a single user sequence submitted to the batch scheduler.
type Request struct {
	ID           string
	PromptTokens []int32
	MaxTokens    int
	CreatedAt    time.Time
	Visibility   ContinuationVisibility
	SessionID    string // Associated multi-turn conversation or session ID
	// OnFinish, if non-nil, receives one detached terminal snapshot before
	// history retirement. It runs synchronously and must not call the scheduler.
	OnFinish func(*Request)

	// Dynamic lifecycle state
	State            RequestState
	PrefilledTokens  int
	GeneratedTokens  []int32
	BlockTable       []int // Physical KV block indices allocated to this request
	PromptBlockCount int   // Number of prefix blocks retained

	// Execution and cancellation synchronization
	cancelRequested  bool
	activeExecution  bool
	completedAt      time.Time
	err              error
	admissionOrdinal int64
	admittedAt       time.Time
	retainedTerminal bool
}

// IsFinished reports whether the request has reached a terminal state.
func (r *Request) IsFinished() bool {
	return r.State == StateCompleted || r.State == StateCanceled
}

// Err returns the terminal cancellation, timeout or execution error, if any.
func (r *Request) Err() error { return r.err }

func copyRequest(r *Request) *Request {
	cp := *r
	cp.PromptTokens = append([]int32(nil), r.PromptTokens...)
	cp.GeneratedTokens = append([]int32(nil), r.GeneratedTokens...)
	cp.BlockTable = append([]int(nil), r.BlockTable...)
	cp.OnFinish = nil
	return &cp
}

// TotalTokens returns the total prompt plus generated tokens so far.
func (r *Request) TotalTokens() int {
	return len(r.PromptTokens) + len(r.GeneratedTokens)
}

// ContextTokens returns the number of input tokens consumed by successful steps.
// The newest generated token is pending input for the next decode; it is not
// included. Reserved KV capacity and failed or incomplete steps do not count.
func (r *Request) ContextTokens() int {
	if r.State == StateWaiting {
		return 0
	}
	n := r.PrefilledTokens
	if len(r.GeneratedTokens) > 0 {
		n += len(r.GeneratedTokens) - 1
	}
	return n
}

// TotalTokensSlice returns a newly allocated slice containing all prompt and generated tokens.
func (r *Request) TotalTokensSlice() []int32 {
	out := make([]int32, 0, len(r.PromptTokens)+len(r.GeneratedTokens))
	out = append(out, r.PromptTokens...)
	out = append(out, r.GeneratedTokens...)
	return out
}
