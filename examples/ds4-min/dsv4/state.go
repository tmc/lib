// Package dsv4 defines the backend-independent contracts used by ds4-min.
//
// A Model owns immutable weights. A Backend may create multiple independent
// States from that model; a State owns the mutable KV, compressor, indexer,
// raw-window, and hyper-connection history for one request. Device-backed
// implementations must keep these values in the device domain and must not
// expose device pointers through this package.
package dsv4

import (
	"errors"
	"io"
)

var (
	// ErrUnimplemented reports that a backend has not implemented independent
	// state yet. Callers must not silently fall back to sharing a session.
	ErrUnimplemented = errors.New("independent backend state is not implemented")

	// ErrClosed reports use of a State, Snapshot, or Backend after Close.
	ErrClosed = errors.New("dsv4 object is closed")

	// ErrDesynchronized reports a session whose backend state may no longer
	// match its token timeline after an evaluation failure. Such a session
	// must be closed instead of being reused.
	ErrDesynchronized = errors.New("dsv4 session state is desynchronized")
)

// Backend owns immutable model weights and creates request-local States.
// Implementations must reject a State created by another Backend.
type Backend interface {
	NewState(context int) (State, error)
	Eval(State, int, int) error
	Logits(State) []float32
	Close() error
}

// BatchEvaluator is an optional backend capability. A batch must be
// observationally equivalent to calling Backend.Eval once per token in
// order, including the final state and logits.
type BatchEvaluator interface {
	EvalBatch(State, []int, int) error
}

// State owns mutable inference history for one request. A State may be used
// by only one request at a time. Snapshot and Save must synchronize pending
// device work before exposing state; failed Restore or Load must leave the
// destination unchanged.
type State interface {
	Reset() error
	Snapshot() (Snapshot, error)
	Restore(Snapshot) error
	Save(io.Writer) error
	Load(io.Reader) error
	Close() error
}

// Snapshot is an immutable, backend-owned copy of a State. It can be restored
// only by a State from the Backend that created it.
type Snapshot interface {
	Close() error
}

// IndependentStates is an optional capability for callers that need to
// distinguish a model owner from a single request state. A backend that does
// not implement it must return ErrUnimplemented from NewState rather than
// aliasing mutable state between requests.
type IndependentStates interface {
	Backend
	NewState(context int) (State, error)
}
