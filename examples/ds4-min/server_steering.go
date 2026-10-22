package main

import (
	"errors"
	"fmt"
	"io"
	"reflect"
	"sync"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// steeringBackend owns immutable host directions and startup controls. Each
// request copies controls into its own State and owns its device directions.
// The owner also retains failed request cleanup until the foundation can close.
type steeringBackend struct {
	dsv4.IndependentStates
	vectors         *steeringVectors
	attention, ffn  float32
	mu              sync.Mutex
	closeMu         sync.Mutex
	closing, closed bool
	states          map[*steeringRequestState]bool
	pending         map[*steeringRequestCleanup]bool
}

func newSteeringBackend(backend dsv4.IndependentStates, cfg config) (dsv4.IndependentStates, error) {
	if cfg.dirSteeringAttn == 0 && cfg.dirSteeringFFN == 0 {
		return backend, nil
	}
	vectors, err := loadSteeringVectors(cfg.dirSteeringFile)
	if err != nil {
		return nil, err
	}
	return &steeringBackend{IndependentStates: backend, vectors: vectors,
		attention: float32(cfg.dirSteeringAttn), ffn: float32(cfg.dirSteeringFFN),
		states: make(map[*steeringRequestState]bool), pending: make(map[*steeringRequestCleanup]bool)}, nil
}

func (b *steeringBackend) NewState(size int) (dsv4.State, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closing || b.closed {
		return nil, dsv4.ErrClosed
	}
	state, err := b.IndependentStates.NewState(size)
	if err != nil {
		owners := backendErrorOwners(err)
		if state != nil {
			owners = appendSteeringOwner(owners, state)
		}
		return nil, b.retainLocked(err, owners)
	}
	if state == nil {
		return nil, errors.New("steering backend returned nil state")
	}
	loader, ok := state.(steeringLoader)
	if !ok {
		err = dsv4.ErrUnimplemented
	} else {
		err = loader.loadSteering(b.vectors, b.attention, b.ffn)
	}
	if err != nil {
		owners := backendErrorOwners(err)
		closeErr := state.Close()
		if closeErr != nil {
			owners = append(owners, state)
		}
		return nil, b.retainLocked(errors.Join(fmt.Errorf("configure request steering: %w", err), closeErr), owners)
	}
	s := &steeringRequestState{state: state, owner: b}
	b.states[s] = true
	return s, nil
}

func (b *steeringBackend) retainLocked(err error, owners []io.Closer) error {
	if len(owners) == 0 {
		return err
	}
	c := &steeringRequestCleanup{cause: err, owners: backendCleanupGroup{owners}, backend: b}
	b.pending[c] = true
	return c
}

// Native request and Snapshot owners have pointer identity. Avoid retaining
// the same owner twice when the original error already exposes that owner.
func appendSteeringOwner(owners []io.Closer, candidate io.Closer) []io.Closer {
	value := reflect.ValueOf(candidate)
	if value.Comparable() {
		for _, owner := range owners {
			if reflect.TypeOf(owner) == value.Type() && owner == candidate {
				return owners
			}
		}
	}
	return append(owners, candidate)
}

func (b *steeringBackend) request(state dsv4.State) (*steeringRequestState, error) {
	s, ok := state.(*steeringRequestState)
	if !ok || s == nil || s.owner != b {
		return nil, fmt.Errorf("steering state belongs to another backend: %w", dsv4.ErrUnimplemented)
	}
	return s, nil
}

func (b *steeringBackend) Eval(state dsv4.State, token, position int) error {
	s, err := b.request(state)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return dsv4.ErrClosed
	}
	err = b.IndependentStates.Eval(s.state, token, position)
	return b.retainEvaluation(err)
}

func (b *steeringBackend) EvalBatch(state dsv4.State, tokens []int, position int) error {
	s, err := b.request(state)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return dsv4.ErrClosed
	}
	if batch, ok := b.IndependentStates.(dsv4.BatchEvaluator); ok {
		return b.retainEvaluation(batch.EvalBatch(s.state, tokens, position))
	}
	for i, token := range tokens {
		if err := b.IndependentStates.Eval(s.state, token, position+i); err != nil {
			return b.retainEvaluation(err)
		}
	}
	return nil
}

func (b *steeringBackend) retainEvaluation(err error) error {
	owners := backendErrorOwners(err)
	if len(owners) == 0 {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.retainLocked(err, owners)
}

func (b *steeringBackend) Logits(state dsv4.State) []float32 {
	s, err := b.request(state)
	if err != nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	return b.IndependentStates.Logits(s.state)
}

func (b *steeringBackend) Close() error {
	b.closeMu.Lock()
	defer b.closeMu.Unlock()
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil
	}
	b.closing = true
	var requests []io.Closer
	for request := range b.pending {
		requests = append(requests, request)
	}
	for state := range b.states {
		requests = append(requests, state)
	}
	b.mu.Unlock()
	group := backendCleanupGroup{requests}
	if err := group.Close(); err != nil {
		return fmt.Errorf("close steering requests: %w", err)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	// An in-flight evaluation can publish a cleanup owner after the snapshot.
	// It must be released on a subsequent Close before the foundation closes.
	if len(b.pending) != 0 || len(b.states) != 0 {
		return errors.New("steering request cleanup remains pending")
	}
	if err := b.IndependentStates.Close(); err != nil {
		return err
	}
	b.closed = true
	b.vectors = nil
	return nil
}

// steeringRequestCleanup is an ownership boundary shared by the returned
// error and the backend ledger. Successful retry removes that ledger entry.
type steeringRequestCleanup struct {
	mu      sync.Mutex
	cause   error
	owners  backendCleanupGroup
	backend *steeringBackend
}

func (c *steeringRequestCleanup) Error() string { return c.cause.Error() }
func (c *steeringRequestCleanup) Unwrap() error { return c.cause }
func (c *steeringRequestCleanup) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.owners.Close(); err != nil {
		return err
	}
	c.backend.mu.Lock()
	delete(c.backend.pending, c)
	c.backend.mu.Unlock()
	return nil
}

type steeringRequestState struct {
	mu     sync.Mutex
	state  dsv4.State
	owner  *steeringBackend
	closed bool
}

func (s *steeringRequestState) Reset() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return dsv4.ErrClosed
	}
	return s.owner.retainEvaluation(s.state.Reset())
}

func (s *steeringRequestState) Snapshot() (dsv4.Snapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, dsv4.ErrClosed
	}
	snapshot, err := s.state.Snapshot()
	if err != nil && snapshot != nil {
		owners := appendSteeringOwner(backendErrorOwners(err), snapshot)
		s.owner.mu.Lock()
		err = s.owner.retainLocked(err, owners)
		s.owner.mu.Unlock()
		return nil, err
	}
	return snapshot, s.owner.retainEvaluation(err)
}

func (s *steeringRequestState) Restore(snapshot dsv4.Snapshot) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return dsv4.ErrClosed
	}
	return s.owner.retainEvaluation(s.state.Restore(snapshot))
}

func (s *steeringRequestState) Save(w io.Writer) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return dsv4.ErrClosed
	}
	return s.owner.retainEvaluation(s.state.Save(w))
}

func (s *steeringRequestState) Load(r io.Reader) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return dsv4.ErrClosed
	}
	return s.owner.retainEvaluation(s.state.Load(r))
}

func (s *steeringRequestState) steeringFFN() float32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0
	}
	control, ok := s.state.(steeringControl)
	if !ok {
		return 0
	}
	return control.steeringFFN()
}

func (s *steeringRequestState) setSteeringFFN(scale float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return dsv4.ErrClosed
	}
	control, ok := s.state.(steeringControl)
	if !ok {
		return dsv4.ErrUnimplemented
	}
	return s.owner.retainEvaluation(control.setSteeringFFN(scale))
}

func (s *steeringRequestState) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	if s.state == nil {
		return nil
	}
	if err := s.state.Close(); err != nil {
		return err
	}
	s.state = nil
	s.owner.mu.Lock()
	delete(s.owner.states, s)
	s.owner.mu.Unlock()
	return nil
}
