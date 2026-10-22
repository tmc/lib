package main

import (
	"errors"
	"io"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4kvstore"
)

// stateSessionBackend adapts the backend-independent State contract to the
// legacy sessionBackend seam. It owns one request state and never shares that
// mutable state with another session.
type stateSessionBackend struct {
	backend dsv4.Backend
	state   dsv4.State
	closed  bool
}

var _ sessionBackend = (*stateSessionBackend)(nil)
var _ io.Closer = (*stateSessionBackend)(nil)

func newStateSessionBackend(backend dsv4.Backend, context int) (sessionBackend, io.Closer, error) {
	if backend == nil {
		return nil, nil, errors.New("missing state backend")
	}
	state, err := backend.NewState(context)
	if err != nil {
		return nil, nil, err
	}
	if state == nil {
		return nil, nil, errors.New("state backend returned nil state")
	}
	s := &stateSessionBackend{backend: backend, state: state}
	if batch, ok := backend.(dsv4.BatchEvaluator); ok {
		b := &batchStateSessionBackend{stateSessionBackend: s, batch: batch}
		return b, b, nil
	}
	return s, s, nil
}

func (s *stateSessionBackend) Reset() error {
	if s == nil || s.closed || s.state == nil {
		return dsv4.ErrClosed
	}
	return s.state.Reset()
}

func (s *stateSessionBackend) Eval(token, position int) error {
	if s == nil || s.closed || s.backend == nil || s.state == nil {
		return dsv4.ErrClosed
	}
	return s.backend.Eval(s.state, token, position)
}

func (s *stateSessionBackend) Logits() []float32 {
	if s == nil || s.closed || s.backend == nil || s.state == nil {
		return nil
	}
	return s.backend.Logits(s.state)
}

func (s *stateSessionBackend) Save(w io.Writer) error {
	if s == nil || s.closed || s.state == nil {
		return dsv4.ErrClosed
	}
	return s.state.Save(w)
}

func (s *stateSessionBackend) Load(r io.Reader) error {
	if s == nil || s.closed || s.state == nil {
		return dsv4.ErrClosed
	}
	return s.state.Load(r)
}

func (s *stateSessionBackend) LoadUpstream(record dsv4kvstore.UpstreamRecord, contract dsv4kvstore.UpstreamRestoreContract) error {
	if s == nil || s.closed || s.state == nil {
		return dsv4.ErrClosed
	}
	loader, ok := s.state.(upstreamStateLoader)
	if !ok {
		return dsv4.ErrUnimplemented
	}
	return loader.LoadUpstream(record, contract)
}

// Close makes the adapter unusable immediately. Failed releases retain their
// owner for retry; the backend is released only after the state is released.
func (s *stateSessionBackend) Close() error {
	if s == nil {
		return nil
	}
	s.closed = true
	if s.state != nil {
		if err := s.state.Close(); err != nil {
			return err
		}
		s.state = nil
	}
	if s.backend != nil {
		if err := s.backend.Close(); err != nil {
			return err
		}
		s.backend = nil
	}
	return nil
}

type batchStateSessionBackend struct {
	*stateSessionBackend
	batch dsv4.BatchEvaluator
}

func (s *batchStateSessionBackend) EvalBatch(tokens []int, position int) error {
	if s == nil || s.stateSessionBackend == nil || s.closed || s.state == nil {
		return dsv4.ErrClosed
	}
	return s.batch.EvalBatch(s.state, tokens, position)
}
