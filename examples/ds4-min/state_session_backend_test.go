package main

import (
	"bytes"
	"errors"
	"io"
	"reflect"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type retryOwnerState struct {
	stateSessionBackendTestState
	events *[]string
	calls  int
}

func (s *retryOwnerState) Close() error {
	*s.events = append(*s.events, "state")
	s.calls++
	if s.calls == 1 {
		return io.ErrUnexpectedEOF
	}
	return nil
}

type retryOwnerBackend struct {
	stateSessionBackendTestBackend
	events *[]string
	calls  int
}

func (b *retryOwnerBackend) Close() error {
	*b.events = append(*b.events, "backend")
	b.calls++
	if b.calls == 1 {
		return io.ErrShortWrite
	}
	return nil
}

func TestStateSessionBackendCloseRetry(t *testing.T) {
	var events []string
	state := &retryOwnerState{events: &events}
	backend := &retryOwnerBackend{events: &events}
	backend.state = state
	s := &stateSessionBackend{backend: backend, state: state}
	if err := s.Close(); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("first Close = %v", err)
	}
	if s.state != state || s.backend != backend || backend.calls != 0 {
		t.Fatal("failed state release lost ownership or closed backend")
	}
	checks := []func() error{
		s.Reset,
		func() error { return s.Eval(1, 0) },
		func() error { return s.Save(io.Discard) },
		func() error { return s.Load(bytes.NewReader(nil)) },
		func() error { return (&batchStateSessionBackend{stateSessionBackend: s}).EvalBatch([]int{1}, 0) },
	}
	for i, call := range checks {
		if err := call(); !errors.Is(err, dsv4.ErrClosed) {
			t.Errorf("method %d = %v", i, err)
		}
	}
	if s.Logits() != nil {
		t.Fatal("closed owner exposes logits")
	}
	if err := s.Close(); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("second Close = %v", err)
	}
	if s.state != nil || s.backend != backend {
		t.Fatal("failed backend release lost ownership")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if want := []string{"state", "state", "backend", "backend"}; !reflect.DeepEqual(events, want) {
		t.Fatalf("release order = %v, want %v", events, want)
	}
}

type stateSessionBackendTestBackend struct {
	state dsv4.State
	close bool
}

func (b *stateSessionBackendTestBackend) NewState(int) (dsv4.State, error) {
	return b.state, nil
}
func (b *stateSessionBackendTestBackend) Eval(dsv4.State, int, int) error { return nil }
func (b *stateSessionBackendTestBackend) Logits(dsv4.State) []float32     { return []float32{1} }
func (b *stateSessionBackendTestBackend) Close() error                    { b.close = true; return nil }

type stateSessionBackendTestState struct{ closed bool }

func (s *stateSessionBackendTestState) Reset() error {
	if s.closed {
		return dsv4.ErrClosed
	}
	return nil
}
func (s *stateSessionBackendTestState) Snapshot() (dsv4.Snapshot, error) {
	return nil, dsv4.ErrUnimplemented
}
func (s *stateSessionBackendTestState) Restore(dsv4.Snapshot) error { return dsv4.ErrUnimplemented }
func (s *stateSessionBackendTestState) Save(_ io.Writer) error {
	return dsv4.ErrUnimplemented
}
func (s *stateSessionBackendTestState) Load(_ io.Reader) error {
	return dsv4.ErrUnimplemented
}
func (s *stateSessionBackendTestState) Close() error { s.closed = true; return nil }

func TestStateSessionBackendLifecycle(t *testing.T) {
	state := &stateSessionBackendTestState{}
	backend := &stateSessionBackendTestBackend{state: state}
	s, closer, err := newStateSessionBackend(backend, 32)
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Logits(); len(got) != 1 || got[0] != 1 {
		t.Fatalf("Logits = %v", got)
	}
	if err := closer.Close(); err != nil {
		t.Fatal(err)
	}
	if !state.closed || !backend.close {
		t.Fatalf("Close state=%v backend=%v", state.closed, backend.close)
	}
	if err := s.Reset(); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("Reset after Close = %v", err)
	}
}
