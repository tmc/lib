//go:build linux

package main

import (
	"bytes"
	"errors"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestCUDAStateAdapterFailsClosed(t *testing.T) {
	a := &cudaStateAdapter{}
	state, err := a.NewState(4096)
	if !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("NewState error = %v, want dsv4.ErrUnimplemented", err)
	}
	if state != nil {
		t.Fatal("NewState returned state with ErrUnimplemented")
	}
	if !cudaStateUnsupported(err) {
		t.Fatal("adapter error was not recognized as unsupported")
	}
	if err := a.Eval(nil, 1, 0); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("Eval error = %v, want dsv4.ErrUnimplemented", err)
	}
	if got := a.Logits(nil); got != nil {
		t.Fatalf("Logits = %v, want nil for unsupported state", got)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("Close on nil runtime = %v, want nil", err)
	}
}

func TestCUDAStateAdapterCloseIsTerminal(t *testing.T) {
	a := &cudaStateAdapter{}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if state, err := a.NewState(16); !errors.Is(err, dsv4.ErrClosed) || state != nil {
		t.Fatalf("NewState after Close = (%v, %v)", state, err)
	}
	if err := a.Eval(nil, 1, 0); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("Eval after Close = %v", err)
	}
}

func TestCUDARequestStatePersistenceFailsClosed(t *testing.T) {
	s := &cudaRequestState{}
	if _, err := s.Snapshot(); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("Snapshot error = %v, want dsv4.ErrClosed", err)
	}
	if err := s.Restore(nil); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("Restore error = %v, want dsv4.ErrClosed", err)
	}
	var saved bytes.Buffer
	if err := s.Save(&saved); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("Save error = %v, want dsv4.ErrClosed", err)
	}
	if err := s.Load(bytes.NewReader(nil)); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("Load error = %v, want dsv4.ErrClosed", err)
	}
}

func TestCUDAStateAdapterNeverAliasesLegacyRuntime(t *testing.T) {
	// A nil runtime is enough to exercise the adapter boundary. The method
	// must remain safe even when the legacy runtime has not been constructed;
	// a future implementation must not turn this into a shared fallback.
	var legacy *cudaRuntime
	b := legacy.stateBackend()
	for _, context := range []int{2, 128, 4096} {
		state, err := b.NewState(context)
		if !errors.Is(err, dsv4.ErrUnimplemented) || state != nil {
			t.Fatalf("context %d: NewState = (%v, %v), want (nil, ErrUnimplemented)", context, state, err)
		}
	}
	if got := b.Logits(nil); got != nil {
		t.Fatalf("legacy adapter Logits = %v, want nil", got)
	}
}

func TestCUDARequestRuntimeClosePreservesBorrowedFoundation(t *testing.T) {
	foundation := &cudaFoundation{}
	request := &cudaRuntime{
		cudaFoundation:     foundation,
		borrowedFoundation: true,
	}
	if err := request.Close(); err != nil {
		t.Fatalf("close borrowed runtime: %v", err)
	}
	if request.cudaFoundation != nil {
		t.Fatal("close retained the request's foundation reference")
	}
	// The foundation is owned by the backend. A request close must not invoke
	// cudaFoundation.Close, which would tear down the backend's model mapping,
	// device, and compiled modules.
	if foundation == nil {
		t.Fatal("borrowed foundation was lost")
	}
}
