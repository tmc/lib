package main

import (
	"bytes"
	"errors"
	"io"
	"reflect"
	"testing"
)

func TestEvalBackendConstructorRetainsPartialOwner(t *testing.T) {
	var events []string
	cause := errors.New("constructor failed")
	release := errors.New("child release failed")
	child := &ownedCleanupFixture{&backendCleanupFixture{"child", release, &events}, cause}
	parent := &backendCleanupFixture{"parent", nil, &events}
	err := runEvalBackend(config{}, func(config) (sessionBackend, io.Closer, error) {
		return nil, parent, child
	}, func(sessionBackend) error { t.Fatal("body after constructor failure"); return nil })
	if !errors.Is(err, cause) || len(events) != 0 {
		t.Fatalf("lost partial owner or released parent prematurely: %v %v", err, events)
	}
	var owner io.Closer
	if !errors.As(err, &owner) {
		t.Fatal("partial cleanup owner missing")
	}
	if err := owner.Close(); !errors.Is(err, release) || !reflect.DeepEqual(events, []string{"child"}) {
		t.Fatalf("child-before-parent cleanup lost: %v %v", err, events)
	}
	child.failure = nil
	if err := owner.Close(); err != nil || !reflect.DeepEqual(events, []string{"child", "child", "parent"}) {
		t.Fatalf("partial cleanup retry lost: %v %v", err, events)
	}
}

func TestEvalBackendTraceCannotQualifyBeforeCleanup(t *testing.T) {
	var events []string
	state := &retryOwnerState{events: &events}
	model := &retryOwnerBackend{events: &events}
	model.state = state
	backend, closer, err := newStateSessionBackend(model, 100)
	if err != nil {
		t.Fatal(err)
	}
	var trace bytes.Buffer
	err = runEvalBackend(config{}, func(config) (sessionBackend, io.Closer, error) {
		return backend, closer, nil
	}, func(sessionBackend) error {
		_, err := io.WriteString(&trace, "complete diagnostic trace\n")
		return err
	})
	if trace.Len() == 0 || !errors.Is(err, io.ErrUnexpectedEOF) || model.calls != 0 {
		t.Fatalf("trace qualified or parent released despite failed child: %v %v", err, events)
	}
	var owner io.Closer
	if !errors.As(err, &owner) {
		t.Fatal("failed successful-body cleanup owner missing")
	}
	if err := owner.Close(); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if state.calls != 2 || model.calls != 2 || !reflect.DeepEqual(events, []string{"state", "state", "backend", "backend"}) {
		t.Fatalf("retry ordering or released owner duplicated: %v", events)
	}
	if err := owner.Close(); err != nil || state.calls != 2 || model.calls != 2 {
		t.Fatal("completed cleanup retried")
	}
}
