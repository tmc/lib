package main

import (
	"context"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4chat"
)

type terminalRequestBackend struct {
	stateSessionBackendTestBackend
	failure error
}

func (b *terminalRequestBackend) Eval(dsv4.State, int, int) error { return b.failure }

func TestChatTerminalOwnersPrecedeFoundation(t *testing.T) {
	var events []string
	cause := errors.New("evaluate failed")
	release := errors.New("evaluate cleanup failed")
	original := &ownedCleanupFixture{&backendCleanupFixture{"evaluate", release, &events}, cause}
	state := &retryOwnerState{events: &events}
	backend := &terminalRequestBackend{failure: original}
	backend.state = state
	cfg := dsv4chat.Config{Backend: backend, Context: 100, Tokens: 1, Stops: dsv4.NewStopSet(9),
		Render: func([]dsv4chat.Message) ([]int, error) { return []int{1}, nil },
		Sample: func([]float32) (int, error) { t.Fatal("sample after failed evaluation"); return 0, nil },
		Decode: func(int) (string, error) { t.Fatal("decode after failed evaluation"); return "", nil },
	}
	err := dsv4chat.Run(context.Background(), strings.NewReader("one"), io.Discard, cfg)
	foundation := &backendCleanupFixture{"foundation", nil, &events}
	err = finishBackendCleanup(err, foundation)
	if !errors.Is(err, cause) || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("lost terminal or session cleanup cause: %v", err)
	}
	var owner io.Closer
	if !errors.As(err, &owner) {
		t.Fatal("terminal cleanup inaccessible")
	}
	if err := owner.Close(); !errors.Is(err, release) {
		t.Fatalf("lost original evaluation owner: %v", err)
	}
	if !reflect.DeepEqual(events, []string{"state", "evaluate", "state"}) {
		t.Fatalf("foundation released before every request owner: %v", events)
	}
	original.failure = nil
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"state", "evaluate", "state", "evaluate", "foundation"}) || state.calls != 2 {
		t.Fatalf("retry dropped an owner or duplicated successful release: %v", events)
	}
	if err := owner.Close(); err != nil || state.calls != 2 || len(events) != 5 {
		t.Fatal("completed cleanup retried a released resource")
	}
}

func TestOneShotRequestCleanupPrecedesFoundation(t *testing.T) {
	for _, phase := range []string{"prefill-config", "generation"} {
		t.Run(phase, func(t *testing.T) {
			var events []string
			state := &retryOwnerState{events: &events}
			model := &retryOwnerBackend{events: &events}
			model.state = state
			backend, closer, err := newStateSessionBackend(model, 100)
			if err != nil {
				t.Fatal(err)
			}
			s, err := newSession(backend, 100, stopSet{9: true})
			if err != nil {
				t.Fatal(err)
			}
			cause := errors.New("generation failed")
			if phase == "prefill-config" {
				cause = configurePrefillChunk(s.backend, 256)
				if !errors.Is(cause, dsv4.ErrUnimplemented) {
					t.Fatal(cause)
				}
			}
			err = finishBackendCleanup(cause, closer)
			if !errors.Is(err, cause) || !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("lost one-shot/cleanup cause: %v", err)
			}
			if !reflect.DeepEqual(events, []string{"state"}) || model.calls != 0 {
				t.Fatalf("premature foundation release %v", events)
			}
			if err := backend.Eval(1, 0); !errors.Is(err, dsv4.ErrClosed) {
				t.Fatalf("computed on terminal request: %v", err)
			}
			var owner io.Closer
			if !errors.As(err, &owner) {
				t.Fatal("one-shot cleanup owner missing")
			}
			if err := owner.Close(); !errors.Is(err, io.ErrShortWrite) {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(events, []string{"state", "state", "backend"}) {
				t.Fatalf("retry order %v", events)
			}
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if state.calls != 2 || model.calls != 2 {
				t.Fatal("retry duplicated released request or lost foundation")
			}
		})
	}
}

type backendCleanupFixture struct {
	name    string
	failure error
	events  *[]string
}

func (f *backendCleanupFixture) Close() error {
	*f.events = append(*f.events, f.name)
	return f.failure
}

type ownedCleanupFixture struct {
	*backendCleanupFixture
	cause error
}

func (f *ownedCleanupFixture) Error() string { return f.cause.Error() }
func (f *ownedCleanupFixture) Unwrap() error { return f.cause }

func TestBackendCleanupPreservesFoundationUntilRequestsReleased(t *testing.T) {
	failure := errors.New("request release failed")
	modelFailure := errors.New("model release failed")
	var events []string
	request := &ownedCleanupFixture{&backendCleanupFixture{"request", failure, &events}, failure}
	model := &backendCleanupFixture{"model", modelFailure, &events}
	err := finishBackendCleanup(request, model)
	if len(events) != 0 || !errors.Is(err, failure) {
		t.Fatalf("premature release %v error %v", events, err)
	}
	var owner io.Closer
	if !errors.As(err, &owner) {
		t.Fatal("cleanup owner missing")
	}
	if err := owner.Close(); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"request"}) {
		t.Fatalf("foundation released before request: %v", events)
	}
	request.failure = nil
	events = nil
	if err := owner.Close(); !errors.Is(err, modelFailure) {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"request", "model"}) {
		t.Fatalf("release order %v", events)
	}
	model.failure = nil
	events = nil
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"model"}) {
		t.Fatal(events)
	}
	events = nil
	if err := owner.Close(); err != nil || len(events) != 0 {
		t.Fatalf("repeated release %v error %v", events, err)
	}
}

func TestBackendCleanupRetainsFailedFoundation(t *testing.T) {
	failure := errors.New("model release failed")
	original := errors.New("generation failed")
	var events []string
	model := &backendCleanupFixture{"model", failure, &events}
	err := finishBackendCleanup(original, model)
	if !errors.Is(err, original) || !errors.Is(err, failure) {
		t.Fatal(err)
	}
	var owner io.Closer
	if !errors.As(err, &owner) {
		t.Fatal("model cleanup owner missing")
	}
	model.failure = nil
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"model", "model"}) {
		t.Fatal(events)
	}
}

func TestBackendCleanupPreservesEveryJoinedRequest(t *testing.T) {
	firstFailure, secondFailure := errors.New("first request"), errors.New("second request")
	var events []string
	first := &ownedCleanupFixture{&backendCleanupFixture{"first", firstFailure, &events}, firstFailure}
	second := &ownedCleanupFixture{&backendCleanupFixture{"second", secondFailure, &events}, secondFailure}
	model := &backendCleanupFixture{"model", nil, &events}
	err := finishBackendCleanup(errors.Join(first, second), model)
	var owner io.Closer
	if !errors.As(err, &owner) || !errors.Is(err, firstFailure) || !errors.Is(err, secondFailure) {
		t.Fatal("lost joined request owner/cause")
	}
	if err := owner.Close(); !errors.Is(err, firstFailure) || !errors.Is(err, secondFailure) {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"first", "second"}) {
		t.Fatalf("premature foundation release %v", events)
	}
	first.failure = nil
	events = nil
	if err := owner.Close(); !errors.Is(err, secondFailure) {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"first", "second"}) {
		t.Fatal(events)
	}
	second.failure = nil
	events = nil
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"second", "model"}) {
		t.Fatalf("retry/foundation order %v", events)
	}
}
