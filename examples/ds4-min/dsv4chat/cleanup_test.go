package dsv4chat

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func ExampleRun_cleanup() {
	b := &resizeBackend{closeErrors: []error{errors.New("release failed")}}
	cfg := fixture(new(backend))
	cfg.Backend = b
	err := Run(context.Background(), strings.NewReader("/quit"), io.Discard, cfg)
	var cleanup io.Closer
	if errors.As(err, &cleanup) {
		b.states[0].closeErr = nil // The fixture's release failure is now resolved.
		fmt.Println(cleanup.Close())
	}
	// Output: <nil>
}

func TestRunRetainsEveryFailedCandidateCleanup(t *testing.T) {
	configureFailures := []error{errors.New("configure one"), errors.New("configure two")}
	closeFailures := []error{errors.New("close one"), errors.New("close two")}
	b := &resizeBackend{closeErrors: []error{nil, closeFailures[0], closeFailures[1]}}
	cfg := fixture(new(backend))
	cfg.Backend = b
	configuration := 0
	publications := 0
	cfg.ConfigureState = func(dsv4.State) error {
		configuration++
		if configuration > 1 {
			return configureFailures[configuration-2]
		}
		return nil
	}
	cfg.StateChanged = func(dsv4.State) { publications++ }
	cfg.Command = func(string) (Action, error) { return Action{Context: 16, Handled: true}, nil }
	cfg.Errors = io.Discard
	err := Run(context.Background(), strings.NewReader("one\n/ctx 16\n/ctx 16\ntwo"), new(bytes.Buffer), cfg)
	for _, cause := range append(configureFailures, closeFailures...) {
		if !errors.Is(err, cause) {
			t.Fatalf("lost cause %v: %v", cause, err)
		}
	}
	var owner io.Closer
	if !errors.As(err, &owner) {
		t.Fatal("cleanup owner inaccessible through error")
	}
	if publications != 1 || len(b.states[0].values) != 8 || b.states[0].closes != 1 {
		t.Fatal("failed replacement published or destroyed old conversation")
	}
	b.states[1].closeErr = nil
	if err := owner.Close(); !errors.Is(err, closeFailures[1]) {
		t.Fatalf("first retry %v", err)
	}
	if b.states[1].closes != 2 || b.states[2].closes != 2 {
		t.Fatal("retry skipped an owner")
	}
	b.states[2].closeErr = nil
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if b.states[1].closes != 2 || b.states[2].closes != 3 {
		t.Fatal("successful owner retried or failed owner lost")
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
}

type cleanupFixture struct {
	name    string
	failure error
	events  *[]string
}

type genericCleanupError struct {
	*cleanupFixture
	cause error
}

func (e *genericCleanupError) Error() string { return e.cause.Error() }
func (e *genericCleanupError) Unwrap() error { return e.cause }

func TestCleanupOwnerPreservesWrappedGenericOwnershipBoundary(t *testing.T) {
	failure := errors.New("generic release failed")
	var events []string
	generic := &genericCleanupError{&cleanupFixture{"generic", failure, &events}, failure}
	wrapped := fmt.Errorf("configure: %w", generic)
	owners := cleanupOwners(wrapped)
	if len(owners) != 1 || owners[0] != generic {
		t.Fatal("unwrap lost ownership boundary")
	}
	owner := &cleanupOwner{cause: wrapped, owners: owners}
	if err := owner.Close(); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"generic"}) {
		t.Fatal(events)
	}
	generic.failure = nil
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(owner, failure) {
		t.Fatal("successful retry lost original cause")
	}
}

type failedNewStateBackend struct {
	resizeBackend
	failure error
}

type terminalOwnedBackend struct {
	resizeBackend
	failure error
}

func (b *terminalOwnedBackend) Eval(dsv4.State, int, int) error { return b.failure }

func TestRunRetainsTerminalCauseAndSessionOwners(t *testing.T) {
	for _, failedClose := range []bool{false, true} {
		t.Run(fmt.Sprint(failedClose), func(t *testing.T) {
			cause := errors.New("evaluate failed")
			release := errors.New("release failed")
			var events []string
			original := &genericCleanupError{&cleanupFixture{"evaluate", release, &events}, cause}
			b := &terminalOwnedBackend{failure: fmt.Errorf("evaluate: %w", original)}
			if failedClose {
				b.closeErrors = []error{release}
			}
			cfg := fixture(new(backend))
			cfg.Backend = b
			err := Run(context.Background(), strings.NewReader("one"), io.Discard, cfg)
			if !errors.Is(err, cause) || !errors.Is(err, dsv4.ErrDesynchronized) {
				t.Fatalf("lost evaluation cause: %v", err)
			}
			var owner io.Closer
			if !errors.As(err, &owner) {
				t.Fatal("terminal cleanup owner missing")
			}
			b.states[0].closeErr = nil
			if err := owner.Close(); !errors.Is(err, release) {
				t.Fatalf("lost original cleanup owner: %v", err)
			}
			wantCloses := 1
			if failedClose {
				wantCloses++
			}
			if b.states[0].closes != wantCloses {
				t.Fatal("successful session release duplicated or failed release lost")
			}
			original.failure = nil
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(events, []string{"evaluate", "evaluate"}) || b.states[0].closes != wantCloses {
				t.Fatal("original owner lost or successful session retried")
			}
		})
	}
}

func (b *failedNewStateBackend) NewState(int) (dsv4.State, error) { return nil, b.failure }

func TestRunRetainsWrappedBackendNewStateOwners(t *testing.T) {
	for _, joined := range []bool{false, true} {
		t.Run(fmt.Sprint(joined), func(t *testing.T) {
			firstFailure, secondFailure := errors.New("allocation one"), errors.New("allocation two")
			var events []string
			first := &genericCleanupError{&cleanupFixture{"first", firstFailure, &events}, firstFailure}
			second := &genericCleanupError{&cleanupFixture{"second", secondFailure, &events}, secondFailure}
			var failed error = first
			if joined {
				failed = errors.Join(first, second)
			}
			b := &failedNewStateBackend{failure: fmt.Errorf("allocate request: %w", failed)}
			cfg := fixture(new(backend))
			cfg.Backend = b
			cfg.ConfigureState = func(dsv4.State) error { t.Fatal("configured failed allocation"); return nil }
			cfg.StateChanged = func(dsv4.State) { t.Fatal("published failed allocation") }
			err := Run(context.Background(), strings.NewReader("ignored"), io.Discard, cfg)
			if !errors.Is(err, firstFailure) || joined && !errors.Is(err, secondFailure) {
				t.Fatal(err)
			}
			var owner io.Closer
			if !errors.As(err, &owner) {
				t.Fatal("allocation cleanup owner lost through NewSession wrapping")
			}
			first.failure = nil
			second.failure = nil
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			want := []string{"first"}
			if joined {
				want = append(want, "second")
			}
			if !reflect.DeepEqual(events, want) {
				t.Fatalf("releases %v want %v", events, want)
			}
		})
	}
}

func TestConfigureFailureRetainsCauseAndCandidateOwners(t *testing.T) {
	configureFailure, candidateFailure := errors.New("configure owner"), errors.New("candidate close")
	var events []string
	configureOwner := &genericCleanupError{&cleanupFixture{"configure", configureFailure, &events}, configureFailure}
	b := &resizeBackend{closeErrors: []error{candidateFailure}}
	cfg := fixture(new(backend))
	cfg.Backend = b
	cfg.ConfigureState = func(dsv4.State) error { return fmt.Errorf("configure: %w", configureOwner) }
	err := Run(context.Background(), strings.NewReader("ignored"), io.Discard, cfg)
	if !errors.Is(err, configureFailure) || !errors.Is(err, candidateFailure) {
		t.Fatal(err)
	}
	var owner io.Closer
	if !errors.As(err, &owner) {
		t.Fatal("configure+candidate cleanup owner missing")
	}
	configureOwner.failure = nil
	b.states[0].closeErr = nil
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"configure"}) || b.states[0].closes != 2 {
		t.Fatal("lost configure or candidate ownership")
	}
}

func (f *cleanupFixture) Close() error { *f.events = append(*f.events, f.name); return f.failure }

func TestCleanupOwnerRetriesAllInOrder(t *testing.T) {
	firstFailure, secondFailure := errors.New("first"), errors.New("second")
	var events []string
	first := &cleanupFixture{"first", firstFailure, &events}
	second := &cleanupFixture{"second", secondFailure, &events}
	wrapped := errors.Join(retainCleanup(firstFailure, first), retainCleanup(secondFailure, second))
	owner := &cleanupOwner{cause: wrapped, owners: cleanupOwners(wrapped)}
	if err := owner.Close(); !errors.Is(err, firstFailure) || !errors.Is(err, secondFailure) {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []string{"first", "second"}) {
		t.Fatalf("order %v", events)
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
	if !reflect.DeepEqual(events, []string{"second"}) {
		t.Fatal(events)
	}
}
