//go:build linux

package main

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/examples/internal/continuousbatch"
)

type fakeServingBackend struct {
	mu      sync.Mutex
	closed  bool
	prefill chan prefillObservation
}

type prefillObservation struct {
	requestIDs []string
	errors     []error
}

func (b *fakeServingBackend) Prefill(ctx context.Context, state *continuousbatch.State) (continuousbatch.Token, error) {
	result := b.PrefillBatch([]continuousbatch.PrefillRequest{{Context: ctx, State: state}})
	return result[0].Token, result[0].Err
}

func (b *fakeServingBackend) PrefillBatch(requests []continuousbatch.PrefillRequest) []continuousbatch.Result {
	observation := prefillObservation{requestIDs: make([]string, len(requests)), errors: make([]error, len(requests))}
	for i, request := range requests {
		observation.requestIDs[i] = request.State.ID
		observation.errors[i] = request.Context.Err()
	}
	b.prefill <- observation
	results := make([]continuousbatch.Result, len(requests))
	for i, request := range requests {
		if err := request.Context.Err(); err != nil {
			results[i].Err = err
			continue
		}
		results[i].Token.ID = 1
	}
	return results
}

func (*fakeServingBackend) DecodeBatch(_ context.Context, states []*continuousbatch.State) []continuousbatch.Result {
	return make([]continuousbatch.Result, len(states))
}

func (*fakeServingBackend) Release(*continuousbatch.State) {}

func (b *fakeServingBackend) Close() {
	b.mu.Lock()
	b.closed = true
	b.mu.Unlock()
}

func TestLinuxServingBackendUnavailable(t *testing.T) {
	backend, err := newPlatformServingBackend(nil, Config{}, 1)
	if !errors.Is(err, errServingBackendUnavailable) || backend != nil {
		t.Fatalf("backend = %T, %v; want unavailable", backend, err)
	}
}

func TestServingBackendBoundaryPreservesRequestContexts(t *testing.T) {
	backend := &fakeServingBackend{prefill: make(chan prefillObservation, 1)}
	scheduler, err := newServingScheduler(backend, continuousbatch.Config{
		MaxActive: 2, MaxQueue: 2, MaxBatch: 2, BatchWait: time.Millisecond,
		TokenBuffer: 1, PrefillWorkers: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer scheduler.Close()

	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := scheduler.Submit(canceled, continuousbatch.Request{ID: "canceled", MaxTokens: 1}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled submit = %v, want context canceled", err)
	}
	response, err := scheduler.Submit(t.Context(), continuousbatch.Request{ID: "live", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	observation := <-backend.prefill
	if len(observation.requestIDs) != 1 || observation.requestIDs[0] != "live" || observation.errors[0] != nil {
		t.Fatalf("prefill observation = %+v", observation)
	}
	if token := <-response.Tokens; token.ID != 1 {
		t.Fatalf("token = %d, want 1", token.ID)
	}
	if err := <-response.Done; err != nil {
		t.Fatal(err)
	}

	done := make(chan struct{})
	go func() {
		backend.Close()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("backend close blocked on scheduler or device work")
	}
}
