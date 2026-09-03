package continuousbatch

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"
)

type auditBackend struct {
	mu      sync.Mutex
	started chan string
	steps   []string
	gate    chan struct{}
}

type tokenPromptBackend struct{ got []int32 }

func (b *tokenPromptBackend) Prefill(_ context.Context, state *State) (Token, error) {
	b.got = append([]int32(nil), state.PromptTokenIDs...)
	return Token{ID: 9}, nil
}
func (*tokenPromptBackend) DecodeBatch(_ context.Context, states []*State) []Result {
	return make([]Result, len(states))
}
func (*tokenPromptBackend) Release(*State) {}

func TestSubmitCopiesExplicitPromptTokens(t *testing.T) {
	backend := new(tokenPromptBackend)
	s := newTestScheduler(t, backend, Config{MaxActive: 1, MaxQueue: 1, MaxBatch: 1, TokenBuffer: 2})
	prompt := []int32{1, 2, 3}
	response, err := s.Submit(context.Background(), Request{ID: "tokens", PromptTokenIDs: prompt, MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	prompt[0] = 99
	if _, err := collect(t, response); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(backend.got, []int32{1, 2, 3}) {
		t.Fatalf("prompt = %v", backend.got)
	}
}

func (b *auditBackend) Prefill(ctx context.Context, state *State) (Token, error) {
	if b.started != nil {
		b.started <- state.ID
	}
	if b.gate != nil {
		select {
		case <-b.gate:
		case <-ctx.Done():
			return Token{}, ctx.Err()
		}
	}
	return Token{ID: 1, Text: "1"}, nil
}

func (b *auditBackend) DecodeBatch(_ context.Context, states []*State) []Result {
	b.mu.Lock()
	for _, state := range states {
		b.steps = append(b.steps, state.ID)
	}
	b.mu.Unlock()
	results := make([]Result, len(states))
	for i, state := range states {
		results[i].Token = Token{ID: state.Current + 1, Text: "x"}
	}
	return results
}

func (b *auditBackend) Release(*State) {}

func TestAdmissionBoundCountsActiveAndQueued(t *testing.T) {
	backend := &auditBackend{started: make(chan string, 3), gate: make(chan struct{})}
	s := newTestScheduler(t, backend, Config{MaxActive: 1, MaxQueue: 2, MaxBatch: 1, TokenBuffer: 8})
	responses := make([]*Response, 3)
	for i, id := range []string{"active", "queued-1", "queued-2"} {
		var err error
		responses[i], err = s.Submit(context.Background(), Request{ID: id, MaxTokens: 2})
		if err != nil {
			t.Fatalf("submit %s: %v", id, err)
		}
	}
	<-backend.started
	if _, err := s.Submit(context.Background(), Request{ID: "overflow", MaxTokens: 2}); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("overflow error = %v", err)
	}
	close(backend.gate)
	for _, response := range responses {
		if _, err := collect(t, response); err != nil {
			t.Fatal(err)
		}
	}
	m := s.Metrics()
	if m.Submitted != 4 || m.Rejected != 1 || m.Completed != 3 {
		t.Fatalf("metrics = %+v", m)
	}
}

func TestRoundRobinFairnessWhenBatchIsSmallerThanActiveSet(t *testing.T) {
	backend := &auditBackend{}
	s := newTestScheduler(t, backend, Config{MaxActive: 4, MaxQueue: 1, MaxBatch: 2, BatchWait: time.Millisecond, TokenBuffer: 8, PrefillWorkers: 4})
	responses := make([]*Response, 4)
	for i, id := range []string{"a", "b", "c", "d"} {
		var err error
		responses[i], err = s.Submit(context.Background(), Request{ID: id, MaxTokens: 3})
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, response := range responses {
		if _, err := collect(t, response); err != nil {
			t.Fatal(err)
		}
	}
	backend.mu.Lock()
	steps := append([]string(nil), backend.steps...)
	backend.mu.Unlock()
	first := make(map[string]int)
	for i, id := range steps {
		if _, ok := first[id]; !ok {
			first[id] = i
		}
	}
	if len(first) != 4 {
		t.Fatalf("decode schedule %v omitted a request", steps)
	}
	for id, i := range first {
		if i >= 4 {
			t.Fatalf("request %s first scheduled at %d in %v", id, i, steps)
		}
	}
}

func TestMetricAccountingInvariants(t *testing.T) {
	backend := &fakeBackend{decode: time.Millisecond}
	s := newTestScheduler(t, backend, Config{MaxActive: 3, MaxQueue: 3, MaxBatch: 3, BatchWait: time.Millisecond, TokenBuffer: 8, PrefillWorkers: 3})
	responses := make([]*Response, 3)
	for i := range responses {
		responses[i], _ = s.Submit(context.Background(), Request{ID: string(rune('a' + i)), MaxTokens: 4})
	}
	for _, response := range responses {
		if _, err := collect(t, response); err != nil {
			t.Fatal(err)
		}
	}
	m := s.Summary()
	var calls uint64
	for _, count := range m.BatchSizes {
		calls += count
	}
	if calls != m.DecodeBatches || calls != m.DecodeDispatch {
		t.Fatalf("histogram=%v calls=%d metrics=%+v", m.BatchSizes, calls, m.Metrics)
	}
	if m.Tokens != 12 || m.Completed != 3 || m.Active != 0 || m.Queued != 0 {
		t.Fatalf("terminal metrics = %+v", m.Metrics)
	}
	if m.TTFTP50 == 0 || m.TTFTP95 < m.TTFTP50 || m.TPOTP50 == 0 || m.TPOTP95 < m.TPOTP50 {
		t.Fatalf("latency summary = %+v", m)
	}
}
