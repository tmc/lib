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

type blockedPrefillBackend struct {
	started chan string
	gate    chan struct{}
	decoded chan []string
}

type committedDecodeBackend struct {
	entered  chan struct{}
	complete chan struct{}
	mu       sync.Mutex
	releases int
}

func (*committedDecodeBackend) Prefill(context.Context, *State) (Token, error) {
	return Token{ID: 1}, nil
}

func (b *committedDecodeBackend) DecodeBatch(_ context.Context, states []*State) []Result {
	close(b.entered)
	<-b.complete
	results := make([]Result, len(states))
	for i := range results {
		results[i].Token = Token{ID: 2}
	}
	return results
}

func (b *committedDecodeBackend) Release(*State) {
	b.mu.Lock()
	b.releases++
	b.mu.Unlock()
}

func (b *blockedPrefillBackend) Prefill(ctx context.Context, state *State) (Token, error) {
	b.started <- state.ID
	if state.ID == "blocked" {
		select {
		case <-b.gate:
		case <-ctx.Done():
			return Token{}, ctx.Err()
		}
	}
	return Token{ID: 1}, nil
}

func (b *blockedPrefillBackend) DecodeBatch(_ context.Context, states []*State) []Result {
	ids := make([]string, len(states))
	results := make([]Result, len(states))
	for i, state := range states {
		ids[i] = state.ID
		results[i].Token = Token{ID: state.Current + 1}
	}
	b.decoded <- ids
	return results
}

func (*blockedPrefillBackend) Release(*State) {}

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

func TestDecodeDeadlineUsesVirtualArrivalTime(t *testing.T) {
	start := time.Unix(100, 0)
	deadline := keepDecodeDeadline(time.Time{}, start, 10*time.Second)
	for _, arrival := range []time.Duration{time.Second, 3 * time.Second, 9 * time.Second} {
		deadline = keepDecodeDeadline(deadline, start.Add(arrival), 10*time.Second)
	}
	if want := start.Add(10 * time.Second); !deadline.Equal(want) {
		t.Fatalf("deadline = %v, want %v", deadline, want)
	}
}

func TestBlockedPrefillDoesNotStarveActiveDecode(t *testing.T) {
	backend := &blockedPrefillBackend{started: make(chan string, 3), gate: make(chan struct{}), decoded: make(chan []string, 1)}
	s := newTestScheduler(t, backend, Config{MaxActive: 3, MaxQueue: 1, MaxBatch: 1, BatchWait: time.Millisecond, TokenBuffer: 4, PrefillWorkers: 1})
	active, err := s.Submit(context.Background(), Request{ID: "active", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	if id := <-backend.started; id != "active" {
		t.Fatalf("first prefill = %q", id)
	}
	if token := <-active.Tokens; token.ID != 1 {
		t.Fatalf("active prefill token = %d", token.ID)
	}
	blocked, err := s.Submit(context.Background(), Request{ID: "blocked", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	if id := <-backend.started; id != "blocked" {
		t.Fatalf("second prefill = %q", id)
	}
	queued, err := s.Submit(context.Background(), Request{ID: "queued", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	select {
	case ids := <-backend.decoded:
		if !slices.Equal(ids, []string{"active"}) {
			t.Fatalf("decode = %v, want active", ids)
		}
	case <-time.After(time.Second):
		t.Fatal("active decode starved behind unavailable prefill worker")
	}
	close(backend.gate)
	for _, response := range []*Response{active, blocked, queued} {
		if _, err := collect(t, response); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCancellationAfterDecodeCommitSuppressesOutputAndReleases(t *testing.T) {
	backend := &committedDecodeBackend{entered: make(chan struct{}), complete: make(chan struct{})}
	s := newTestScheduler(t, backend, Config{MaxActive: 1, MaxQueue: 1, MaxBatch: 1, BatchWait: time.Millisecond, TokenBuffer: 2})
	response, err := s.Submit(context.Background(), Request{ID: "cancel-after-commit", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	if token := <-response.Tokens; token.ID != 1 {
		t.Fatalf("prefill token = %d", token.ID)
	}
	<-backend.entered
	response.Cancel()
	close(backend.complete)
	if tokens, err := collect(t, response); !errors.Is(err, context.Canceled) || len(tokens) != 0 {
		t.Fatalf("post-cancel tokens=%v err=%v", tokens, err)
	}
	backend.mu.Lock()
	releases := backend.releases
	backend.mu.Unlock()
	if releases != 1 {
		t.Fatalf("releases = %d, want 1", releases)
	}
	metrics := s.Metrics()
	if metrics.Canceled != 1 {
		t.Fatalf("terminal metrics = %+v", metrics)
	}
	replacement, err := s.Submit(context.Background(), Request{ID: "replacement", MaxTokens: 1})
	if err != nil {
		t.Fatalf("replacement admission: %v", err)
	}
	if _, err := collect(t, replacement); err != nil {
		t.Fatal(err)
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
