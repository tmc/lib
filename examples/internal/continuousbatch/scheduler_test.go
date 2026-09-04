package continuousbatch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

type fakeBackend struct {
	mu        sync.Mutex
	batches   []int
	prefill   time.Duration
	decode    time.Duration
	failState string
}

type fakeBatchBackend struct {
	fakeBackend
	prefillBatches []int
}

type pressurePrefillBackend struct {
	started chan struct{}
	unblock chan struct{}
	once    sync.Once
}

func (b *pressurePrefillBackend) Prefill(ctx context.Context, state *State) (Token, error) {
	b.once.Do(func() { close(b.started) })
	select {
	case <-b.unblock:
		state.KV = new(int)
		return Token{ID: 1, Text: "1"}, nil
	case <-ctx.Done():
		return Token{}, ctx.Err()
	}
}

func (*pressurePrefillBackend) DecodeBatch(context.Context, []*State) []Result { return nil }
func (*pressurePrefillBackend) Release(*State)                                 {}

func (b *fakeBatchBackend) PrefillBatch(requests []PrefillRequest) []Result {
	b.mu.Lock()
	b.prefillBatches = append(b.prefillBatches, len(requests))
	b.mu.Unlock()
	results := make([]Result, len(requests))
	for i, request := range requests {
		if err := request.Context.Err(); err != nil {
			results[i].Err = err
			continue
		}
		if request.State.ID == b.failState {
			results[i].Err = errors.New("injected prefill failure")
			continue
		}
		request.State.KV = new(int)
		results[i].Token = Token{ID: 1, Text: "1"}
	}
	return results
}

func (b *fakeBackend) Prefill(ctx context.Context, state *State) (Token, error) {
	select {
	case <-time.After(b.prefill):
	case <-ctx.Done():
		return Token{}, ctx.Err()
	}
	state.KV = new(int)
	return Token{ID: 1, Text: "1"}, nil
}

func (b *fakeBackend) DecodeBatch(_ context.Context, states []*State) []Result {
	time.Sleep(b.decode)
	b.mu.Lock()
	b.batches = append(b.batches, len(states))
	b.mu.Unlock()
	results := make([]Result, len(states))
	for i, state := range states {
		if state.ID == b.failState {
			results[i].Err = errors.New("injected")
			continue
		}
		next := state.Current + 1
		results[i].Token = Token{ID: next, Text: string(rune('0' + next))}
	}
	return results
}

func (b *fakeBackend) Release(*State) {}

func newTestScheduler(t *testing.T, backend Backend, cfg Config) *Scheduler {
	t.Helper()
	s, err := New(backend, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func collect(t *testing.T, response *Response) ([]Token, error) {
	t.Helper()
	var tokens []Token
	for token := range response.Tokens {
		tokens = append(tokens, token)
	}
	return tokens, <-response.Done
}

func TestStaggeredUnequalRequestsShareDecodeBatch(t *testing.T) {
	backend := &fakeBackend{prefill: 2 * time.Millisecond, decode: time.Millisecond}
	s := newTestScheduler(t, backend, Config{MaxActive: 4, MaxQueue: 8, MaxBatch: 4, BatchWait: 4 * time.Millisecond, TokenBuffer: 16})
	var responses []*Response
	for i, limit := range []int{5, 3, 4} {
		response, err := s.Submit(context.Background(), Request{ID: string(rune('a' + i)), MaxTokens: limit})
		if err != nil {
			t.Fatal(err)
		}
		responses = append(responses, response)
		time.Sleep(time.Millisecond)
	}
	for i, response := range responses {
		tokens, err := collect(t, response)
		if err != nil {
			t.Fatal(err)
		}
		want := []int{5, 3, 4}[i]
		if len(tokens) != want {
			t.Fatalf("request %d got %d tokens, want %d", i, len(tokens), want)
		}
	}
	backend.mu.Lock()
	defer backend.mu.Unlock()
	shared := false
	for _, n := range backend.batches {
		if n > 1 {
			shared = true
		}
	}
	if !shared {
		t.Fatalf("decode batch sizes %v contain no shared operation", backend.batches)
	}
}

func TestCancellationAndFailureIsolation(t *testing.T) {
	backend := &fakeBackend{decode: time.Millisecond, failState: "bad"}
	s := newTestScheduler(t, backend, Config{MaxActive: 4, MaxQueue: 8, MaxBatch: 4, BatchWait: time.Millisecond, TokenBuffer: 16})
	good, _ := s.Submit(context.Background(), Request{ID: "good", MaxTokens: 5})
	bad, _ := s.Submit(context.Background(), Request{ID: "bad", MaxTokens: 5})
	canceled, _ := s.Submit(context.Background(), Request{ID: "cancel", MaxTokens: 20})
	canceled.Cancel()
	if tokens, err := collect(t, good); err != nil || len(tokens) != 5 {
		t.Fatalf("good: tokens=%d err=%v", len(tokens), err)
	}
	if _, err := collect(t, bad); err == nil {
		t.Fatal("failed request succeeded")
	}
	if _, err := collect(t, canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel error = %v", err)
	}
}

func TestBatchedPrefillExcludesCanceledAndIsolatesFailure(t *testing.T) {
	backend := &fakeBatchBackend{fakeBackend: fakeBackend{failState: "bad"}}
	s := newTestScheduler(t, backend, Config{MaxActive: 4, MaxQueue: 4, MaxBatch: 4, BatchWait: 5 * time.Millisecond, TokenBuffer: 8, PrefillWorkers: 4})
	good, err := s.Submit(context.Background(), Request{ID: "good", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	bad, err := s.Submit(context.Background(), Request{ID: "bad", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	canceled, err := s.Submit(context.Background(), Request{ID: "canceled", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	canceled.Cancel()
	if tokens, err := collect(t, good); err != nil || len(tokens) != 2 {
		t.Fatalf("good: tokens=%d err=%v", len(tokens), err)
	}
	if _, err := collect(t, bad); err == nil {
		t.Fatal("injected prefill failure succeeded")
	}
	if _, err := collect(t, canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
	}
	backend.mu.Lock()
	defer backend.mu.Unlock()
	if len(backend.prefillBatches) != 1 || backend.prefillBatches[0] != 2 {
		t.Fatalf("prefill batches = %v, want [2]", backend.prefillBatches)
	}
}

func TestSlowConsumerAndQueuePressure(t *testing.T) {
	backend := &fakeBackend{decode: time.Millisecond}
	s := newTestScheduler(t, backend, Config{MaxActive: 1, MaxQueue: 1, MaxBatch: 1, BatchWait: time.Millisecond, TokenBuffer: 1})
	slow, err := s.Submit(context.Background(), Request{ID: "slow", MaxTokens: 20})
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(10 * time.Millisecond)
	for range slow.Tokens {
	}
	if err := <-slow.Done; !errors.Is(err, ErrSlowConsumer) {
		t.Fatalf("slow consumer error = %v", err)
	}

	blocker, _ := s.Submit(context.Background(), Request{ID: "blocker", MaxTokens: 20})
	queued, err := s.Submit(context.Background(), Request{ID: "queued", MaxTokens: 20})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Submit(context.Background(), Request{ID: "overflow", MaxTokens: 20}); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("overflow error = %v", err)
	}
	blocker.Cancel()
	queued.Cancel()
}

func TestCanceledSubmitDoesNotConsumeQueueCapacity(t *testing.T) {
	backend := &pressurePrefillBackend{started: make(chan struct{}), unblock: make(chan struct{})}
	s := newTestScheduler(t, backend, Config{MaxActive: 1, MaxQueue: 1, MaxBatch: 1})
	first, err := s.Submit(context.Background(), Request{ID: "running", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	<-backend.started
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.Submit(canceled, Request{ID: "already-canceled", MaxTokens: 1}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled submit error = %v, want context.Canceled", err)
	}
	queued, err := s.Submit(context.Background(), Request{ID: "queued", MaxTokens: 1})
	if err != nil {
		t.Fatalf("healthy submit behind canceled request: %v", err)
	}
	close(backend.unblock)
	if _, err := collect(t, first); err != nil {
		t.Fatal(err)
	}
	if _, err := collect(t, queued); err != nil {
		t.Fatal(err)
	}
}

func TestHTTPRequestsReachSharedDecodeBatch(t *testing.T) {
	backend := &fakeBackend{prefill: 2 * time.Millisecond, decode: time.Millisecond}
	s := newTestScheduler(t, backend, Config{MaxActive: 8, MaxQueue: 16, MaxBatch: 8, BatchWait: 5 * time.Millisecond, TokenBuffer: 16, PrefillWorkers: 4})
	server := httptest.NewServer(&Handler{Scheduler: s, Model: "fake"})
	defer server.Close()
	start := make(chan struct{})
	errCh := make(chan error, 4)
	for i := 0; i < 4; i++ {
		go func() {
			<-start
			body, _ := json.Marshal(map[string]any{"prompt": "hello", "max_tokens": 6, "stream": true})
			resp, err := http.Post(server.URL, "application/json", bytes.NewReader(body))
			if err == nil {
				_, err = io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
			}
			errCh <- err
		}()
	}
	close(start)
	for i := 0; i < 4; i++ {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
	backend.mu.Lock()
	defer backend.mu.Unlock()
	shared := false
	for _, n := range backend.batches {
		if n >= 4 {
			shared = true
		}
	}
	if !shared {
		t.Fatalf("HTTP requests did not share a four-way decode batch: %v", backend.batches)
	}
}

func TestShutdownFinishesRequests(t *testing.T) {
	backend := &fakeBackend{prefill: 50 * time.Millisecond}
	s, err := New(backend, Config{MaxActive: 1, MaxQueue: 4, MaxBatch: 1})
	if err != nil {
		t.Fatal(err)
	}
	response, _ := s.Submit(context.Background(), Request{ID: "shutdown", MaxTokens: 2})
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	for range response.Tokens {
	}
	if err := <-response.Done; !errors.Is(err, ErrClosed) {
		t.Fatalf("shutdown error = %v", err)
	}
}
