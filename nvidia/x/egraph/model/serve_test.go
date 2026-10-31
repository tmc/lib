package model

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeEngine emits token Prompt[0]*100+len(Generated) for each sequence
// and records every batch it is given.
type fakeEngine struct {
	gate     chan struct{} // if non-nil, each Step waits for a value or close
	eosAfter int           // if positive, emit EOS once this many tokens are generated; -1: at once
	err      error         // if non-nil, Step fails
	short    bool          // if set, Step returns one token too few

	mu      sync.Mutex
	entered []seqView // batch of the step most recently entered
	batches [][]seqView
}

type seqView struct {
	id        uint64
	generated int
	cost      int
}

func (e *fakeEngine) Step(ctx context.Context, batch []*Sequence) ([]Token, error) {
	views := make([]seqView, len(batch))
	for i, s := range batch {
		views[i] = seqView{s.ID, len(s.Generated), len(s.Prompt) + s.MaxTokens}
	}
	e.mu.Lock()
	e.entered = views
	e.mu.Unlock()
	if e.gate != nil {
		select {
		case <-e.gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	e.mu.Lock()
	e.batches = append(e.batches, views)
	e.mu.Unlock()
	if e.err != nil {
		return nil, e.err
	}
	toks := make([]Token, len(batch))
	for i, s := range batch {
		id := s.Prompt[0]*100 + len(s.Generated)
		toks[i] = Token{ID: id, Text: fmt.Sprintf("t%d ", id), EOS: e.eosAfter != 0 && len(s.Generated) == max(e.eosAfter, 0)}
	}
	if e.short {
		toks = toks[1:]
	}
	return toks, nil
}

func (e *fakeEngine) enteredWith(id uint64) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.ContainsFunc(e.entered, func(s seqView) bool { return s.id == id })
}

func (e *fakeEngine) recorded() [][]seqView {
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.batches)
}

func newTestGateway(t *testing.T, e Engine, cfg BatchConfig) *Gateway {
	t.Helper()
	g := NewGateway()
	if err := g.Register("m", e, cfg); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { g.Close() })
	return g
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

// generate runs one request through g's handler and returns the recorder.
func generate(ctx context.Context, g *Gateway, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequestWithContext(ctx, "POST", "/v1/generate", strings.NewReader(body))
	rec := httptest.NewRecorder()
	g.ServeHTTP(rec, req)
	return rec
}

// asyncGenerate runs generate in a goroutine.
func asyncGenerate(ctx context.Context, g *Gateway, body string) <-chan *httptest.ResponseRecorder {
	c := make(chan *httptest.ResponseRecorder, 1)
	go func() { c <- generate(ctx, g, body) }()
	return c
}

func parseEvents(t *testing.T, body string) []TokenEvent {
	t.Helper()
	var evs []TokenEvent
	sc := bufio.NewScanner(strings.NewReader(body))
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			t.Fatalf("bad event line %q", line)
		}
		var ev TokenEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			t.Fatalf("event %q: %v", data, err)
		}
		evs = append(evs, ev)
	}
	return evs
}

var testBatch = BatchConfig{MaxBatch: 4, MaxTokens: 64, MaxQueue: 8}

func TestGatewayGenerate(t *testing.T) {
	tests := []struct {
		name     string
		method   string
		path     string
		body     string
		eosAfter int
		code     int
		want     string // JSON response or substring of the error body
	}{
		{"length", "POST", "/v1/generate", `{"prompt_tokens":[3,1],"max_tokens":3}`, 0, 200,
			`{"tokens":[300,301,302],"text":"t300 t301 t302 ","finish_reason":"length"}`},
		{"named model", "POST", "/v1/generate", `{"model":"m","prompt_tokens":[1],"max_tokens":1}`, 0, 200,
			`{"tokens":[100],"text":"t100 ","finish_reason":"length"}`},
		{"stop", "POST", "/v1/generate", `{"prompt_tokens":[2],"max_tokens":9}`, 2, 200,
			`{"tokens":[200,201],"text":"t200 t201 ","finish_reason":"stop"}`},
		{"stop at once", "POST", "/v1/generate", `{"prompt_tokens":[2],"max_tokens":9}`, -1, 200,
			`{"tokens":[],"text":"","finish_reason":"stop"}`},
		{"unknown model", "POST", "/v1/generate", `{"model":"x","prompt_tokens":[1],"max_tokens":1}`, 0, 404, `unknown model "x"`},
		{"bad json", "POST", "/v1/generate", `{"prompt_tokens":`, 0, 400, "bad request"},
		{"unknown field", "POST", "/v1/generate", `{"prompt":"hi","max_tokens":1}`, 0, 400, `unknown field "prompt"`},
		{"empty prompt", "POST", "/v1/generate", `{"prompt_tokens":[],"max_tokens":1}`, 0, 400, "must be non-empty"},
		{"zero max", "POST", "/v1/generate", `{"prompt_tokens":[1],"max_tokens":0}`, 0, 400, "must be non-empty"},
		{"over budget", "POST", "/v1/generate", `{"prompt_tokens":[1,2],"max_tokens":63}`, 0, 400, "request needs 65 tokens, model allows 64"},
		{"wrong method", "GET", "/v1/generate", ``, 0, 405, ""},
		{"wrong path", "POST", "/v1/complete", `{}`, 0, 404, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := newTestGateway(t, &fakeEngine{eosAfter: tt.eosAfter}, testBatch)
			rec := httptest.NewRecorder()
			g.ServeHTTP(rec, httptest.NewRequest(tt.method, tt.path, strings.NewReader(tt.body)))
			if rec.Code != tt.code {
				t.Fatalf("code = %d, want %d; body %q", rec.Code, tt.code, rec.Body)
			}
			got := strings.TrimSpace(rec.Body.String())
			if tt.code == 200 && got != tt.want || tt.code != 200 && !strings.Contains(got, tt.want) {
				t.Fatalf("body = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestGatewayStream(t *testing.T) {
	g := newTestGateway(t, &fakeEngine{eosAfter: 3}, testBatch)
	srv := httptest.NewServer(g)
	defer srv.Close()
	resp, err := http.Post(srv.URL+"/v1/generate", "application/json",
		strings.NewReader(`{"prompt_tokens":[7],"max_tokens":10,"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type = %q", ct)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	got := parseEvents(t, string(body))
	want := []TokenEvent{
		{Index: 0, Token: 700, Text: "t700 "},
		{Index: 1, Token: 701, Text: "t701 "},
		{Index: 2, Token: 702, Text: "t702 "},
		{Index: 3, FinishReason: "stop"},
	}
	if !slices.Equal(got, want) {
		t.Fatalf("events:\n got %+v\nwant %+v", got, want)
	}
}

// TestGatewayContinuousBatching checks that a request arriving while
// another is decoding joins the running batch instead of waiting for it.
func TestGatewayContinuousBatching(t *testing.T) {
	e := &fakeEngine{gate: make(chan struct{})}
	g := newTestGateway(t, e, testBatch)
	ctx := context.Background()

	a := asyncGenerate(ctx, g, `{"prompt_tokens":[1],"max_tokens":6}`)
	e.gate <- struct{}{} // A's first step
	b := asyncGenerate(ctx, g, `{"prompt_tokens":[2],"max_tokens":2}`)
	waitFor(t, "B queued or admitted", func() bool {
		return g.batchers["m"].waiting.Load() == 1 || e.enteredWith(2)
	})
	close(e.gate)

	for _, c := range []<-chan *httptest.ResponseRecorder{a, b} {
		if rec := <-c; rec.Code != 200 || !strings.Contains(rec.Body.String(), `"finish_reason":"length"`) {
			t.Fatalf("response %d %s", rec.Code, rec.Body)
		}
	}
	joined := false
	for _, batch := range e.recorded() {
		if len(batch) == 2 && batch[0].id == 1 && batch[0].generated > 0 && batch[1].generated == 0 {
			joined = true
		}
	}
	if !joined {
		t.Fatalf("B never joined A's batch mid-decode: %v", e.recorded())
	}
	if got := len(e.recorded()); got >= 6+2 {
		t.Fatalf("%d steps for 6+2 tokens: requests ran serially", got)
	}
}

// TestGatewayAdmissionLimits queues n requests behind a blocked step and
// checks the largest batch the scheduler forms once it is released.
func TestGatewayAdmissionLimits(t *testing.T) {
	tests := []struct {
		name      string
		cfg       BatchConfig
		n         int
		body      string
		wantBatch int
	}{
		{"max batch", BatchConfig{MaxBatch: 3, MaxTokens: 1000, MaxQueue: 16}, 6, `{"prompt_tokens":[1,1],"max_tokens":8}`, 3},
		{"tokens in flight", BatchConfig{MaxBatch: 8, MaxTokens: 25, MaxQueue: 16}, 6, `{"prompt_tokens":[1,1],"max_tokens":8}`, 2},
		{"both loose", BatchConfig{MaxBatch: 8, MaxTokens: 1000, MaxQueue: 16}, 5, `{"prompt_tokens":[1,1],"max_tokens":8}`, 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &fakeEngine{gate: make(chan struct{})}
			g := newTestGateway(t, e, tt.cfg)
			// The first request blocks in its first step; the rest queue.
			cs := []<-chan *httptest.ResponseRecorder{asyncGenerate(context.Background(), g, tt.body)}
			waitFor(t, "first step", func() bool { return e.enteredWith(1) })
			for range tt.n - 1 {
				cs = append(cs, asyncGenerate(context.Background(), g, tt.body))
			}
			waitFor(t, "requests queued", func() bool { return int(g.batchers["m"].waiting.Load()) == tt.n-1 })
			close(e.gate)
			for _, c := range cs {
				if rec := <-c; rec.Code != 200 {
					t.Fatalf("response %d %s", rec.Code, rec.Body)
				}
			}
			maxBatch := 0
			for _, batch := range e.recorded() {
				cost := 0
				for _, s := range batch {
					cost += s.cost
				}
				if cost > tt.cfg.MaxTokens {
					t.Fatalf("batch %v has %d tokens in flight, limit %d", batch, cost, tt.cfg.MaxTokens)
				}
				maxBatch = max(maxBatch, len(batch))
			}
			if maxBatch != tt.wantBatch {
				t.Fatalf("largest batch %d, want %d", maxBatch, tt.wantBatch)
			}
		})
	}
}

func TestGatewayQueueFull(t *testing.T) {
	e := &fakeEngine{gate: make(chan struct{})}
	g := newTestGateway(t, e, BatchConfig{MaxBatch: 1, MaxTokens: 64, MaxQueue: 1})
	body := `{"prompt_tokens":[1],"max_tokens":2}`
	a := asyncGenerate(context.Background(), g, body)
	e.gate <- struct{}{} // A's first step, so it has left the queue
	b := asyncGenerate(context.Background(), g, body)
	waitFor(t, "B queued", func() bool { return g.batchers["m"].waiting.Load() == 1 })
	rec := generate(context.Background(), g, body)
	if rec.Code != http.StatusServiceUnavailable || !strings.Contains(rec.Body.String(), "queue full") {
		t.Fatalf("third request: %d %s", rec.Code, rec.Body)
	}
	close(e.gate)
	<-a
	<-b
}

// TestGatewayCancel checks that a disconnected client's sequence is
// retired, freeing its batch slot.
func TestGatewayCancel(t *testing.T) {
	e := &fakeEngine{gate: make(chan struct{})}
	g := newTestGateway(t, e, BatchConfig{MaxBatch: 1, MaxTokens: 2000, MaxQueue: 4})
	ctx, cancel := context.WithCancel(context.Background())
	a := asyncGenerate(ctx, g, `{"prompt_tokens":[1],"max_tokens":1000,"stream":true}`)
	e.gate <- struct{}{}
	b := asyncGenerate(context.Background(), g, `{"prompt_tokens":[2],"max_tokens":2}`)
	waitFor(t, "B queued", func() bool { return g.batchers["m"].waiting.Load() == 1 })
	cancel()
	<-a
	close(e.gate)
	if rec := <-b; rec.Code != 200 {
		t.Fatalf("B: %d %s", rec.Code, rec.Body)
	}
	stepsWithA := 0
	for _, batch := range e.recorded() {
		for _, s := range batch {
			if s.id == 1 {
				stepsWithA++
			}
		}
	}
	if stepsWithA > 3 {
		t.Fatalf("canceled sequence ran %d steps", stepsWithA)
	}
}

func TestGatewayEngineError(t *testing.T) {
	tests := []struct {
		name   string
		engine *fakeEngine
		stream bool
		want   string
	}{
		{"error", &fakeEngine{err: errors.New("out of memory")}, false, "step: out of memory"},
		{"error stream", &fakeEngine{err: errors.New("out of memory")}, true, `"error":"step: out of memory"`},
		{"short", &fakeEngine{short: true}, false, "engine returned 0 tokens for 1 sequences"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := newTestGateway(t, tt.engine, testBatch)
			rec := generate(context.Background(), g, fmt.Sprintf(`{"prompt_tokens":[1],"max_tokens":4,"stream":%v}`, tt.stream))
			if !tt.stream && rec.Code != 500 {
				t.Fatalf("code %d", rec.Code)
			}
			if !strings.Contains(rec.Body.String(), tt.want) {
				t.Fatalf("body %q, want %q", rec.Body, tt.want)
			}
			// The scheduler keeps serving after a failed step.
			tt.engine.err, tt.engine.short = nil, false
			if rec := generate(context.Background(), g, `{"prompt_tokens":[1],"max_tokens":1}`); rec.Code != 200 {
				t.Fatalf("after error: %d %s", rec.Code, rec.Body)
			}
		})
	}
}

func TestGatewayClose(t *testing.T) {
	e := &fakeEngine{gate: make(chan struct{})}
	g := NewGateway()
	if err := g.Register("m", e, testBatch); err != nil {
		t.Fatal(err)
	}
	a := asyncGenerate(context.Background(), g, `{"prompt_tokens":[1],"max_tokens":4}`)
	e.gate <- struct{}{}
	g.Close()
	if rec := <-a; rec.Code != 500 || !strings.Contains(rec.Body.String(), "gateway closed") {
		t.Fatalf("in flight: %d %s", rec.Code, rec.Body)
	}
	if rec := generate(context.Background(), g, `{"prompt_tokens":[1],"max_tokens":4}`); rec.Code != 503 {
		t.Fatalf("after close: %d %s", rec.Code, rec.Body)
	}
	if err := g.Register("n", e, testBatch); err == nil {
		t.Fatal("Register after Close succeeded")
	}
}

func TestGatewayRegisterRejects(t *testing.T) {
	tests := []struct {
		name  string
		model string
		cfg   BatchConfig
		want  string
	}{
		{"empty name", "", testBatch, "empty model name"},
		{"zero batch", "a", BatchConfig{MaxTokens: 1, MaxQueue: 1}, "all limits must be positive"},
		{"zero tokens", "a", BatchConfig{MaxBatch: 1, MaxQueue: 1}, "all limits must be positive"},
		{"zero queue", "a", BatchConfig{MaxBatch: 1, MaxTokens: 1}, "all limits must be positive"},
		{"duplicate", "m", testBatch, "register m: already registered"},
	}
	g := newTestGateway(t, &fakeEngine{}, testBatch)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := g.Register(tt.model, &fakeEngine{}, tt.cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestGatewayModels(t *testing.T) {
	g := newTestGateway(t, &fakeEngine{}, testBatch)
	if err := g.Register("a", &fakeEngine{}, BatchConfig{MaxBatch: 1, MaxTokens: 2, MaxQueue: 3}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	g.ServeHTTP(rec, httptest.NewRequest("GET", "/v1/models", nil))
	want := `{"models":[{"name":"a","MaxBatch":1,"MaxTokens":2,"MaxQueue":3},{"name":"m","MaxBatch":4,"MaxTokens":64,"MaxQueue":8}]}`
	if got := strings.TrimSpace(rec.Body.String()); got != want {
		t.Fatalf("got %s\nwant %s", got, want)
	}
	// Two engines: the model must be named.
	if rec := generate(context.Background(), g, `{"prompt_tokens":[1],"max_tokens":1}`); rec.Code != 404 {
		t.Fatalf("unnamed model with two engines: %d", rec.Code)
	}
}

// echoEngine returns each sequence's prompt back, one token per step.
type echoEngine struct{}

func (echoEngine) Step(ctx context.Context, batch []*Sequence) ([]Token, error) {
	toks := make([]Token, len(batch))
	for i, s := range batch {
		n := len(s.Generated)
		if n == len(s.Prompt) {
			toks[i] = Token{EOS: true}
			continue
		}
		toks[i] = Token{ID: s.Prompt[n], Text: fmt.Sprint(s.Prompt[n], " ")}
	}
	return toks, nil
}

func ExampleGateway() {
	g := NewGateway()
	defer g.Close()
	g.Register("echo", echoEngine{}, BatchConfig{MaxBatch: 8, MaxTokens: 4096, MaxQueue: 64})
	srv := httptest.NewServer(g)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/generate", "application/json",
		strings.NewReader(`{"model":"echo","prompt_tokens":[5,6,7],"max_tokens":16}`))
	if err != nil {
		fmt.Println(err)
		return
	}
	defer resp.Body.Close()
	io.Copy(os.Stdout, resp.Body)
	// Output:
	// {"tokens":[5,6,7],"text":"5 6 7 ","finish_reason":"stop"}
}
