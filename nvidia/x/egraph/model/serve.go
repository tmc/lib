package model

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
)

// An Engine runs batched decode steps for a [Gateway].
//
// The gateway calls Step from a single goroutine per registered engine.
// Step must return exactly one [Token] per sequence in batch, in order.
// A sequence with no Generated tokens joined the batch in this step and
// needs its prompt prefilled. The batch slice and the sequences belong
// to the gateway; Step must not modify or retain them.
type Engine interface {
	Step(ctx context.Context, batch []*Sequence) ([]Token, error)
}

// A Sequence is one admitted generate request as an [Engine] sees it.
type Sequence struct {
	ID        uint64
	Prompt    []int
	Generated []int // tokens produced so far, excluding EOS
	MaxTokens int
}

// A Token is one decode step's output for one sequence.
type Token struct {
	ID   int
	Text string
	EOS  bool // the sequence is finished; ID and Text are not streamed
}

// BatchConfig sets continuous-batching admission for one engine.
// All fields must be positive.
type BatchConfig struct {
	// MaxBatch is the most sequences passed to one Step.
	MaxBatch int
	// MaxTokens bounds the tokens in flight: the sum of
	// len(Prompt)+MaxTokens over admitted sequences. A request that
	// alone exceeds it is rejected.
	MaxTokens int
	// MaxQueue is the most requests waiting for admission.
	// Requests beyond it are rejected with 503.
	MaxQueue int
}

// GenerateRequest is the JSON body of POST /v1/generate.
type GenerateRequest struct {
	Model        string `json:"model,omitempty"` // may be empty when one engine is registered
	PromptTokens []int  `json:"prompt_tokens"`
	MaxTokens    int    `json:"max_tokens"`
	Stream       bool   `json:"stream,omitempty"`
}

// GenerateResponse is the JSON body of a non-streamed generate response.
type GenerateResponse struct {
	Tokens       []int  `json:"tokens"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"` // "stop" or "length"
}

// TokenEvent is the data of one server-sent event in a streamed generate
// response. The last event sets FinishReason or Error and carries no
// token; its Index is the number of tokens streamed.
type TokenEvent struct {
	Index        int    `json:"index"`
	Token        int    `json:"token"`
	Text         string `json:"text,omitempty"`
	FinishReason string `json:"finish_reason,omitempty"`
	Error        string `json:"error,omitempty"`
}

var errGatewayClosed = errors.New("gateway closed")

// A Gateway routes generate requests to registered engines and batches
// them continuously: each engine's scheduler admits waiting requests
// between decode steps, in arrival order, while the batch has room under
// [BatchConfig], and retires sequences as they finish or their clients
// disconnect.
//
// Gateway serves
//
//	POST /v1/generate  GenerateRequest -> GenerateResponse, or
//	                   text/event-stream of TokenEvent when Stream is set
//	GET  /v1/models    registered engine names and limits
//
// The zero Gateway is not usable; call [NewGateway].
type Gateway struct {
	mux    *http.ServeMux
	nextID atomic.Uint64

	mu       sync.Mutex
	batchers map[string]*batcher
	closed   bool
}

// NewGateway returns a Gateway with no engines.
func NewGateway() *Gateway {
	g := &Gateway{mux: http.NewServeMux(), batchers: make(map[string]*batcher)}
	g.mux.HandleFunc("POST /v1/generate", g.serveGenerate)
	g.mux.HandleFunc("GET /v1/models", g.serveModels)
	return g
}

// Register starts a scheduler for e under name.
func (g *Gateway) Register(name string, e Engine, cfg BatchConfig) error {
	if name == "" {
		return fmt.Errorf("register: empty model name")
	}
	if cfg.MaxBatch <= 0 || cfg.MaxTokens <= 0 || cfg.MaxQueue <= 0 {
		return fmt.Errorf("register %s: batch config %+v: all limits must be positive", name, cfg)
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return fmt.Errorf("register %s: %w", name, errGatewayClosed)
	}
	if _, ok := g.batchers[name]; ok {
		return fmt.Errorf("register %s: already registered", name)
	}
	b := newBatcher(e, cfg)
	g.batchers[name] = b
	go b.run()
	return nil
}

// Close stops every scheduler. Requests in flight end with an error.
func (g *Gateway) Close() error {
	g.mu.Lock()
	g.closed = true
	bs := make([]*batcher, 0, len(g.batchers))
	for _, b := range g.batchers {
		bs = append(bs, b)
	}
	g.mu.Unlock()
	for _, b := range bs {
		b.stop()
	}
	return nil
}

func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	g.mux.ServeHTTP(w, r)
}

func (g *Gateway) lookup(name string) (*batcher, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if name == "" && len(g.batchers) == 1 {
		for _, b := range g.batchers {
			return b, true
		}
	}
	b, ok := g.batchers[name]
	return b, ok
}

func (g *Gateway) serveModels(w http.ResponseWriter, r *http.Request) {
	type model struct {
		Name string `json:"name"`
		BatchConfig
	}
	var models []model
	g.mu.Lock()
	for name, b := range g.batchers {
		models = append(models, model{name, b.cfg})
	}
	g.mu.Unlock()
	slices.SortFunc(models, func(a, b model) int { return strings.Compare(a.Name, b.Name) })
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(struct {
		Models []model `json:"models"`
	}{models})
}

func (g *Gateway) serveGenerate(w http.ResponseWriter, r *http.Request) {
	var req GenerateRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	b, ok := g.lookup(req.Model)
	if !ok {
		http.Error(w, fmt.Sprintf("unknown model %q", req.Model), http.StatusNotFound)
		return
	}
	if len(req.PromptTokens) == 0 || req.MaxTokens <= 0 {
		http.Error(w, "prompt_tokens and max_tokens must be non-empty and positive", http.StatusBadRequest)
		return
	}
	if cost := len(req.PromptTokens) + req.MaxTokens; cost > b.cfg.MaxTokens {
		http.Error(w, fmt.Sprintf("request needs %d tokens, model allows %d", cost, b.cfg.MaxTokens), http.StatusBadRequest)
		return
	}
	q := &request{
		ctx: r.Context(),
		seq: Sequence{ID: g.nextID.Add(1), Prompt: req.PromptTokens, MaxTokens: req.MaxTokens},
		// Room for every token plus the finish or error event,
		// so the scheduler never blocks on a slow client.
		out: make(chan event, req.MaxTokens+1),
	}
	if err := b.submit(q); err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}

	if !req.Stream {
		var resp GenerateResponse
		var text strings.Builder
		for {
			ev, err := q.next(b)
			if err != nil {
				if r.Context().Err() == nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
				}
				return
			}
			if ev.finish != "" {
				resp.Text = text.String()
				resp.FinishReason = ev.finish
				if resp.Tokens == nil {
					resp.Tokens = []int{}
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
				return
			}
			resp.Tokens = append(resp.Tokens, ev.tok.ID)
			text.WriteString(ev.tok.Text)
		}
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	rc := http.NewResponseController(w)
	send := func(te TokenEvent) bool {
		data, _ := json.Marshal(te)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			return false
		}
		rc.Flush()
		return true
	}
	for i := 0; ; i++ {
		ev, err := q.next(b)
		switch {
		case err != nil:
			if r.Context().Err() == nil {
				send(TokenEvent{Index: i, Error: err.Error()})
			}
			return
		case ev.finish != "":
			send(TokenEvent{Index: i, FinishReason: ev.finish})
			return
		}
		if !send(TokenEvent{Index: i, Token: ev.tok.ID, Text: ev.tok.Text}) {
			return
		}
	}
}

// A request is a generate request between its handler and a scheduler.
type request struct {
	ctx context.Context
	seq Sequence
	out chan event
}

// An event is one scheduler output for a request:
// a token, a finish reason, or an error.
type event struct {
	tok    Token
	finish string
	err    error
}

func (q *request) cost() int { return len(q.seq.Prompt) + q.seq.MaxTokens }

// next returns the next token or finish event for q.
func (q *request) next(b *batcher) (event, error) {
	select {
	case ev := <-q.out:
		return ev, ev.err
	case <-q.ctx.Done():
		return event{}, q.ctx.Err()
	case <-b.done:
		// Events sent before the scheduler stopped take precedence.
		select {
		case ev := <-q.out:
			return ev, ev.err
		default:
			return event{}, errGatewayClosed
		}
	}
}

// A batcher is the continuous-batching scheduler for one engine.
type batcher struct {
	engine Engine
	cfg    BatchConfig
	queue  chan *request
	ctx    context.Context
	cancel context.CancelFunc
	quit   chan struct{}
	done   chan struct{}
	once   sync.Once

	waiting atomic.Int32 // requests in queue, for tests
}

func newBatcher(e Engine, cfg BatchConfig) *batcher {
	ctx, cancel := context.WithCancel(context.Background())
	return &batcher{
		engine: e,
		cfg:    cfg,
		queue:  make(chan *request, cfg.MaxQueue),
		ctx:    ctx,
		cancel: cancel,
		quit:   make(chan struct{}),
		done:   make(chan struct{}),
	}
}

func (b *batcher) submit(q *request) error {
	select {
	case <-b.done:
		return errGatewayClosed
	default:
	}
	select {
	case b.queue <- q:
		b.waiting.Add(1)
		return nil
	default:
		return fmt.Errorf("queue full (%d waiting)", b.cfg.MaxQueue)
	}
}

func (b *batcher) stop() {
	b.once.Do(func() {
		close(b.quit)
		b.cancel()
	})
	<-b.done
}

// receive takes the next queued request, blocking only if block is set.
// It returns nil when nothing is queued or the batcher is stopping.
func (b *batcher) receive(block bool) *request {
	var q *request
	if block {
		select {
		case q = <-b.queue:
		case <-b.quit:
		}
	} else {
		select {
		case q = <-b.queue:
		default:
		}
	}
	if q != nil {
		b.waiting.Add(-1)
	}
	return q
}

func (b *batcher) run() {
	defer close(b.done)
	var (
		active   []*request
		head     *request // next request in arrival order, not yet admitted
		inFlight int      // sum of cost over active
	)
	defer func() {
		for _, q := range active {
			q.out <- event{err: errGatewayClosed}
		}
	}()
	for {
		active = slices.DeleteFunc(active, func(q *request) bool {
			if q.ctx.Err() != nil {
				inFlight -= q.cost()
				return true
			}
			return false
		})

		// Admit in arrival order. A head request that does not fit
		// waits for running sequences to finish rather than being
		// overtaken by smaller ones.
		for len(active) < b.cfg.MaxBatch {
			if head == nil {
				head = b.receive(len(active) == 0)
				if head == nil {
					break
				}
			}
			if head.ctx.Err() != nil {
				head = nil
				continue
			}
			if inFlight+head.cost() > b.cfg.MaxTokens {
				break
			}
			active = append(active, head)
			inFlight += head.cost()
			head = nil
		}
		select {
		case <-b.quit:
			return
		default:
		}
		if len(active) == 0 {
			continue
		}

		batch := make([]*Sequence, len(active))
		for i, q := range active {
			batch[i] = &q.seq
		}
		toks, err := b.engine.Step(b.ctx, batch)
		if err == nil && len(toks) != len(batch) {
			err = fmt.Errorf("engine returned %d tokens for %d sequences", len(toks), len(batch))
		}
		if err != nil {
			select {
			case <-b.quit:
				return
			default:
			}
			for _, q := range active {
				q.out <- event{err: fmt.Errorf("step: %w", err)}
			}
			active, inFlight = active[:0], 0
			continue
		}

		keep := active[:0]
		for i, q := range active {
			t := toks[i]
			finish := "stop"
			if !t.EOS {
				q.seq.Generated = append(q.seq.Generated, t.ID)
				q.out <- event{tok: t}
				finish = ""
				if len(q.seq.Generated) >= q.seq.MaxTokens {
					finish = "length"
				}
			}
			if finish != "" {
				q.out <- event{finish: finish}
				inFlight -= q.cost()
				continue
			}
			keep = append(keep, q)
		}
		clear(active[len(keep):])
		active = keep
	}
}
