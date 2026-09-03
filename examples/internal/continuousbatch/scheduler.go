// Package continuousbatch schedules concurrent generation requests into decode batches.
package continuousbatch

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"
)

var (
	ErrClosed       = errors.New("scheduler closed")
	ErrQueueFull    = errors.New("scheduler queue full")
	ErrSlowConsumer = errors.New("token consumer too slow")
)

type Token struct {
	ID   int32
	Text string
}

// State is the per-request state owned by the scheduler and passed to Backend.
// Backend may store a device-specific KV cache in KV and sampler state in Sampler.
type State struct {
	ID        string
	Prompt    string
	Position  int
	Current   int32
	Generated int
	Limit     int
	KV        any
	Sampler   any
}

// Backend performs model work. DecodeBatch must isolate per-state failures by
// returning one Result for every input state.
type Backend interface {
	Prefill(context.Context, *State) (Token, error)
	DecodeBatch(context.Context, []*State) []Result
	Release(*State)
}

type Result struct {
	Token Token
	Done  bool
	Err   error
}

type Request struct {
	ID        string
	Prompt    string
	MaxTokens int
	Sampler   any
}

type Response struct {
	Tokens <-chan Token
	Done   <-chan error
	cancel context.CancelFunc
}

func (r *Response) Cancel() { r.cancel() }

type Config struct {
	MaxActive      int
	MaxQueue       int
	MaxBatch       int
	BatchWait      time.Duration
	TokenBuffer    int
	PrefillWorkers int
}

type Metrics struct {
	Submitted      uint64
	Rejected       uint64
	Canceled       uint64
	Failed         uint64
	Completed      uint64
	Prefills       uint64
	DecodeBatches  uint64
	DecodeDispatch uint64
	Tokens         uint64
	BatchSizes     map[int]uint64
	QueueTime      time.Duration
	TTFT           []time.Duration
	TPOT           []time.Duration
	Active         int
	Queued         int
}

type Summary struct {
	Metrics
	TTFTP50 time.Duration
	TTFTP95 time.Duration
	TPOTP50 time.Duration
	TPOTP95 time.Duration
}

type sequence struct {
	state    State
	ctx      context.Context
	cancel   context.CancelFunc
	tokens   chan Token
	done     chan error
	queuedAt time.Time
	firstAt  time.Time
	lastAt   time.Time
	finish   sync.Once
}

type Scheduler struct {
	backend    Backend
	cfg        Config
	submit     chan *sequence
	prefill    chan *sequence
	prefilled  chan prefillResult
	wake       chan struct{}
	stop       chan struct{}
	done       chan struct{}
	workers    sync.WaitGroup
	admission  chan struct{}
	workCtx    context.Context
	workCancel context.CancelFunc

	mu      sync.Mutex
	closed  bool
	live    map[*sequence]struct{}
	metrics Metrics
}

type prefillResult struct {
	seq    *sequence
	active bool
}

func New(backend Backend, cfg Config) (*Scheduler, error) {
	if backend == nil {
		return nil, errors.New("nil backend")
	}
	if cfg.MaxActive <= 0 {
		cfg.MaxActive = 32
	}
	if cfg.MaxQueue <= 0 {
		cfg.MaxQueue = 128
	}
	if cfg.MaxBatch <= 0 || cfg.MaxBatch > cfg.MaxActive {
		cfg.MaxBatch = cfg.MaxActive
	}
	if cfg.BatchWait <= 0 {
		cfg.BatchWait = time.Millisecond
	}
	if cfg.TokenBuffer <= 0 {
		cfg.TokenBuffer = 8
	}
	if cfg.PrefillWorkers <= 0 {
		cfg.PrefillWorkers = 1
	}
	workCtx, workCancel := context.WithCancel(context.Background())
	s := &Scheduler{
		backend: backend, cfg: cfg,
		submit: make(chan *sequence, cfg.MaxActive+cfg.MaxQueue), prefill: make(chan *sequence),
		prefilled: make(chan prefillResult, cfg.MaxActive), wake: make(chan struct{}, 1),
		stop: make(chan struct{}), done: make(chan struct{}),
		admission: make(chan struct{}, cfg.MaxActive+cfg.MaxQueue), workCtx: workCtx, workCancel: workCancel,
		live:    make(map[*sequence]struct{}),
		metrics: Metrics{BatchSizes: make(map[int]uint64)},
	}
	for i := 0; i < cfg.PrefillWorkers; i++ {
		s.workers.Add(1)
		go func() { defer s.workers.Done(); s.prefillLoop() }()
	}
	go s.run()
	return s, nil
}

func (s *Scheduler) Submit(ctx context.Context, req Request) (*Response, error) {
	if req.ID == "" {
		return nil, errors.New("empty request id")
	}
	if req.MaxTokens <= 0 {
		return nil, errors.New("max tokens must be positive")
	}
	ctx, cancel := context.WithCancel(ctx)
	seq := &sequence{state: State{ID: req.ID, Prompt: req.Prompt, Limit: req.MaxTokens, Sampler: req.Sampler}, ctx: ctx, cancel: cancel, tokens: make(chan Token, s.cfg.TokenBuffer), done: make(chan error, 1), queuedAt: time.Now()}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		cancel()
		return nil, ErrClosed
	}
	s.metrics.Submitted++
	select {
	case s.admission <- struct{}{}:
		s.live[seq] = struct{}{}
		s.mu.Unlock()
		s.submit <- seq
		s.signal()
		return &Response{Tokens: seq.tokens, Done: seq.done, cancel: cancel}, nil
	default:
		s.metrics.Rejected++
		s.mu.Unlock()
		cancel()
		return nil, ErrQueueFull
	}
}

func (s *Scheduler) run() {
	defer func() {
		s.abortAll(ErrClosed)
		close(s.done)
	}()
	active := make([]*sequence, 0, s.cfg.MaxActive)
	queued := make([]*sequence, 0, s.cfg.MaxQueue)
	inflight := 0
	for {
		for len(active)+inflight < s.cfg.MaxActive && len(queued) > 0 {
			seq := queued[0]
			queued = queued[1:]
			select {
			case s.prefill <- seq:
				inflight++
			case <-s.stop:
				return
			}
		}
		s.setCounts(len(active), len(queued)+len(s.submit))
		if len(active) > 0 {
			timer := time.NewTimer(s.cfg.BatchWait)
			select {
			case seq := <-s.submit:
				queued = append(queued, seq)
				timer.Stop()
				continue
			case result := <-s.prefilled:
				inflight--
				if result.active {
					active = append(active, result.seq)
				}
				timer.Stop()
				continue
			case <-s.wake:
				timer.Stop()
				continue
			case <-s.stop:
				timer.Stop()
				return
			case <-timer.C:
			}
			active = s.decode(active)
			continue
		}
		select {
		case seq := <-s.submit:
			queued = append(queued, seq)
		case result := <-s.prefilled:
			inflight--
			if result.active {
				active = append(active, result.seq)
			}
		case <-s.wake:
		case <-s.stop:
			return
		}
	}
}

func (s *Scheduler) prefillLoop() {
	for {
		select {
		case seq := <-s.prefill:
			if err := seq.ctx.Err(); err != nil {
				s.finish(seq, err)
				s.prefilled <- prefillResult{seq: seq}
				continue
			}
			queueTime := time.Since(seq.queuedAt)
			tok, err := s.backend.Prefill(seq.ctx, &seq.state)
			s.mu.Lock()
			s.metrics.Prefills++
			s.metrics.QueueTime += queueTime
			s.mu.Unlock()
			if err != nil {
				s.finish(seq, fmt.Errorf("prefill: %w", err))
				s.prefilled <- prefillResult{seq: seq}
				continue
			}
			seq.state.Current = tok.ID
			seq.state.Generated = 1
			now := time.Now()
			seq.firstAt, seq.lastAt = now, now
			s.recordTTFT(now.Sub(seq.queuedAt))
			if !s.emit(seq, tok) {
				s.prefilled <- prefillResult{seq: seq}
				continue
			}
			if seq.state.Generated >= seq.state.Limit {
				s.finish(seq, nil)
				s.prefilled <- prefillResult{seq: seq}
				continue
			}
			s.prefilled <- prefillResult{seq: seq, active: true}
			s.signal()
		case <-s.stop:
			return
		}
	}
}

func (s *Scheduler) decode(active []*sequence) []*sequence {
	batch := make([]*sequence, 0, s.cfg.MaxBatch)
	rest := active[:0]
	for _, seq := range active {
		if err := seq.ctx.Err(); err != nil {
			s.finish(seq, err)
			continue
		}
		if len(batch) < s.cfg.MaxBatch {
			batch = append(batch, seq)
		} else {
			rest = append(rest, seq)
		}
	}
	if len(batch) == 0 {
		return rest
	}
	states := make([]*State, len(batch))
	for i := range batch {
		states[i] = &batch[i].state
	}
	results := s.backend.DecodeBatch(s.workCtx, states)
	s.mu.Lock()
	s.metrics.DecodeBatches++
	s.metrics.DecodeDispatch++
	s.metrics.BatchSizes[len(batch)]++
	s.mu.Unlock()
	if len(results) != len(batch) {
		err := fmt.Errorf("decode batch returned %d results for %d states", len(results), len(batch))
		for _, seq := range batch {
			s.finish(seq, err)
		}
		return rest
	}
	for i, result := range results {
		seq := batch[i]
		if result.Err != nil {
			s.finish(seq, fmt.Errorf("decode: %w", result.Err))
			continue
		}
		now := time.Now()
		s.recordTPOT(now.Sub(seq.lastAt))
		seq.lastAt = now
		seq.state.Current = result.Token.ID
		seq.state.Position++
		seq.state.Generated++
		if !s.emit(seq, result.Token) {
			continue
		}
		if result.Done || seq.state.Generated >= seq.state.Limit {
			s.finish(seq, nil)
			continue
		}
		rest = append(rest, seq)
	}
	return rest
}

func (s *Scheduler) emit(seq *sequence, tok Token) bool {
	select {
	case seq.tokens <- tok:
		s.mu.Lock()
		s.metrics.Tokens++
		s.mu.Unlock()
		return true
	case <-seq.ctx.Done():
		s.finish(seq, seq.ctx.Err())
		return false
	default:
		s.finish(seq, ErrSlowConsumer)
		return false
	}
}

func (s *Scheduler) finish(seq *sequence, err error) {
	seq.finish.Do(func() {
		seq.cancel()
		s.backend.Release(&seq.state)
		close(seq.tokens)
		seq.done <- err
		close(seq.done)
		s.mu.Lock()
		delete(s.live, seq)
		<-s.admission
		switch {
		case errors.Is(err, context.Canceled):
			s.metrics.Canceled++
		case err != nil:
			s.metrics.Failed++
		default:
			s.metrics.Completed++
		}
		s.mu.Unlock()
	})
}

func (s *Scheduler) abortAll(err error) {
	s.mu.Lock()
	live := make([]*sequence, 0, len(s.live))
	for seq := range s.live {
		live = append(live, seq)
	}
	s.mu.Unlock()
	for _, seq := range live {
		s.finish(seq, err)
	}
}

func (s *Scheduler) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		<-s.done
		s.workers.Wait()
		return nil
	}
	s.closed = true
	s.workCancel()
	close(s.stop)
	s.mu.Unlock()
	<-s.done
	s.workers.Wait()
	return nil
}

func (s *Scheduler) Metrics() Metrics {
	s.mu.Lock()
	defer s.mu.Unlock()
	m := s.metrics
	m.BatchSizes = make(map[int]uint64, len(s.metrics.BatchSizes))
	for k, v := range s.metrics.BatchSizes {
		m.BatchSizes[k] = v
	}
	m.TTFT = append([]time.Duration(nil), s.metrics.TTFT...)
	m.TPOT = append([]time.Duration(nil), s.metrics.TPOT...)
	return m
}

func (s *Scheduler) Summary() Summary {
	m := s.Metrics()
	return Summary{Metrics: m, TTFTP50: percentile(m.TTFT, 0.50), TTFTP95: percentile(m.TTFT, 0.95), TPOTP50: percentile(m.TPOT, 0.50), TPOTP95: percentile(m.TPOT, 0.95)}
}

func percentile(samples []time.Duration, q float64) time.Duration {
	if len(samples) == 0 {
		return 0
	}
	samples = slices.Clone(samples)
	slices.Sort(samples)
	i := int(float64(len(samples)-1)*q + 0.5)
	return samples[i]
}

func (s *Scheduler) signal() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}
func (s *Scheduler) setCounts(active, queued int) {
	s.mu.Lock()
	s.metrics.Active, s.metrics.Queued = active, queued
	s.mu.Unlock()
}
func (s *Scheduler) recordTTFT(d time.Duration) {
	s.mu.Lock()
	s.metrics.TTFT = append(s.metrics.TTFT, d)
	s.mu.Unlock()
}
func (s *Scheduler) recordTPOT(d time.Duration) {
	s.mu.Lock()
	s.metrics.TPOT = append(s.metrics.TPOT, d)
	s.mu.Unlock()
}
