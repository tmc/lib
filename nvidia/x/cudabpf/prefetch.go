package cudabpf

import (
	"sync"
	"time"
)

// A PrefetchRequest is one queued region to prefetch. ID identifies it for
// cancellation; a request already taken by the worker cannot be cancelled,
// which [PrefetchWorker.Cancel] reports rather than pretending otherwise.
type PrefetchRequest struct {
	ID      uint64
	VASpace uint64
	Region  Region
	// Decision is a host-model context. It must remain valid through
	// completion. Never retain a live kernel fault-context pointer here:
	// kernel prefetch requires its own verified asynchronous interface.
	Decision *PrefetchDecision
}

// A PrefetchResult is what became of one request.
type PrefetchResult struct {
	ID        uint64
	Issued    bool
	Cancelled bool
	Err       error
	// Latency is the time from submission to issue. It is [Unmeasured]
	// unless the worker was given a clock.
	Latency time.Duration
}

// A PrefetchWorker issues prefetch regions off the fault path.
//
// Issuance is non-blocking by construction: [PrefetchWorker.Submit] never
// waits, and a full queue is reported to the caller rather than absorbed.
// Anything expensive belongs here and not in a handler, and the time it takes
// is reported separately from fault-handling time because the two overlap and
// must never be summed.
type PrefetchWorker struct {
	kfuncs Kfuncs
	clock  func() time.Time

	queue chan PrefetchRequest

	mu        sync.Mutex
	cancelled map[uint64]bool
	submitted map[uint64]time.Time
	results   []PrefetchResult

	wg      sync.WaitGroup
	closed  bool
	started bool
	depth   int
	pending map[uint64]bool
}

// NewPrefetchWorker returns a worker with a bounded queue. The bound is the
// whole point: an unbounded queue turns a burst of faults into unbounded
// memory, and the fault path has nowhere to report that.
func NewPrefetchWorker(k Kfuncs, depth int, clock func() time.Time) *PrefetchWorker {
	if depth <= 0 {
		depth = 1
	}
	return &PrefetchWorker{
		kfuncs:    k,
		clock:     clock,
		queue:     make(chan PrefetchRequest, depth),
		depth:     depth,
		pending:   make(map[uint64]bool),
		cancelled: make(map[uint64]bool),
		submitted: make(map[uint64]time.Time),
	}
}

// Start begins issuing. A worker that is not started still accepts
// submissions up to its queue depth, so a handler never blocks on startup
// order.
func (w *PrefetchWorker) Start() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.started || w.closed {
		return
	}
	w.started = true
	w.wg.Add(1)
	go func() {
		defer w.wg.Done()
		for req := range w.queue {
			w.issue(req)
		}
	}()
}

// Submit queues a request. It never blocks. It reports false when the queue
// is full, when the worker is closed, or when the region is not a valid
// range; in every one of those cases nothing was queued and the caller knows
// it.
func (w *PrefetchWorker) Submit(req PrefetchRequest) bool {
	if req.Decision == nil || req.ID == 0 {
		return false
	}
	if err := req.Region.Valid(); err != nil {
		return false
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	_, exists := w.pending[req.ID]
	if w.closed || exists {
		return false
	}
	if w.clock != nil {
		w.submitted[req.ID] = w.clock()
	}

	select {
	case w.queue <- req:
		w.pending[req.ID] = true
		return true
	default:
		delete(w.submitted, req.ID)
		return false
	}
}

// Cancel marks a request cancelled. It reports whether the cancellation
// arrived in time: a request the worker has already issued cannot be recalled,
// and saying otherwise would make a cancelled prefetch indistinguishable from
// one that never ran.
func (w *PrefetchWorker) Cancel(id uint64) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.pending[id] {
		return false
	}
	w.cancelled[id] = true
	return true
}

func (w *PrefetchWorker) issue(req PrefetchRequest) {
	w.mu.Lock()
	cancelled := w.cancelled[req.ID]
	delete(w.cancelled, req.ID)
	w.pending[req.ID] = false // claimed; cancellation no longer possible
	start, timed := w.submitted[req.ID]
	delete(w.submitted, req.ID)
	w.mu.Unlock()

	res := PrefetchResult{ID: req.ID, Latency: Unmeasured}
	if timed && w.clock != nil {
		res.Latency = w.clock().Sub(start)
	}
	switch {
	case cancelled:
		res.Cancelled = true
	case w.kfuncs == nil:
		res.Err = ErrNotAttached
	default:
		if err := w.kfuncs.SetPrefetchRegion(req.Decision, req.Region.First, req.Region.Outer); err != nil {
			res.Err = err
		} else {
			res.Issued = true
		}
	}

	w.mu.Lock()
	delete(w.pending, req.ID)
	if len(w.results) == w.depth {
		copy(w.results, w.results[1:])
		w.results = w.results[:w.depth-1]
	}
	w.results = append(w.results, res)
	w.mu.Unlock()
}

// Results returns the most recent completed requests, bounded by queue depth.
func (w *PrefetchWorker) Results() []PrefetchResult {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]PrefetchResult(nil), w.results...)
}

// Close stops the worker and waits for in-flight issuance to finish.
func (w *PrefetchWorker) Close() {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		w.wg.Wait()
		return
	}
	w.closed = true
	if !w.started {
		for len(w.queue) > 0 {
			req := <-w.queue
			w.results = append(w.results, PrefetchResult{ID: req.ID, Cancelled: true, Latency: Unmeasured})
			delete(w.pending, req.ID)
			delete(w.cancelled, req.ID)
			delete(w.submitted, req.ID)
		}
	}
	close(w.queue)
	w.mu.Unlock()
	w.wg.Wait()
}
