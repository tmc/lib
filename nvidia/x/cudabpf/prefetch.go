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
	// Decision is the driver context the region is written into. The
	// worker passes it back to the kfunc unchanged.
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

	wg     sync.WaitGroup
	closed bool
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
		cancelled: make(map[uint64]bool),
		submitted: make(map[uint64]time.Time),
	}
}

// Start begins issuing. A worker that is not started still accepts
// submissions up to its queue depth, so a handler never blocks on startup
// order.
func (w *PrefetchWorker) Start() {
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
	if err := req.Region.Valid(); err != nil {
		return false
	}
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return false
	}
	if w.clock != nil {
		w.submitted[req.ID] = w.clock()
	}
	w.mu.Unlock()

	select {
	case w.queue <- req:
		return true
	default:
		w.mu.Lock()
		delete(w.submitted, req.ID)
		w.mu.Unlock()
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
	for _, r := range w.results {
		if r.ID == id {
			return false
		}
	}
	w.cancelled[id] = true
	return true
}

func (w *PrefetchWorker) issue(req PrefetchRequest) {
	w.mu.Lock()
	cancelled := w.cancelled[req.ID]
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
	w.results = append(w.results, res)
	w.mu.Unlock()
}

// Results returns what became of every request the worker has finished.
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
		return
	}
	w.closed = true
	w.mu.Unlock()
	close(w.queue)
	w.wg.Wait()
}
