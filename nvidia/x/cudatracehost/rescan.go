package cudatracehost

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ThreadRescan records bounded attachment attempts. Times use elapsed monotonic
// nanoseconds from initial discovery, not the BPF event clock. Missed calls and
// threads born and gone between scans are unknown, even after a verified scan.
type ThreadRescan struct {
	Sequence                                  int
	Status, Clock, BlindInterval, MissedCalls string
	StartNanos, EndNanos                      int64
	Threads                                   []ThreadSnapshot
	AttachmentAttempted                       bool
	Reason                                    string
}

type threadRescanner struct {
	mu             sync.Mutex
	initial, known map[uint32]uint64
	history        []ThreadRescan
	limit, probes  int
	failed         bool
	process        func() error
	discover       func() ([]ThreadSnapshot, error)
	generation     func(int) (uint64, error)
	attach         func([]ThreadSnapshot, func() error) error
	now            func() int64
}

func validateRescan(cfg Config, backend string) error {
	if cfg.MaxThreadRescans < 0 || cfg.MaxThreadRescans > 128 || cfg.ThreadRescanInterval < 0 {
		return fmt.Errorf("cudatracehost: invalid thread rescan bounds")
	}
	if cfg.ThreadRescanInterval == 0 {
		return nil
	}
	if backend != "bpf" || cfg.ThreadRescanInterval < 10*time.Millisecond || cfg.ThreadRescanInterval > time.Hour || cfg.MaxThreadRescans == 0 {
		return fmt.Errorf("cudatracehost: thread rescans require bpf, 10ms to 1h interval and 1 to 128 scans")
	}
	return nil
}

func (s *threadRescanner) step(ctx context.Context) (r ThreadRescan, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failed {
		return r, fmt.Errorf("cudatracehost: thread rescanning already stopped")
	}
	r = ThreadRescan{Sequence: len(s.history) + 1, Clock: "elapsed_since_discovery", Status: "refused", BlindInterval: "between_scans_unknown", MissedCalls: "unknown", StartNanos: s.now()}
	defer func() {
		r.EndNanos = s.now()
		if err != nil {
			s.failed = true
			r.Reason = err.Error()
		}
		s.history = append(s.history, r)
	}()
	if len(s.history) >= s.limit {
		return r, fmt.Errorf("cudatracehost: thread rescan budget exhausted")
	}
	verifyProcess := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		return s.process()
	}
	if err = verifyProcess(); err != nil {
		return r, err
	}
	threads, err := s.discover()
	if err != nil {
		return r, err
	}
	if len(threads) == 0 || len(threads) > 256/s.probes {
		return r, fmt.Errorf("cudatracehost: rescan thread bound exceeded")
	}
	seen := make(map[uint32]bool)
	for _, t := range threads {
		if t.TID == 0 || t.Generation == 0 || seen[t.TID] {
			return r, fmt.Errorf("cudatracehost: invalid rescan thread identity")
		}
		seen[t.TID] = true
		if g, ok := s.known[t.TID]; ok {
			if g != t.Generation {
				return r, fmt.Errorf("cudatracehost: recycled thread refused")
			}
		} else {
			r.Threads = append(r.Threads, t)
		}
	}
	if (len(s.known)+len(r.Threads))*s.probes > 256 {
		return r, fmt.Errorf("cudatracehost: lifetime attachment budget exhausted")
	}
	verify := func() error {
		if e := verifyProcess(); e != nil {
			return e
		}
		return verifyThreads(r.Threads, s.generation)
	}
	if err = verify(); err != nil {
		return r, err
	}
	if len(r.Threads) > 0 {
		r.AttachmentAttempted = true
		if err = s.attach(r.Threads, verify); err != nil {
			r.BlindInterval = "attachment_and_rollback_unknown"
			return r, err
		}
	}
	if err = verify(); err != nil {
		r.BlindInterval = "post_attachment_unknown"
		return r, err
	}
	for _, t := range r.Threads {
		s.known[t.TID] = t.Generation
	}
	r.Status = "generation_verified_attachment"
	return r, nil
}

func (s *threadRescanner) snapshot() []ThreadRescan {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := append([]ThreadRescan(nil), s.history...)
	for i := range out {
		out[i].Threads = append([]ThreadSnapshot(nil), out[i].Threads...)
	}
	return out
}

func (c *Collector) startRescans(ctx context.Context, interval time.Duration) {
	ctx, c.rescanCancel = context.WithCancel(ctx)
	c.rescanWait.Add(1)
	go func() {
		defer c.rescanWait.Done()
		tick := time.NewTicker(interval)
		defer tick.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
			}
			r, err := c.rescanner.step(ctx)
			select {
			case c.events <- transportEvent{coverage: &r, err: err}:
			case <-ctx.Done():
				return
			}
			if err != nil {
				return
			}
		}
	}()
}

func (c *Collector) stopRescans() {
	if c.rescanCancel != nil {
		c.rescanCancel()
	}
	c.rescanWait.Wait()
}
