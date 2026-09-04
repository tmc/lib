package metalmem

import "sync"

// Tracker accounts explicit Metal buffer allocations. It does not measure
// driver caches, residency, compression, or other opaque device memory.
// The zero Tracker is ready for concurrent use.
type Tracker struct {
	mu      sync.Mutex
	current uint64
	peak    uint64
}

type charge struct {
	once    sync.Once
	tracker *Tracker
	bytes   uint64
}

func (t *Tracker) acquire(bytes uint64) *charge {
	if t == nil || bytes == 0 {
		return nil
	}
	t.mu.Lock()
	t.current += bytes
	if t.current > t.peak {
		t.peak = t.current
	}
	t.mu.Unlock()
	return &charge{tracker: t, bytes: bytes}
}

func (c *charge) release() {
	if c == nil {
		return
	}
	c.once.Do(func() {
		c.tracker.mu.Lock()
		c.tracker.current -= c.bytes
		c.tracker.mu.Unlock()
	})
}

// Snapshot returns the current and high-water explicit allocation counts.
func (t *Tracker) Snapshot() (current, peak uint64) {
	if t == nil {
		return 0, 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.current, t.peak
}

// ResetPeak begins a new high-water interval at the current allocation count.
func (t *Tracker) ResetPeak() {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.peak = t.current
	t.mu.Unlock()
}
