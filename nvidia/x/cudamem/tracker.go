package cudamem

import "sync"

// Tracker accounts for live device bytes allocated through it.
//
// Reset starts a new observation interval. The interval peak begins at the
// current allocation, so persistent allocations remain part of the measured
// high-water mark.
type Tracker struct {
	mu      sync.Mutex
	current uint64
	peak    uint64
}

// Reset starts a new observation interval and returns the current allocation.
func (t *Tracker) Reset() uint64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.peak = t.current
	return t.current
}

// Snapshot returns the current allocation and the peak since the last Reset.
func (t *Tracker) Snapshot() (current, peak uint64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.current, t.peak
}

func (t *Tracker) allocated(bytes uint64) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.current += bytes
	if t.current > t.peak {
		t.peak = t.current
	}
	t.mu.Unlock()
}

func (t *Tracker) freed(bytes uint64) {
	if t == nil {
		return
	}
	t.mu.Lock()
	if bytes > t.current {
		t.current = 0
	} else {
		t.current -= bytes
	}
	t.mu.Unlock()
}
