package egraph

import (
	"sync"
	"time"
)

// PGOTracker records measured candidate durations. The zero value is usable.
type PGOTracker struct {
	mu    sync.RWMutex
	sums  map[string]time.Duration
	count map[string]int
}

// Observation is one measured execution to feed the PGO tracker.
type Observation struct {
	Candidate string
	Duration  time.Duration
	Labels    map[string]string
}

// Record adds one positive observation for candidate. Candidate is normally
// TuningCandidate.Key or a stable kernel name.
func (t *PGOTracker) Record(candidate string, duration time.Duration) {
	if t == nil || candidate == "" || duration <= 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.sums == nil {
		t.sums = make(map[string]time.Duration)
		t.count = make(map[string]int)
	}
	t.sums[candidate] += duration
	t.count[candidate]++
}

// RecordObservation records a measured execution. Labels are retained only by
// the caller; they are accepted here so producer adapters can pass through
// contextual metadata without a second API or a lossy conversion.
func (t *PGOTracker) RecordObservation(o Observation) {
	t.Record(o.Candidate, o.Duration)
}

// Score returns the mean measured duration for candidate.
func (t *PGOTracker) Score(candidate string) (time.Duration, bool) {
	if t == nil {
		return 0, false
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	n := t.count[candidate]
	if n == 0 {
		return 0, false
	}
	return t.sums[candidate] / time.Duration(n), true
}
