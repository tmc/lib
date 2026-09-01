package main

import (
	"fmt"
	"sync"
	"time"

	"github.com/tmc/lib/apple/x/metal/egraph"
)

// DispatchTuner selects between compiled GEMV pipeline variants using measured
// GPU command-buffer durations.
type DispatchTuner struct {
	mu sync.Mutex

	tracker    *egraph.PGOTracker
	candidates []int
	samples    map[int][]time.Duration
	current    int
	locked     bool
}

func NewDispatchTuner() *DispatchTuner {
	return &DispatchTuner{
		tracker:    egraph.NewPGOTracker("FastGEMV pipeline variant"),
		candidates: []int{64, 128},
		samples:    make(map[int][]time.Duration),
		current:    64,
	}
}

// Width returns the selected threadgroup width for the next forward pass.
func (t *DispatchTuner) Width() int {
	if t == nil {
		return 128
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.current
}

// Tracker returns the observed PGO telemetry tracker.
func (t *DispatchTuner) Tracker() *egraph.PGOTracker {
	if t == nil {
		return nil
	}
	return t.tracker
}

// Record measures one executed variant and changes pipelines only after each
// candidate has accumulated three GPU measurements.
func (t *DispatchTuner) Record(width int, gpuTime time.Duration) {
	if t == nil || gpuTime <= 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.locked {
		return
	}
	t.samples[width] = append(t.samples[width], gpuTime)
	t.tracker.RecordTrial("FastGEMV", [3]int{width, 1, 1}, float64(gpuTime)/float64(time.Microsecond), 1)

	for _, candidate := range t.candidates {
		if len(t.samples[candidate]) < 3 {
			if t.current != candidate {
				t.tracker.RecordHotSwap("FastGEMV", fmt.Sprintf("threadgroup=%d", t.current), fmt.Sprintf("threadgroup=%d", candidate), "continue measured PGO exploration")
			}
			t.current = candidate
			return
		}
	}

	best := t.candidates[0]
	bestMean := meanDuration(t.samples[best])
	for _, candidate := range t.candidates[1:] {
		if mean := meanDuration(t.samples[candidate]); mean < bestMean {
			best, bestMean = candidate, mean
		}
	}
	from := t.current
	t.current = best
	t.locked = true
	t.tracker.RecordIteration("measured FastGEMV variant", float64(bestMean)/float64(time.Millisecond), true)
	if from != best {
		t.tracker.RecordHotSwap("FastGEMV", fmt.Sprintf("threadgroup=%d", from), fmt.Sprintf("threadgroup=%d", best), "lowest mean GPU duration across three measured trials per variant")
	}
	t.tracker.LockConvergence(true)
}

func meanDuration(samples []time.Duration) time.Duration {
	var total time.Duration
	for _, sample := range samples {
		total += sample
	}
	return total / time.Duration(len(samples))
}
