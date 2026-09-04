package metalprof

import (
	"fmt"
	"io"
	"sync"
	"time"
)

// KernelEvent records caller-supplied information about a Metal compute event.
// Duration has the clock semantics of the value passed to [FlightRecorder.Record].
// TimestampNs is host wall time recorded when Record is called.
type KernelEvent struct {
	Name        string
	Duration    time.Duration
	GridSize    [3]int
	TGSize      [3]int
	TimestampNs uint64
	Labels      map[string]string
}

// TimelineEvent represents a caller-supplied interval anchored to host wall time.
type TimelineEvent struct {
	Name       string            `json:"name"`
	StartNs    uint64            `json:"start_ns"`
	DurationNs uint64            `json:"duration_ns"`
	Labels     map[string]string `json:"labels,omitempty"`
}

// FlightRecorder maintains an in-memory ring buffer of caller-supplied events.
type FlightRecorder struct {
	mu       sync.RWMutex
	events   []KernelEvent
	capacity int
	head     int
	count    int
}

// NewFlightRecorder creates a flight recorder holding up to capacity events.
func NewFlightRecorder(capacity int) *FlightRecorder {
	if capacity <= 0 {
		capacity = 2048
	}
	return &FlightRecorder{
		events:   make([]KernelEvent, capacity),
		capacity: capacity,
	}
}

// Record adds an event to the in-memory ring buffer. Record does not measure
// dur or verify that it came from a GPU clock.
func (fr *FlightRecorder) Record(name string, dur time.Duration, grid, tg [3]int, labels map[string]string) {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	lblCopy := make(map[string]string, len(labels))
	for k, v := range labels {
		lblCopy[k] = v
	}

	fr.events[fr.head] = KernelEvent{
		Name:        name,
		Duration:    dur,
		GridSize:    grid,
		TGSize:      tg,
		TimestampNs: uint64(time.Now().UnixNano()),
		Labels:      lblCopy,
	}
	fr.head = (fr.head + 1) % fr.capacity
	if fr.count < fr.capacity {
		fr.count++
	}
}

// Events returns a chronological snapshot of the recorded kernel events.
func (fr *FlightRecorder) Events() []KernelEvent {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	out := make([]KernelEvent, fr.count)
	if fr.count < fr.capacity {
		copy(out, fr.events[:fr.count])
		return out
	}
	// Ring wrapped around
	n1 := copy(out, fr.events[fr.head:])
	copy(out[n1:], fr.events[:fr.head])
	return out
}

// Timeline returns chronological timeline events from the recorder.
func (fr *FlightRecorder) Timeline() []TimelineEvent {
	events := fr.Events()
	out := make([]TimelineEvent, len(events))
	for i, ev := range events {
		out[i] = TimelineEvent{
			Name:       ev.Name,
			StartNs:    ev.TimestampNs,
			DurationNs: uint64(ev.Duration.Nanoseconds()),
			Labels:     ev.Labels,
		}
	}
	return out
}

// Reset clears the flight recorder buffer.
func (fr *FlightRecorder) Reset() {
	fr.mu.Lock()
	defer fr.mu.Unlock()
	fr.head = 0
	fr.count = 0
}

// SnapshotPprof serializes the current ring buffer history into a compressed pprof profile.
func (fr *FlightRecorder) SnapshotPprof(w io.Writer) error {
	events := fr.Events()
	if len(events) == 0 {
		b := NewBuilder()
		return b.Encode(w)
	}

	b := NewBuilder()
	for _, ev := range events {
		fnID := b.AddFunction(ev.Name, "MetalShaders.metal", 1)
		locID := b.AddLocation(fnID, 1)

		vals := SampleValues{
			Samples:   1,
			GPUTimeNs: ev.Duration.Nanoseconds(),
		}
		b.AddSample([]uint64{locID}, vals, ev.Labels)
	}

	return b.Encode(w)
}

// SnapshotSummary formats a quick summary table of recent kernel execution times.
func (fr *FlightRecorder) SnapshotSummary() string {
	events := fr.Events()
	if len(events) == 0 {
		return "metalprof: flight recorder empty"
	}

	kernelTotals := make(map[string]time.Duration)
	kernelCounts := make(map[string]int)
	var totalDuration time.Duration

	for _, ev := range events {
		kernelTotals[ev.Name] += ev.Duration
		kernelCounts[ev.Name]++
		totalDuration += ev.Duration
	}

	res := fmt.Sprintf("=== Metal FlightRecorder: %d Events (Total Recorded Time: %v) ===\n", len(events), totalDuration)
	for name, dur := range kernelTotals {
		cnt := kernelCounts[name]
		avg := dur / time.Duration(cnt)
		res += fmt.Sprintf("  - %-32s : count=%-5d total=%-10v avg=%-10v\n", name, cnt, dur, avg)
	}
	return res
}
