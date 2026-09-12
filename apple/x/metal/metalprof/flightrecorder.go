package metalprof

import (
	"fmt"
	"io"
	"maps"
	"runtime"
	"slices"
	"sync"
	"time"
)

// KernelEvent records caller-supplied information about a Metal compute event.
// Duration has the clock semantics of the value passed to [FlightRecorder.Record].
// TimestampNs is host wall time recorded when Record is called.
type KernelEvent struct {
	stack       []runtime.Frame
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
// dur or verify that it came from a GPU clock. The pprof snapshot includes the
// Go stack at Record, which is the recording site, not necessarily the submission
// site if Record is called from a completion callback.
func (fr *FlightRecorder) Record(name string, dur time.Duration, grid, tg [3]int, labels map[string]string) {
	var pcs [64]uintptr
	n := runtime.Callers(2, pcs[:])
	frames := runtime.CallersFrames(pcs[:n])
	var stack []runtime.Frame
	for {
		frame, more := frames.Next()
		stack = append(stack, frame)
		if !more {
			break
		}
	}
	fr.mu.Lock()
	defer fr.mu.Unlock()

	lblCopy := make(map[string]string, len(labels))
	for k, v := range labels {
		lblCopy[k] = v
	}

	fr.events[fr.head] = KernelEvent{
		stack:       stack,
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
	} else {
		// Ring wrapped around
		n1 := copy(out, fr.events[fr.head:])
		copy(out[n1:], fr.events[:fr.head])
	}
	for i := range out {
		out[i].Labels = maps.Clone(out[i].Labels)
		out[i].stack = slices.Clone(out[i].stack)
	}
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
// Each sample has a source-unmapped kernel leaf followed by the Go recording
// stack. No shader source mapping or asynchronous submission correlation is inferred.
func (fr *FlightRecorder) SnapshotPprof(w io.Writer) error {
	events := fr.Events()
	if len(events) == 0 {
		b := NewBuilder()
		return b.Encode(w)
	}

	b := NewBuilder()
	for _, ev := range events {
		fnID := b.AddFunction(ev.Name, "", 0)
		locID := b.AddLocation(fnID, 0)

		vals := SampleValues{
			Samples:   1,
			GPUTimeNs: ev.Duration.Nanoseconds(),
		}
		locations := []uint64{locID}
		for _, frame := range ev.stack {
			fn := b.AddFunction(frame.Function, frame.File, int64(frame.Line))
			locations = append(locations, b.AddLocation(fn, int64(frame.Line)))
		}
		labels := maps.Clone(ev.Labels)
		if labels == nil {
			labels = make(map[string]string)
		}
		labels["stack_origin"] = "record"
		b.AddSample(locations, vals, labels)
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
