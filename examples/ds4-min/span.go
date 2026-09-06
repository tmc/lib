package main

// Host-span recording for gputrace.
//
// One recorder feeds the two sidecar protocols gputrace reads, because a
// Metal run and a CUDA run are analyzed by different commands:
//
//   - gputrace.host-event/v1 (-spans file), newline-delimited JSON stamped
//     on the uptime clock. "gputrace host-receipt" binds it to a Metal
//     timing sidecar, whose CPU timestamps are mach_absolute_time.
//   - gpuevent spans appended to $GPUTRACE_APP_EVENTS, stamped on
//     CLOCK_REALTIME and declared clock "unix". "gputrace capture" sets
//     that variable and merges the file into the .gpucapture bundle, whose
//     clock_sync record translates the stamps into the CUPTI domain.
//
// Without spans gputrace degrades to bare kernel names: every kernel is
// attributed to itself and nothing says which phase, token, or layer paid
// for it. The names below are what "gputrace pprof" turns into stacks, so
// they are generic rather than indexed: 43 identical layers aggregate.

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"sync"
)

// A spanLevel bounds how deep recording goes. Both sidecars are bounded
// artifacts — the host-event reader rejects a file over 100,000 records —
// so depth is a choice, not a default. One prefill of 2048 tokens is
// 88,064 layer spans; the same prefill at levelOp would be over a million.
type spanLevel int

const (
	levelOff spanLevel = iota
	levelPhase
	levelToken
	levelLayer
	levelOp
)

const spanRecordLimit = 100_000

func parseSpanLevel(s string) (spanLevel, error) {
	switch s {
	case "off":
		return levelOff, nil
	case "phase":
		return levelPhase, nil
	case "token":
		return levelToken, nil
	case "layer":
		return levelLayer, nil
	case "op":
		return levelOp, nil
	}
	return levelOff, fmt.Errorf("unknown span level %q", s)
}

// A span is one recorded interval, held until Close writes it. Both
// clocks are stamped at Begin because the two sidecars disagree about
// which one they are in, and reading them at Begin is what makes the
// pair describe the same instant.
type span struct {
	name      string
	labels    map[string]string
	uptimeNS  uint64
	unixNS    uint64
	durationN uint64
}

// A spanRecorder collects intervals for one run. Its methods are safe for
// concurrent use, though ds4-min evaluates on one goroutine.
type spanRecorder struct {
	level spanLevel
	runID string

	mu      sync.Mutex
	spans   []span
	dropped int
}

// spanEnd ends a span. The zero value is the no-op end, so a call site
// needs no guard for recording being off.
type spanEnd struct {
	r     *spanRecorder
	index int
}

func (e spanEnd) end() {
	if e.r == nil {
		return
	}
	e.r.mu.Lock()
	s := &e.r.spans[e.index]
	s.durationN = uptimeNS() - s.uptimeNS
	if s.durationN == 0 {
		// The reader rejects a zero duration, and a span that began and
		// ended inside one clock tick still happened.
		s.durationN = 1
	}
	e.r.mu.Unlock()
}

// spans is the recorder for the process, nil when recording is off.
var spans *spanRecorder

// begin starts a span named name, or does nothing when level is deeper
// than what was asked for. Callers hold the returned value and call end.
func begin(level spanLevel, name string, labels map[string]string) spanEnd {
	r := spans
	if r == nil || level > r.level {
		return spanEnd{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.spans) >= spanRecordLimit {
		r.dropped++
		return spanEnd{}
	}
	r.spans = append(r.spans, span{
		name: name, labels: labels,
		uptimeNS: uptimeNS(), unixNS: unixNS(),
	})
	return spanEnd{r: r, index: len(r.spans) - 1}
}

// beginLayer starts a span for one transformer layer. The layer index is
// a label rather than part of the name so that "gputrace pprof" sums the
// 43 layers into one stack frame instead of 43 leaves.
func beginLayer(i int) spanEnd {
	if spans == nil || levelLayer > spans.level {
		return spanEnd{}
	}
	return begin(levelLayer, "layer", map[string]string{"layer": strconv.Itoa(i)})
}

// beginOp starts a span for one encoded operation.
func beginOp(name string) spanEnd {
	if spans == nil || levelOp > spans.level {
		return spanEnd{}
	}
	return begin(levelOp, name, nil)
}

// newSpanRecorder installs the process recorder. runID identifies the run
// in both sidecars; the host-event reader refuses a file that mixes IDs.
func newSpanRecorder(level spanLevel, runID string) *spanRecorder {
	return &spanRecorder{level: level, runID: runID}
}

// hostEventRecord is one gputrace.host-event/v1 line. The reader decodes
// with DisallowUnknownFields, so the field set is exact.
type hostEventRecord struct {
	Schema      string `json:"schema"`
	Kind        string `json:"kind"`
	RunID       string `json:"run_id"`
	ID          string `json:"id"`
	Name        string `json:"name"`
	ClockDomain string `json:"clock_domain"`
	TimestampNS uint64 `json:"timestamp_ns"`
	DurationNS  uint64 `json:"duration_ns"`
}

// gpuEventSpan is one gpuevent span record. The decoder dispatches on
// kind and silently drops a record without it, which reads downstream as
// a workload that emitted no spans at all. Clock is "unix" because the
// stamps are CLOCK_REALTIME; the bundle's clock_sync record translates
// them into the capture's CUPTI domain at decode.
type gpuEventSpan struct {
	Kind    string            `json:"kind"`
	Name    string            `json:"name"`
	StartNS uint64            `json:"start_ns"`
	EndNS   uint64            `json:"end_ns"`
	Labels  map[string]string `json:"labels,omitempty"`
	Clock   string            `json:"clock"`
}

// WriteHostEvents writes the gputrace.host-event/v1 sidecar.
func (r *spanRecorder) WriteHostEvents(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create span sidecar: %w", err)
	}
	enc := json.NewEncoder(f)
	r.mu.Lock()
	defer r.mu.Unlock()
	for i, s := range r.spans {
		if s.durationN == 0 {
			continue // never ended; the run failed inside it
		}
		if err := enc.Encode(hostEventRecord{
			Schema:      "gputrace.host-event/v1",
			Kind:        "interval",
			RunID:       r.runID,
			ID:          strconv.Itoa(i),
			Name:        s.name,
			ClockDomain: "cpu_uptime_ns",
			TimestampNS: s.uptimeNS,
			DurationNS:  s.durationN,
		}); err != nil {
			return errors.Join(fmt.Errorf("write span sidecar: %w", err), f.Close())
		}
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close span sidecar: %w", err)
	}
	return nil
}

// AppendGPUEvents appends gpuevent spans to the GPUTRACE_APP_EVENTS
// sidecar. It appends because every process in a captured target tree
// shares the one file.
func (r *spanRecorder) AppendGPUEvents(path string) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("open app-events sidecar: %w", err)
	}
	enc := json.NewEncoder(f)
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, s := range r.spans {
		if s.durationN == 0 {
			continue
		}
		if err := enc.Encode(gpuEventSpan{
			Kind:    "span",
			Name:    s.name,
			StartNS: s.unixNS,
			EndNS:   s.unixNS + s.durationN,
			Labels:  s.labels,
			Clock:   "unix",
		}); err != nil {
			return errors.Join(fmt.Errorf("write app-events sidecar: %w", err), f.Close())
		}
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close app-events sidecar: %w", err)
	}
	return nil
}

// Dropped reports spans discarded at the record limit.
func (r *spanRecorder) Dropped() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.dropped
}

// Len reports recorded spans.
func (r *spanRecorder) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.spans)
}
