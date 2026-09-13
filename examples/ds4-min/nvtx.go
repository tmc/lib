package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"

	"github.com/tmc/lib/nvidia/x/nvtx"
)

// NVTX ranges mirror the spans the recorder keeps, but are not subject to
// its record and byte caps. The sidecar is bounded because its reader holds
// it whole; Nsight streams instead, so a run that drops sidecar records
// still has a complete device-side timeline. Without a profiler attached
// every call here is a no-op, so the ranges cost nothing in ordinary runs.

// A tracedEnd ends a span and its NVTX range. The zero value is the no-op
// end, so a call site needs no guard for tracing being off.
// The zero value's nil done makes end a no-op, and a non-nil done makes it
// idempotent, so a call site can both defer end for its error paths and
// call it at the interval's real boundary. Closing twice would otherwise
// restamp the span's duration and unbalance the NVTX range.
type tracedEnd struct {
	span   spanEnd
	stream streamEnd
	rng    nvtx.RangeID
	done   *bool
}

func (e tracedEnd) end() {
	if e.done == nil || *e.done {
		return
	}
	*e.done = true
	e.span.end()
	e.stream.end()
	if e.rng != 0 {
		nvtx.RangeEnd(e.rng)
	}
}

// tracePhase runs fn as one traced phase, closing the span and its NVTX
// range however fn returns. The phase spans go through here rather than
// pairing begin and end at the call site, so that the close lives in one
// place a test can drive: a defer written beside a long loop is exactly
// the kind that a later edit drops without anything noticing.
func tracePhase(level spanLevel, name string, fn func() error) error {
	e := beginTraced(level, name, nil)
	defer e.end()
	return fn()
}

// beginTraced starts a recorded span and an NVTX range for the same
// interval. The two are independent on purpose: the recorder drops a span
// past its caps, and the range still opens, so a truncated sidecar does
// not truncate the Nsight timeline with it.
//
// span.go is a copy of gputrace's hostspan and stays byte-identical to it,
// so the pairing lives here rather than inside begin.
func beginTraced(level spanLevel, name string, labels map[string]string) tracedEnd {
	if !tracing(level) {
		return tracedEnd{}
	}
	return tracedEnd{
		span:   begin(level, name, labels),
		stream: beginStream(level, name),
		rng:    nvtx.RangeStart(name),
		done:   new(bool),
	}
}

// tracing reports whether any recorder wants a span at this level. The two
// recorders are alternatives -- the buffered one in span.go caps its record
// count, the streaming one in span_stream.go does not -- and a run installs
// at most one, but the NVTX range opens for either.
func tracing(level spanLevel) bool {
	if spans != nil && level <= spans.level {
		return true
	}
	return streamSpans != nil && level <= streamSpans.level
}

// beginTracedLayer is beginTraced for one transformer layer. The NVTX name
// carries the layer index because Nsight groups by name and has no labels;
// the span keeps it as a label so "gputrace pprof" still sums the 43 layers
// into one frame.
func beginTracedLayer(i int) tracedEnd {
	if !tracing(levelLayer) {
		return tracedEnd{}
	}
	// The streamed sidecar carries the index in the name because
	// gputrace.host-event/v1 has no labels; the buffered sidecar keeps it
	// as a label, which is the same reason the NVTX name carries it.
	return tracedEnd{
		span:   beginLayer(i),
		stream: beginStream(levelLayer, "layer "+strconv.Itoa(i)),
		rng:    nvtx.RangeStart("layer " + strconv.Itoa(i)),
		done:   new(bool),
	}
}

// requireNVTXProvider reports why calibration cannot run. It is checked at
// flag time because the alternative is paying a model load and a GPU
// context to learn that nsys is not attached.
func requireNVTXProvider() error {
	if !nvtx.Enabled() {
		return errors.New("-calibration needs an attached NVTX provider; run under nsys")
	}
	return nil
}

// A calibration brackets one NVTX range start between two reads of the span
// clock. Nsight stamps the range's start in its own domain, so the offset
// from the span clock to that domain lies in [start-after, start-before].
// One calibration gives an interval, not an offset: two taken far apart
// must admit a common value, or the two domains are not related by a
// constant and merging them would invent the difference.
type calibration struct {
	Schema string `json:"schema"`
	RunID  string `json:"run_id"`
	Label  string `json:"label"`
	// Name is the NVTX range text, and the only key the Nsight export
	// offers: the RangeStart handle is not exported.
	Name string `json:"name"`
	// Before and After bracket the RangeStart call in span-clock
	// nanoseconds. After is read before the range is closed so the
	// bracket covers the start alone.
	Before uint64 `json:"host_before_ns"`
	After  uint64 `json:"host_after_ns"`
}

// calibrationName is unique per run and label so a capture holding more
// than one run cannot match the wrong range.
func calibrationName(runID, label string) string {
	return fmt.Sprintf("ds4-min/calibration/%s/%s", runID, label)
}

// calibrate opens and immediately closes a uniquely named NVTX range,
// bracketing the start with two reads of the span clock.
func calibrate(runID, label string) calibration {
	name := calibrationName(runID, label)
	before := uptimeNS()
	id := nvtx.RangeStart(name)
	after := uptimeNS()
	nvtx.RangeEnd(id)
	return calibration{
		Schema: "ds4-min.calibration/v1",
		RunID:  runID, Label: label, Name: name,
		Before: before, After: after,
	}
}

// valid reports whether the bracket is usable. Equal readings are the
// tightest bracket, not a broken one: they pin the offset to a point. Only
// a reversed bracket is unusable, and on a monotonic clock it means the
// clock is not the one the spans are stamped on.
func (c calibration) valid() bool { return c.After >= c.Before }

// writeCalibrations emits the receipt the offset is later derived from.
// It refuses to write a bracket it cannot defend, because a receipt that
// records an unusable bracket reads downstream as a measured one.
func writeCalibrations(w io.Writer, cs []calibration) error {
	if len(cs) == 0 {
		return errors.New("no calibrations recorded")
	}
	enc := json.NewEncoder(w)
	for _, c := range cs {
		if !c.valid() {
			return fmt.Errorf("calibration %q: host bracket %d..%d runs backwards", c.Label, c.Before, c.After)
		}
		if err := enc.Encode(c); err != nil {
			return fmt.Errorf("write calibration %q: %w", c.Label, err)
		}
	}
	return nil
}
