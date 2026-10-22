package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"sync"
)

// The buffered recorder in span.go holds every interval until the run ends
// and drops what does not fit in spanRecordLimit records or spanByteLimit
// bytes. At op granularity a 2048-token prefill produces on the order of
// 700,000 intervals, seven times that record cap, so the sidecar it writes
// is a silent sample rather than a trace.
//
// A streamRecorder writes each record as its span closes. What it holds is
// the set of spans currently open, which is bounded by nesting depth rather
// than by run length, so raising the cap is not the fix here -- there is no
// cap to raise. It keeps the gputrace.host-event/v1 schema exactly: that
// schema has no labels by design, so fine granularity has to live in the
// span name, which is what op-level names already are.
//
// span.go is a byte-identical copy of gputrace's hostspan, so this lives
// beside it rather than inside it.

// streamOpen is one span that has begun and not yet ended.
type streamOpen struct {
	name     string
	uptimeNS uint64
}

// A streamRecorder streams host-event records to a file as spans close.
// Its methods are safe for concurrent use.
type streamRecorder struct {
	level spanLevel
	runID string
	path  string

	mu   sync.Mutex
	f    streamSink
	buf  *bufio.Writer
	enc  *json.Encoder
	open map[int]streamOpen

	next     int
	ended    int
	written  int
	openPeak int
	// err holds the first write failure. Recording continues to count
	// after it so the accounting still reports the true totals, but no
	// further bytes are written and Close reports the failure.
	err error
}

// A streamSink is what the recorder writes to. *os.File satisfies it; the
// interface exists so a test can supply a writer that fails, which is the
// only way to reach the accounting's short-file path.
type streamSink interface {
	io.Writer
	Stat() (os.FileInfo, error)
	Close() error
}

// newStreamRecorder creates the sidecar and installs nothing; the caller
// assigns it to streamSpans.
func newStreamRecorder(level spanLevel, runID, path string) (*streamRecorder, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create streaming span sidecar: %w", err)
	}
	return newStreamRecorderSink(level, runID, path, f), nil
}

// newStreamRecorderSink is newStreamRecorder over an already-open sink.
func newStreamRecorderSink(level spanLevel, runID, path string, f streamSink) *streamRecorder {
	buf := bufio.NewWriterSize(f, 1<<20)
	return &streamRecorder{
		level: level, runID: runID, path: path,
		f: f, buf: buf, enc: json.NewEncoder(buf),
		open: make(map[int]streamOpen),
	}
}

// streamSpans is the process streaming recorder, nil when streaming is off.
var streamSpans *streamRecorder

// streamEnd ends a streamed span. The zero value is the no-op end.
type streamEnd struct {
	r  *streamRecorder
	id int
}

// beginStream starts a streamed span, or returns the no-op end when level
// is deeper than what was asked for. Unlike begin it cannot drop the span:
// the record is written at end, and nothing between here and there is
// bounded by a record count.
func beginStream(level spanLevel, name string) streamEnd {
	r := streamSpans
	if r == nil || level > r.level {
		return streamEnd{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	id := r.next
	r.next++
	r.open[id] = streamOpen{name: name, uptimeNS: uptimeNS()}
	if len(r.open) > r.openPeak {
		r.openPeak = len(r.open)
	}
	return streamEnd{r: r, id: id}
}

func (e streamEnd) end() {
	if e.r == nil {
		return
	}
	r := e.r
	r.mu.Lock()
	defer r.mu.Unlock()
	s, ok := r.open[e.id]
	if !ok {
		// Already ended. tracedEnd makes end idempotent, so this is
		// reachable only through a second recorder wrapping the same
		// span; writing the record twice would double-count it.
		return
	}
	delete(r.open, e.id)
	d := uptimeNS() - s.uptimeNS
	if d == 0 {
		// A span that began and ended inside one clock tick still
		// happened, and a zero-duration interval carries no width for
		// a viewer to draw.
		d = 1
	}
	r.ended++
	if r.err != nil {
		return
	}
	if err := r.enc.Encode(hostEventRecord{
		Schema:      "gputrace.host-event/v1",
		Kind:        "interval",
		RunID:       r.runID,
		ID:          strconv.Itoa(e.id),
		Name:        s.name,
		ClockDomain: "cpu_uptime_ns",
		TimestampNS: s.uptimeNS,
		DurationNS:  d,
	}); err != nil {
		r.err = fmt.Errorf("write streaming span sidecar: %w", err)
		return
	}
	// Counted only on encode success, so that Encoded can never claim a
	// record the encoder did not accept. It is still not a count of
	// records on disk; see streamAccounting.
	r.written++
}

// streamAccounting is the receipt written beside the sidecar. It has to be
// readable on its own: anyone holding the sidecar and this file, without
// the process's exit status, must be able to tell a complete trace from a
// short one.
//
// So the three counts are kept apart. Begun is every span that started,
// Ended is every span that closed, and Encoded is only those the encoder
// accepted. Begun minus Ended is what the run was still inside when it
// stopped; Ended minus Encoded is what a failing writer swallowed.
//
// Encoded counts what the encoder took, not what reached the disk: the
// sidecar is buffered, so a run whose final flush fails can have encoded
// every span and still hold a short file. That is what Complete and
// BytesFile are for, and why neither field is named "written". Dropped
// is structurally zero because this recorder has no cap. Complete is false
// unless the counts agree and every write, the final flush, and the close
// all succeeded, and Error carries why when they did not.
type streamAccounting struct {
	Schema    string `json:"schema"`
	RunID     string `json:"run_id"`
	Path      string `json:"path"`
	Level     string `json:"level"`
	Begun     int    `json:"spans_begun"`
	Ended     int    `json:"spans_ended"`
	Encoded   int    `json:"spans_encoded"`
	Unended   int    `json:"spans_unended"`
	Dropped   int    `json:"spans_dropped"`
	OpenPeak  int    `json:"open_peak"`
	BytesFile int64  `json:"sidecar_bytes"`
	Complete  bool   `json:"complete"`
	Error     string `json:"error,omitempty"`
}

// accountingPath is the receipt's path. It sits beside the sidecar rather
// than inside it because the host-event reader decodes with
// DisallowUnknownFields and would reject a trailing summary line.
func accountingPath(sidecar string) string { return sidecar + ".accounting.json" }

// Close flushes the sidecar and writes the accounting receipt. It reports
// the first write error, and a run that ends inside open spans still gets
// a receipt saying how many.
func (r *streamRecorder) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	flushErr := r.buf.Flush()
	if flushErr != nil {
		flushErr = fmt.Errorf("flush streaming span sidecar: %w", flushErr)
	}
	// Stat before Close, and after Flush so the size is the file's, not
	// the buffer's. A stat failure is an error like any other: without it
	// the receipt would report sidecar_bytes 0 and look like an empty run.
	var bytes int64
	info, statErr := r.f.Stat()
	if statErr != nil {
		statErr = fmt.Errorf("stat streaming span sidecar: %w", statErr)
	} else {
		bytes = info.Size()
	}
	closeErr := r.f.Close()
	if closeErr != nil {
		closeErr = fmt.Errorf("close streaming span sidecar: %w", closeErr)
	}
	outErr := errors.Join(r.err, flushErr, statErr, closeErr)
	acc := streamAccounting{
		Schema:    "ds4-min.span-accounting/v1",
		RunID:     r.runID,
		Path:      r.path,
		Level:     r.level.String(),
		Begun:     r.next,
		Ended:     r.ended,
		Encoded:   r.written,
		Unended:   r.next - r.ended,
		Dropped:   0,
		OpenPeak:  r.openPeak,
		BytesFile: bytes,
		Complete:  outErr == nil && r.next == r.ended && r.ended == r.written,
	}
	if outErr != nil {
		acc.Error = outErr.Error()
	}
	accErr := writeStreamAccounting(accountingPath(r.path), acc)
	return errors.Join(outErr, accErr)
}

func writeStreamAccounting(path string, acc streamAccounting) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create span accounting: %w", err)
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(acc); err != nil {
		return errors.Join(fmt.Errorf("write span accounting: %w", err), f.Close())
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close span accounting: %w", err)
	}
	return nil
}

// String names the level as the -span-level flag spells it. span.go parses
// these and never prints them, and the accounting receipt has to record
// which granularity produced the file.
func (l spanLevel) String() string {
	switch l {
	case levelOff:
		return "off"
	case levelPhase:
		return "phase"
	case levelToken:
		return "token"
	case levelLayer:
		return "layer"
	case levelOp:
		return "op"
	}
	return "spanLevel(" + strconv.Itoa(int(l)) + ")"
}
