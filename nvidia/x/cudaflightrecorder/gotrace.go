package cudaflightrecorder

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"runtime/trace"
	"sync"
	"time"
)

// GoTraceRecorder captures Go execution traces into an in-memory buffer.
type GoTraceRecorder interface {
	// Start begins recording Go runtime execution events.
	Start() error

	// Stop halts recording Go runtime execution events.
	Stop()

	// Snapshot returns a copy of the recorded Go execution trace.
	Snapshot(ctx context.Context) ([]byte, error)
}

// flightRecorder wraps runtime/trace.FlightRecorder to maintain
// a continuous circular trace buffer in host DRAM.
type flightRecorder struct {
	mu        sync.Mutex
	cfg       trace.FlightRecorderConfig
	fr        *trace.FlightRecorder
	startTime time.Time
	running   bool
	snapshot  []byte
}

// NewFlightRecorder returns a GoTraceRecorder backed by runtime/trace.FlightRecorder.
// The window parameter specifies the minimum event age to retain in the circular buffer.
// The maxBytes parameter specifies the maximum buffer size in bytes (0 for runtime default).
func NewFlightRecorder(window time.Duration, maxBytes uint64) GoTraceRecorder {
	cfg := trace.FlightRecorderConfig{
		MinAge:   window,
		MaxBytes: maxBytes,
	}
	return &flightRecorder{
		cfg: cfg,
		fr:  trace.NewFlightRecorder(cfg),
	}
}

func (f *flightRecorder) Start() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.running {
		return errors.New("cudaflightrecorder: flight recorder already running")
	}

	if f.fr == nil {
		f.fr = trace.NewFlightRecorder(f.cfg)
	}

	if err := f.fr.Start(); err != nil {
		return fmt.Errorf("cudaflightrecorder: start flight recorder: %w", err)
	}

	f.running = true
	f.startTime = time.Now()
	f.snapshot = nil
	return nil
}

func (f *flightRecorder) Stop() {
	f.mu.Lock()
	defer f.mu.Unlock()

	if !f.running {
		return
	}

	// Capture a snapshot before stopping so post-stop snapshots can still retrieve
	// the final recorded window.
	var buf bytes.Buffer
	if _, err := f.fr.WriteTo(&buf); err == nil && buf.Len() > 0 {
		f.snapshot = buf.Bytes()
	}

	f.fr.Stop()
	f.running = false
}

func (f *flightRecorder) Snapshot(ctx context.Context) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if !f.running {
		if len(f.snapshot) > 0 {
			cp := make([]byte, len(f.snapshot))
			copy(cp, f.snapshot)
			return cp, nil
		}
		return nil, errors.New("cudaflightrecorder: flight recorder not running")
	}

	var buf bytes.Buffer
	if _, err := f.fr.WriteTo(&buf); err != nil {
		return nil, fmt.Errorf("cudaflightrecorder: snapshot flight recorder: %w", err)
	}

	f.snapshot = buf.Bytes()
	cp := make([]byte, len(f.snapshot))
	copy(cp, f.snapshot)
	return cp, nil
}

func (f *flightRecorder) StartTime() time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.startTime
}

// syncBuffer provides a thread-safe in-memory buffer for standard execution tracing.
type syncBuffer struct {
	mu       sync.Mutex
	buf      bytes.Buffer
	maxBytes uint64
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.maxBytes > 0 && uint64(b.buf.Len()+len(p)) > b.maxBytes {
		remaining := int(b.maxBytes) - b.buf.Len()
		if remaining <= 0 {
			return len(p), nil // Silently discard when capped
		}
		p = p[:remaining]
	}
	return b.buf.Write(p)
}

func (b *syncBuffer) Bytes() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()
	cp := make([]byte, b.buf.Len())
	copy(cp, b.buf.Bytes())
	return cp
}

// standardTraceRecorder implements GoTraceRecorder using standard trace.Start
// and trace.Stop as a fallback or adapter.
type standardTraceRecorder struct {
	mu        sync.Mutex
	maxBytes  uint64
	buf       *syncBuffer
	startTime time.Time
	running   bool
	snapshot  []byte
}

// NewStandardTraceAdapter returns a GoTraceRecorder that uses trace.Start and trace.Stop.
// It serves as a clean fallback or adapter for tests and environments without FlightRecorder.
func NewStandardTraceAdapter(maxBytes uint64) GoTraceRecorder {
	return &standardTraceRecorder{
		maxBytes: maxBytes,
	}
}

func (s *standardTraceRecorder) Start() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		return errors.New("cudaflightrecorder: standard trace recorder already running")
	}

	s.buf = &syncBuffer{maxBytes: s.maxBytes}
	if err := trace.Start(s.buf); err != nil {
		return fmt.Errorf("cudaflightrecorder: start standard trace: %w", err)
	}

	s.running = true
	s.startTime = time.Now()
	s.snapshot = nil
	return nil
}

func (s *standardTraceRecorder) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.running {
		return
	}

	trace.Stop()
	s.running = false
	if s.buf != nil {
		s.snapshot = s.buf.Bytes()
	}
}

func (s *standardTraceRecorder) Snapshot(ctx context.Context) ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if !s.running {
		if len(s.snapshot) > 0 {
			cp := make([]byte, len(s.snapshot))
			copy(cp, s.snapshot)
			return cp, nil
		}
		return nil, errors.New("cudaflightrecorder: trace recorder not running")
	}

	if s.buf == nil {
		return nil, errors.New("cudaflightrecorder: trace buffer is nil")
	}

	data := s.buf.Bytes()
	if len(data) == 0 {
		return nil, errors.New("cudaflightrecorder: trace buffer empty")
	}
	return data, nil
}

func (s *standardTraceRecorder) StartTime() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.startTime
}

// adaptiveTraceRecorder attempts to use runtime/trace.FlightRecorder,
// but seamlessly falls back to standard trace.Start/Stop if flight recording
// is not available or fails to initialize.
type adaptiveTraceRecorder struct {
	mu        sync.Mutex
	window    time.Duration
	maxBytes  uint64
	active    GoTraceRecorder
	startTime time.Time
}

// NewAdaptiveTraceRecorder creates a GoTraceRecorder that prefers runtime/trace.FlightRecorder
// and falls back to standard trace.Start/Stop.
func NewAdaptiveTraceRecorder(window time.Duration, maxBytes uint64) GoTraceRecorder {
	return &adaptiveTraceRecorder{
		window:   window,
		maxBytes: maxBytes,
	}
}

func (a *adaptiveTraceRecorder) Start() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	fr := NewFlightRecorder(a.window, a.maxBytes)
	if err := fr.Start(); err == nil {
		a.active = fr
		a.startTime = time.Now()
		return nil
	}

	// Fallback to standard trace
	std := NewStandardTraceAdapter(a.maxBytes)
	if err := std.Start(); err != nil {
		return fmt.Errorf("cudaflightrecorder: start adaptive trace (flight and standard failed): %w", err)
	}

	a.active = std
	a.startTime = time.Now()
	return nil
}

func (a *adaptiveTraceRecorder) Stop() {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.active != nil {
		a.active.Stop()
	}
}

func (a *adaptiveTraceRecorder) Snapshot(ctx context.Context) ([]byte, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.active == nil {
		return nil, errors.New("cudaflightrecorder: adaptive trace recorder not running")
	}
	return a.active.Snapshot(ctx)
}

func (a *adaptiveTraceRecorder) StartTime() time.Time {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.startTime
}

// AlignSpans computes observed time spans and alignment metrics across host Go tracing
// and GPU device observations relative to the snapshot time.
func AlignSpans(
	nowNs int64,
	windowNs int64,
	goStartNs int64,
	goHasData bool,
	records []Record,
	clockUncertaintyNs int64,
) (goSpanNs, gpuSpanNs, overlapSpanNs int64, overlapReliable bool) {
	// 1. Compute Go trace span
	if goHasData {
		if goStartNs <= 0 {
			goSpanNs = windowNs
			goStartNs = nowNs - windowNs
		} else {
			elapsed := nowNs - goStartNs
			if elapsed > windowNs {
				goSpanNs = windowNs
				goStartNs = nowNs - windowNs
			} else if elapsed > 0 {
				goSpanNs = elapsed
			} else {
				goSpanNs = 0
			}
		}
	}

	// 2. Compute GPU records span
	var gpuMinTs, gpuMaxTs int64
	if len(records) > 0 {
		gpuMinTs = records[0].TimestampNs
		gpuMaxTs = records[0].TimestampNs + int64(records[0].DurationNs)
		for _, rec := range records[1:] {
			if rec.TimestampNs < gpuMinTs {
				gpuMinTs = rec.TimestampNs
			}
			end := rec.TimestampNs + int64(rec.DurationNs)
			if end > gpuMaxTs {
				gpuMaxTs = end
			}
		}
		if gpuMaxTs > gpuMinTs {
			gpuSpanNs = gpuMaxTs - gpuMinTs
		}
	}

	// 3. Compute overlap between host Go execution trace and GPU observations
	if goHasData && len(records) > 0 && goSpanNs > 0 {
		goEndNs := nowNs
		overlapStart := goStartNs
		if gpuMinTs > overlapStart {
			overlapStart = gpuMinTs
		}
		overlapEnd := goEndNs
		if gpuMaxTs < overlapEnd {
			overlapEnd = gpuMaxTs
		}

		if overlapEnd > overlapStart {
			overlapSpanNs = overlapEnd - overlapStart
		}
	}

	// 4. Overlap reliability
	// An overlap is reliable if:
	// - Non-zero overlap span exists
	// - Clock uncertainty is non-negative and strictly below 100 microseconds (100,000 ns)
	// - Clock uncertainty is strictly less than the overlap span
	const maxClockUncertaintyNs = 100000 // 100 microseconds
	if overlapSpanNs > 0 && clockUncertaintyNs >= 0 &&
		clockUncertaintyNs < maxClockUncertaintyNs &&
		clockUncertaintyNs < overlapSpanNs {
		overlapReliable = true
	}

	return goSpanNs, gpuSpanNs, overlapSpanNs, overlapReliable
}
