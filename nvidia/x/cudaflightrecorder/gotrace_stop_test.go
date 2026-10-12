package cudaflightrecorder

import (
	"bytes"
	"context"
	"errors"
	"testing"
	"time"
)

// TestCloseReportsFinalTraceWriteFailure stops the runtime flight recorder
// underneath the wrapper so that the final WriteTo in Stop fails. Close must
// report that failure, and the window retained by the earlier snapshot must
// survive unchanged and keep its unvalidated status.
func TestCloseReportsFinalTraceWriteFailure(t *testing.T) {
	fr := NewFlightRecorder(time.Second, 0).(*flightRecorder)
	rec, err := New(Config{
		Window:         5 * time.Second,
		Capacity:       8,
		GoTraceEnabled: true,
		GoTracer:       fr,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { fr.fr.Stop() })

	retained, err := fr.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(retained) == 0 {
		t.Fatal("running flight recorder returned an empty window")
	}

	fr.fr.Stop()
	closeErr := rec.Close()
	if closeErr == nil {
		t.Fatal("Close() = nil after the final flight recorder write failed")
	}
	if again := rec.Close(); again != closeErr {
		t.Errorf("second Close() = %v, want first error %v", again, closeErr)
	}

	var buf bytes.Buffer
	m, err := rec.Snapshot(context.Background(), &buf, "after_failed_stop")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := m.ProviderStatus["go.trace"], "unvalidated: Go execution trace decoder unavailable; raw artifact retained, timing unavailable"; got != want {
		t.Errorf("go.trace status = %q, want %q", got, want)
	}
	b, err := ReadBundle(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b.Files["go.trace"], retained) {
		t.Errorf("go.trace after failed stop: %d bytes, want the %d retained bytes", len(b.Files["go.trace"]), len(retained))
	}
}

// stopErrTracer is a GoTraceRecorder whose Stop fails with err.
type stopErrTracer struct {
	err   error
	stops int
}

func (s *stopErrTracer) Start() error { return nil }
func (s *stopErrTracer) Stop() error  { s.stops++; return s.err }
func (s *stopErrTracer) Snapshot(context.Context) ([]byte, error) {
	return nil, errors.New("no trace")
}

// TestRecorderKeepsFirstTracerStopError checks that a Go tracer stop failure
// becomes the recorder's cleanup error under the first-error rule.
func TestRecorderKeepsFirstTracerStopError(t *testing.T) {
	errStop := errors.New("final trace write failed")
	tracer := &stopErrTracer{err: errStop}
	rec, err := New(Config{
		Window:         time.Second,
		Capacity:       8,
		GoTraceEnabled: true,
		GoTracer:       tracer,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.Start(); err != nil {
		t.Fatal(err)
	}
	rec.Stop()
	rec.SetCleanupErr(errors.New("later cleanup failure"))
	if err := rec.Close(); !errors.Is(err, errStop) {
		t.Errorf("Close() = %v, want %v", err, errStop)
	}
	if tracer.stops != 2 {
		t.Errorf("tracer stops = %d, want 2", tracer.stops)
	}
}

func TestAdaptiveStopReturnsActiveError(t *testing.T) {
	errStop := errors.New("final trace write failed")
	a := &adaptiveTraceRecorder{active: &stopErrTracer{err: errStop}}
	if err := a.Stop(); !errors.Is(err, errStop) {
		t.Errorf("Stop() = %v, want %v", err, errStop)
	}
	if err := (&adaptiveTraceRecorder{}).Stop(); err != nil {
		t.Errorf("Stop() before Start = %v, want nil", err)
	}
}
