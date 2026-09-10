//go:build !darwin && !tpu

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"sync"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudatrace"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// cudaTracePath names the Perfetto trace written for CUDA execution. Tracing
// measures device intervals with CUDA events, which drains the stream once per
// traced region, so it is off unless a path is given.
var cudaTracePath = flag.String("cuda-trace", "", "write a Perfetto trace of CUDA submissions and device intervals to this path")

// cudaTracer records CUDA submission regions and measured device intervals in a
// Perfetto trace.
//
// A nil tracer is disabled and every method is then a plain pass-through that
// adds no synchronization, so the forward path can call region unconditionally.
//
// Two tracks are written per traced region. The submission track carries the
// host-side duration of the Go call, which is what cudatrace.Span measures and
// is not GPU execution. The device track carries an interval bracketed by CUDA
// events and converted into the host clock domain by a calibrated
// cudatrace.Clock; that conversion has a stated uncertainty, recorded on every
// device slice so a reader is never invited to treat it as exact.
//
// Tracing serializes traced regions, because the reused event pair describes
// one interval at a time. A run with tracing on therefore has different
// concurrency from the same run with tracing off, and its timings are not
// interchangeable with an untraced run's.
type cudaTracer struct {
	tracer *cudatrace.Tracer
	file   *os.File
	clock  *cudatrace.Clock
	anchor *cuda.Event
	stream *cuda.Stream
	track  uint64

	// mu serializes the event pair, which is reused across regions and can
	// describe only one interval at a time.
	mu    sync.Mutex
	start *cuda.Event
	end   *cuda.Event
}

// newCUDATracer opens path and calibrates the device clock against s.
//
// Calibration drains s, so it must happen before the workload starts.
func newCUDATracer(ctx context.Context, path string, s *cuda.Stream) (_ *cudaTracer, resultErr error) {
	if path == "" {
		return nil, nil
	}
	if s == nil {
		return nil, errors.New("cuda trace: nil stream")
	}
	dev := s.Device()
	if dev == nil {
		return nil, errors.New("cuda trace: stream has no device")
	}

	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("cuda trace: create %s: %w", path, err)
	}
	t := &cudaTracer{file: f, stream: s, tracer: cudatrace.New(f)}
	defer func() {
		if resultErr != nil {
			_ = t.close()
		}
	}()

	if t.anchor, err = dev.NewEvent(); err != nil {
		return nil, fmt.Errorf("cuda trace: anchor event: %w", err)
	}
	if t.start, err = dev.NewEvent(); err != nil {
		return nil, fmt.Errorf("cuda trace: start event: %w", err)
	}
	if t.end, err = dev.NewEvent(); err != nil {
		return nil, fmt.Errorf("cuda trace: end event: %w", err)
	}
	if t.clock, err = cudatrace.Calibrate(ctx, s, t.anchor); err != nil {
		return nil, fmt.Errorf("cuda trace: calibrate: %w", err)
	}
	if _, err := t.tracer.RegisterStream(s, "CUDA submissions"); err != nil {
		return nil, fmt.Errorf("cuda trace: register stream: %w", err)
	}
	if t.track, err = t.tracer.RegisterTimeline("CUDA device", 0); err != nil {
		return nil, fmt.Errorf("cuda trace: register device timeline: %w", err)
	}
	return t, nil
}

// enabled reports whether t records anything.
func (t *cudaTracer) enabled() bool { return t != nil && t.tracer != nil }

// region runs fn, recording its host submission span and the device interval it
// produced.
//
// A disabled tracer calls fn and returns, changing neither the work nor its
// timing. A failed recording is reported alongside fn's own error rather than
// replacing it: a trace that could not be written must not turn a successful
// forward pass into a failed one, and must not be silently dropped either.
func (t *cudaTracer) region(ctx context.Context, name, category string, fn func() error) error {
	if !t.enabled() {
		return fn()
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	if err := t.start.Record(t.stream); err != nil {
		return errors.Join(fn(), fmt.Errorf("cuda trace: record start: %w", err))
	}

	var inner error
	spanErr := t.tracer.Span(t.stream, name, category, func() { inner = fn() })
	if inner != nil {
		return inner
	}
	if spanErr != nil {
		return fmt.Errorf("cuda trace: submission span: %w", spanErr)
	}
	return t.recordDevice(ctx, name, category)
}

// recordDevice converts the bracketed event pair into host-domain timestamps
// and writes one device slice.
func (t *cudaTracer) recordDevice(ctx context.Context, name, category string) error {
	if err := t.end.Record(t.stream); err != nil {
		return fmt.Errorf("cuda trace: record end: %w", err)
	}
	if err := t.end.Synchronize(ctx); err != nil {
		return fmt.Errorf("cuda trace: await device interval: %w", err)
	}
	startNs, err := t.clock.HostNs(t.start)
	if err != nil {
		return fmt.Errorf("cuda trace: start timestamp: %w", err)
	}
	endNs, err := t.clock.HostNs(t.end)
	if err != nil {
		return fmt.Errorf("cuda trace: end timestamp: %w", err)
	}
	if endNs < startNs {
		return fmt.Errorf("cuda trace: device interval ends %d ns before it starts", startNs-endNs)
	}
	return t.tracer.RecordSpan(t.track, startNs, endNs, name, category,
		perfetto.DebugAnnotation{
			Name:     "clock_uncertainty_ns",
			IntValue: t.clock.UncertaintyNs(),
			Type:     perfetto.AnnotationTypeInt,
		})
}

// close destroys the events and closes the trace file. A closed tracer is
// disabled, so regions entered afterwards pass straight through.
func (t *cudaTracer) close() error {
	if t == nil {
		return nil
	}
	var err error
	for _, e := range []*cuda.Event{t.start, t.end, t.anchor} {
		if e != nil {
			err = errors.Join(err, e.Destroy())
		}
	}
	if t.file != nil {
		err = errors.Join(err, t.file.Close())
	}
	t.tracer = nil
	return err
}
