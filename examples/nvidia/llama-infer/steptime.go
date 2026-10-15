package main

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudatrace"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// stepTimer measures one forward pass on the device with CUDA events and
// reports the interval to the pprof profile and the Perfetto timeline.
//
// It reports only what the events measure. Per-kernel and per-instruction
// attribution needs CUPTI PC sampling, which this example does not collect, so
// no such breakdown is synthesized.
type stepTimer struct {
	stream *cuda.Stream
	anchor *cuda.Event
	start  *cuda.Event
	end    *cuda.Event
	clock  *cudatrace.Clock
	track  uint64
	prof   *cudaprof.Session
	tracer *cudatrace.Tracer
	source string
}

// newStepTimer builds a timer on its own stream and calibrates its event clock
// against the host clock. source names the file reported for the recorded frame.
func newStepTimer(dev *cuda.Device, prof *cudaprof.Session, tracer *cudatrace.Tracer, source string) (*stepTimer, error) {
	stream, err := dev.NewStream()
	if err != nil {
		return nil, fmt.Errorf("new stream: %w", err)
	}
	st := &stepTimer{stream: stream, prof: prof, tracer: tracer, source: source}
	// fail releases what was built and reports err, with any release error
	// joined after it.
	fail := func(err error) (*stepTimer, error) {
		if cerr := st.close(); cerr != nil {
			err = errors.Join(err, fmt.Errorf("close step timer: %w", cerr))
		}
		return nil, err
	}
	for _, p := range []**cuda.Event{&st.anchor, &st.start, &st.end} {
		e, err := dev.NewEvent()
		if err != nil {
			return fail(fmt.Errorf("new event: %w", err))
		}
		*p = e
	}
	st.clock, err = cudatrace.Calibrate(context.Background(), stream, st.anchor)
	if err != nil {
		return fail(err)
	}
	if tracer != nil {
		st.track, err = tracer.RegisterStream(stream, "GPU forward pass")
		if err != nil {
			return fail(err)
		}
	}
	return st, nil
}

// close waits for the timer's stream to drain, then closes its events and
// the stream. It attempts every step even after one fails and returns the
// failures joined, in that order. Callers must call it before freeing any
// buffer that work enqueued through measure may still use.
func (st *stepTimer) close() error {
	return st.closeWith(closeStepTimer[*cuda.Stream, *cuda.Event])
}

// closeWith passes the timer's stream and its events, in the order they
// are built, to closer and returns closer's error.
func (st *stepTimer) closeWith(closer func(*cuda.Stream, []*cuda.Event) error) error {
	return closer(st.stream, []*cuda.Event{st.anchor, st.start, st.end})
}

// timerStream is what closeStepTimer needs of a stream.
type timerStream interface {
	comparable
	Synchronize(context.Context) error
	Close() error
}

// timerEvent is what closeStepTimer needs of an event.
type timerEvent interface {
	comparable
	Close() error
}

// closeStepTimer implements stepTimer.close. A zero stream or event, such
// as a nil pointer, is skipped.
func closeStepTimer[S timerStream, E timerEvent](stream S, events []E) error {
	var noStream S
	var noEvent E
	var errs []error
	if stream != noStream {
		// Cancellation stops observation, not device work. Keep events and
		// caller-owned buffers alive until the pending stream has drained.
		if err := stream.Synchronize(context.Background()); err != nil {
			errs = append(errs, fmt.Errorf("synchronize stream: %w", err))
		}
	}
	for _, e := range events {
		if e != noEvent {
			if err := e.Close(); err != nil {
				errs = append(errs, fmt.Errorf("close event: %w", err))
			}
		}
	}
	if stream != noStream {
		if err := stream.Close(); err != nil {
			errs = append(errs, fmt.Errorf("close stream: %w", err))
		}
	}
	return errors.Join(errs...)
}

// measure runs fn on the timer's stream between two CUDA events and records the
// device interval as a gpu_time sample and a calibrated timeline span. fn must
// only enqueue work: it must not synchronize or consume asynchronous results.
// The interval includes device idle time between host submissions, not only
// active kernel execution. On error, close must drain before buffers are freed.
func (st *stepTimer) measure(ctx context.Context, name string, annotations []perfetto.DebugAnnotation, fn func(*cuda.Stream) error) error {
	if err := st.start.Record(st.stream); err != nil {
		return err
	}
	if err := fn(st.stream); err != nil {
		return err
	}
	if err := st.end.Record(st.stream); err != nil {
		return err
	}
	if err := st.stream.Synchronize(ctx); err != nil {
		return err
	}
	d, err := st.end.ElapsedSince(st.start)
	if err != nil {
		return err
	}
	if st.prof != nil {
		st.prof.RecordDetailedSample(ctx, "GPUForwardPass", st.source, 0, 0, cudaprof.SampleValues{
			Samples:  1,
			GPUNanos: int64(d),
		})
	}
	if st.tracer == nil {
		return nil
	}
	startNs, err := st.clock.HostNs(st.start)
	if err != nil {
		return err
	}
	endNs, err := st.clock.HostNs(st.end)
	if err != nil {
		return err
	}
	annotations = append(annotations,
		perfetto.NewIntAnnotation("gpu.nanos", int64(d)),
		perfetto.NewIntAnnotation("clock.uncertainty_ns", st.clock.UncertaintyNs()),
	)
	return st.tracer.RecordSpan(st.track, startNs, endNs, name, "forward_pass", annotations...)
}

// stepAnnotations describes one forward pass on the timeline.
func stepAnnotations(phase string, pos int, tokenID int32, major, minor int) []perfetto.DebugAnnotation {
	return []perfetto.DebugAnnotation{
		perfetto.NewStringAnnotation("phase", phase),
		perfetto.NewIntAnnotation("pos", int64(pos)),
		perfetto.NewIntAnnotation("token_id", int64(tokenID)),
		perfetto.NewStringAnnotation("gpu.arch", fmt.Sprintf("sm_%d%d", major, minor)),
	}
}
