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
	// drained records that close has established that the stream finished
	// its work. A retried close does not probe the stream again, since a
	// close that failed may already have destroyed it.
	drained bool
	clock   *cudatrace.Clock
	track   uint64
	prof    *cudaprof.Session
	tracer  *cudatrace.Tracer
	source  string
}

// errUndrained marks a release that could not establish that the timer's
// stream has finished its work. Buffers, modules and the device context
// that the work may use must not be released after it.
var errUndrained = errors.New("stream not drained")

// errTimerClosed is returned by measure once close has drained the timer's
// stream.
var errTimerClosed = errors.New("step timer closed")

// newStepTimer builds a timer on its own stream and calibrates its event clock
// against the host clock. source names the file reported for the recorded frame.
//
// If building fails, newStepTimer releases what it built. If that release
// fails too, it returns the partly built timer along with both errors, and
// the caller must close it.
func newStepTimer(dev *cuda.Device, prof *cudaprof.Session, tracer *cudatrace.Tracer, source string) (*stepTimer, error) {
	st := &stepTimer{prof: prof, tracer: tracer, source: source}
	return st.build(dev.NewStream, dev.NewEvent, st.setup, st.close)
}

// build fills st with a stream from newStream and its anchor, start and end
// events from newEvent, then calls setup. A stream or event returned along
// with an error is kept, so release sees it. On failure build calls release
// and returns st only if release fails, with its error joined after the
// first.
func (st *stepTimer) build(newStream func() (*cuda.Stream, error), newEvent func() (*cuda.Event, error), setup, release func() error) (*stepTimer, error) {
	var err error
	st.stream, err = newStream()
	if err != nil {
		err = fmt.Errorf("new stream: %w", err)
	}
	for _, p := range []**cuda.Event{&st.anchor, &st.start, &st.end} {
		if err != nil {
			break
		}
		if *p, err = newEvent(); err != nil {
			err = fmt.Errorf("new event: %w", err)
		}
	}
	if err == nil {
		err = setup()
	}
	if err == nil {
		return st, nil
	}
	if cerr := release(); cerr != nil {
		return st, errors.Join(err, fmt.Errorf("close step timer: %w", cerr))
	}
	return nil, err
}

// setup calibrates the timer's clock and registers its stream with the
// tracer, if any.
func (st *stepTimer) setup() error {
	clock, err := cudatrace.Calibrate(context.Background(), st.stream, st.anchor)
	if err != nil {
		return err
	}
	st.clock = clock
	if st.tracer != nil {
		st.track, err = st.tracer.RegisterStream(st.stream, "GPU forward pass")
	}
	return err
}

// close waits for the timer's stream to drain, then closes its events and
// the stream, forgetting each one it closes. If the stream does not drain,
// close releases nothing and returns an error matching errUndrained.
// Otherwise it records that the stream drained, attempts every close even
// after one fails, and returns the failures joined; calling close again
// retries only what is still held, without waiting on the stream again.
// Once close has established the drain, measure refuses to enqueue more
// work. Callers must call close before freeing any buffer that work
// enqueued through measure may still use, and must not free those buffers
// if it reports errUndrained.
func (st *stepTimer) close() error {
	return closeStepTimer(&st.stream, &st.drained, []**cuda.Event{&st.anchor, &st.start, &st.end})
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

// closeStepTimer implements stepTimer.close on the stream and events that
// stream and events point to, setting each one it closes to its zero
// value. A zero stream or event, such as a nil pointer, is skipped. It
// waits on the stream only while *drained is false and sets *drained once
// the wait succeeds.
func closeStepTimer[S timerStream, E timerEvent](stream *S, drained *bool, events []*E) error {
	var noStream S
	var noEvent E
	if *stream != noStream && !*drained {
		// Cancellation stops observation, not device work. Keep events and
		// caller-owned buffers alive until the pending stream has drained.
		if err := (*stream).Synchronize(context.Background()); err != nil {
			return fmt.Errorf("synchronize stream: %w: %w", errUndrained, err)
		}
		*drained = true
	}
	var errs []error
	for _, e := range events {
		if *e == noEvent {
			continue
		}
		if err := (*e).Close(); err != nil {
			errs = append(errs, fmt.Errorf("close event: %w", err))
			continue
		}
		*e = noEvent
	}
	if *stream != noStream {
		if err := (*stream).Close(); err != nil {
			errs = append(errs, fmt.Errorf("close stream: %w", err))
		} else {
			*stream = noStream
		}
	}
	return errors.Join(errs...)
}

// teardown collects the release errors of a function's deferred cleanups.
// Once a release reports errUndrained, the cleanups that run after it must
// keep their resources, which work on the undrained stream may still use.
type teardown struct {
	errs []error
	hold bool
}

// add records a release error.
func (td *teardown) add(err error) {
	td.errs = append(td.errs, err)
	if errors.Is(err, errUndrained) {
		td.hold = true
	}
}

// held reports whether a release has reported errUndrained, so the
// remaining cleanups must keep their resources.
func (td *teardown) held() bool { return td.hold }

// errReleasesHeld is joined into a teardown's error when its remaining
// releases were skipped. Unlike errUndrained, which a release that was
// retried successfully can also have reported, it means that resources
// are still in use by undrained work, so the caller must keep the device
// context too.
var errReleasesHeld = errors.New("releases held for undrained work")

// err returns the recorded release errors joined, in the order they were
// added, followed by errReleasesHeld if releases were skipped.
func (td *teardown) err() error {
	if td.hold {
		return errors.Join(append(td.errs, errReleasesHeld)...)
	}
	return errors.Join(td.errs...)
}

// measure runs fn on the timer's stream between two CUDA events and records the
// device interval as a gpu_time sample and a calibrated timeline span. fn must
// only enqueue work: it must not synchronize or consume asynchronous results.
// The interval includes device idle time between host submissions, not only
// active kernel execution. On error, close must drain before buffers are freed.
func (st *stepTimer) measure(ctx context.Context, name string, annotations []perfetto.DebugAnnotation, fn func(*cuda.Stream) error) error {
	if st.drained {
		return errTimerClosed
	}
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
