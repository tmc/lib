package main

import (
	"context"
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
	for _, p := range []**cuda.Event{&st.anchor, &st.start, &st.end} {
		e, err := dev.NewEvent()
		if err != nil {
			st.close()
			return nil, fmt.Errorf("new event: %w", err)
		}
		*p = e
	}
	st.clock, err = cudatrace.Calibrate(context.Background(), stream, st.anchor)
	if err != nil {
		st.close()
		return nil, err
	}
	if tracer != nil {
		st.track, err = tracer.RegisterStream(stream, "GPU forward pass")
		if err != nil {
			st.close()
			return nil, err
		}
	}
	return st, nil
}

func (st *stepTimer) close() {
	if st.stream != nil {
		// Cancellation stops observation, not device work. Keep events and
		// caller-owned buffers alive until the pending stream has drained.
		_ = st.stream.Synchronize(context.Background())
	}
	for _, e := range []*cuda.Event{st.anchor, st.start, st.end} {
		if e != nil {
			_ = e.Destroy()
		}
	}
	if st.stream != nil {
		_ = st.stream.Destroy()
	}
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
