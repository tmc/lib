package cudabench

import (
	"runtime"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// B wraps testing.B with GPU event-timing instrumentation.
type B struct {
	tb         *testing.B
	dev        *cuda.Device
	stream     *cuda.Stream
	bytesPerOp int64
	flopsPerOp int64
}

// SetBytes configures the number of bytes processed per benchmark iteration.
func (b *B) SetBytes(n int64) {
	b.bytesPerOp = n
}

// SetFlops configures the number of floating-point operations performed per benchmark iteration.
func (b *B) SetFlops(n int64) {
	b.flopsPerOp = n
}

// Stream returns the active CUDA stream for this benchmark.
func (b *B) Stream() *cuda.Stream {
	return b.stream
}

// Device returns the active CUDA device.
func (b *B) Device() *cuda.Device {
	return b.dev
}

// AllocAsync allocates a temporary typed slice from the stream-ordered pool without global allocator lock contention.
func AllocAsync[T any](b *B, length int) (cudamem.Slice[T], error) {
	return cudamem.AllocAsync[T](b.stream, length)
}

// Time measures the GPU hardware execution time of fn over b.tb.N iterations using CUDA events.
func (b *B) Time(fn func(stream *cuda.Stream)) {
	if b.tb.N == 0 {
		return
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	start, err := b.dev.NewEvent()
	if err != nil {
		b.tb.Fatalf("cudabench: create start event: %v", err)
	}
	defer start.Destroy()

	stop, err := b.dev.NewEvent()
	if err != nil {
		b.tb.Fatalf("cudabench: create stop event: %v", err)
	}
	defer stop.Destroy()

	// Warmup run
	fn(b.stream)
	if err := b.stream.Synchronize(nil); err != nil {
		b.tb.Fatalf("cudabench: warmup synchronize: %v", err)
	}

	b.tb.ResetTimer()

	if err := start.Record(b.stream); err != nil {
		b.tb.Fatalf("cudabench: record start: %v", err)
	}

	for i := 0; i < b.tb.N; i++ {
		fn(b.stream)
	}

	if err := stop.Record(b.stream); err != nil {
		b.tb.Fatalf("cudabench: record stop: %v", err)
	}

	if err := stop.Synchronize(nil); err != nil {
		b.tb.Fatalf("cudabench: stop synchronize: %v", err)
	}

	elapsed, err := stop.ElapsedSince(start)
	if err != nil {
		b.tb.Fatalf("cudabench: elapsed time calculation: %v", err)
	}

	m, ok := computeMetrics(elapsed, b.tb.N, b.bytesPerOp, b.flopsPerOp)
	if !ok {
		// The device clock did not resolve the run. Reporting rates here would
		// divide by zero and publish an infinite throughput, so report nothing
		// and say why: an absent column is visible, a wrong number is not.
		b.tb.Logf("cudabench: device elapsed time over %d iterations was %v; no rate is derivable", b.tb.N, elapsed)
		return
	}

	b.tb.ReportMetric(m.gpuNsPerOp, "gpu-ns/op")
	if b.bytesPerOp > 0 {
		b.tb.ReportMetric(m.gbPerSec, "GB/s")
	}
	if b.flopsPerOp > 0 {
		b.tb.ReportMetric(m.tflops, "TFLOPS")
	}
}

// metrics holds the rates derived from one event-timed run.
type metrics struct {
	gpuNsPerOp float64
	gbPerSec   float64 // decimal GB/s; zero unless bytesPerOp is positive
	tflops     float64 // zero unless flopsPerOp is positive
}

// computeMetrics derives per-operation rates from the device time elapsed over n
// iterations. Throughput uses decimal units (1 GB is 1e9 bytes), matching the units
// vendors and memory-bandwidth ceilings are quoted in.
//
// It reports false when n or elapsed is non-positive: no rate can be derived from a
// run the device clock did not resolve.
func computeMetrics(elapsed time.Duration, n int, bytesPerOp, flopsPerOp int64) (metrics, bool) {
	if n <= 0 || elapsed <= 0 {
		return metrics{}, false
	}
	sec := elapsed.Seconds()
	m := metrics{gpuNsPerOp: float64(elapsed.Nanoseconds()) / float64(n)}
	if bytesPerOp > 0 {
		m.gbPerSec = (float64(bytesPerOp) * float64(n) / 1e9) / sec
	}
	if flopsPerOp > 0 {
		m.tflops = (float64(flopsPerOp) * float64(n) / 1e12) / sec
	}
	return m, true
}

// Run runs a benchmark function using GPU event timing on a dedicated stream.
func Run(b *testing.B, dev *cuda.Device, benchmarkFn func(cb *B)) {
	if dev == nil {
		b.Skip("cudabench: nil device")
		return
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	stream, err := dev.NewStream()
	if err != nil {
		b.Fatalf("cudabench: create stream: %v", err)
	}
	defer stream.Destroy()

	cb := &B{
		tb:     b,
		dev:    dev,
		stream: stream,
	}

	benchmarkFn(cb)
}
