package cudabench

import (
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// B wraps testing.B with GPU event-timing instrumentation and ephemeral memory pools.
type B struct {
	tb         *testing.B
	dev        *cuda.Device
	stream     *cuda.Stream
	pool       *cudamem.Pool
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

	elapsedMs, err := stop.ElapsedSince(start)
	if err != nil {
		b.tb.Fatalf("cudabench: elapsed time calculation: %v", err)
	}

	totalGpuNs := float64(elapsedMs) * 1e6
	gpuNsPerOp := totalGpuNs / float64(b.tb.N)

	b.tb.ReportMetric(gpuNsPerOp, "gpu-ns/op")

	if b.bytesPerOp > 0 {
		sec := float64(elapsedMs) / 1000.0
		totalBytes := float64(b.bytesPerOp) * float64(b.tb.N)
		gbPerSec := (totalBytes / (1024 * 1024 * 1024)) / sec
		b.tb.ReportMetric(gbPerSec, "GB/s")
	}

	if b.flopsPerOp > 0 {
		sec := float64(elapsedMs) / 1000.0
		totalFlops := float64(b.flopsPerOp) * float64(b.tb.N)
		tflops := (totalFlops / 1e12) / sec
		b.tb.ReportMetric(tflops, "TFLOPS")
	}
}

// Run runs a benchmark function using GPU event timing and isolated stream-ordered memory.
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

	pool, _ := cudamem.DefaultPool(dev)

	cb := &B{
		tb:     b,
		dev:    dev,
		stream: stream,
		pool:   pool,
	}

	benchmarkFn(cb)
}
