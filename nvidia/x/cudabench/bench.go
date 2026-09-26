package cudabench

import (
	"fmt"
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
	evict      func(*cuda.Stream) error
	evictName  string
	perIter    bool
}

// spanPerIteration reports whether the run is timed with one event pair per
// iteration rather than one pair around the whole loop. An evictor requires
// that form, because eviction must fall in the gaps between spans; a caller can
// also ask for it alone. Run and limits must agree on the answer, so both ask
// here rather than testing the fields themselves.
func (b *B) spanPerIteration() bool { return b.perIter || b.evict != nil }

// SpanPerIteration times each iteration with its own event pair and reports the
// sum, which is the form an evictor forces. It exists so the two arms of a
// cache-regime comparison can be expressed in one unit: without it the warm arm
// is only reachable in the single-span form, which includes the inter-launch
// gaps the per-iteration sum excludes, and the comment in Run is right that the
// two must not be compared.
//
// The sum is not the cheaper number. Each iteration pays a stop synchronize, so
// this form serializes submission where the single-span form pipelines it. Use
// it when the comparison needs it, not by default.
func (b *B) SpanPerIteration() { b.perIter = true }

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
	defer start.Close()

	stop, err := b.dev.NewEvent()
	if err != nil {
		b.tb.Fatalf("cudabench: create stop event: %v", err)
	}
	defer stop.Close()

	// Warmup run
	fn(b.stream)
	if err := b.stream.Synchronize(nil); err != nil {
		b.tb.Fatalf("cudabench: warmup synchronize: %v", err)
	}

	b.tb.ResetTimer()

	var elapsed time.Duration
	if !b.spanPerIteration() {
		// One event pair around the whole loop. The reported per-iteration time
		// is therefore a mean over N, not a median: a single scheduling hiccup
		// moves it by its full size divided by N.
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
		var err error
		if elapsed, err = stop.ElapsedSince(start); err != nil {
			b.tb.Fatalf("cudabench: elapsed time calculation: %v", err)
		}
	} else {
		// Each iteration gets its own span, so an evictor -- which must not be
		// timed -- runs between them. The sum excludes eviction but also
		// excludes the launch gaps the single-span form includes, so the two
		// forms are not interchangeable and must not be compared to each other.
		for i := 0; i < b.tb.N; i++ {
			if b.evict != nil {
				if err := b.evict(b.stream); err != nil {
					b.tb.Fatalf("cudabench: evict before iteration %d: %v", i, err)
				}
			}
			if err := start.Record(b.stream); err != nil {
				b.tb.Fatalf("cudabench: record start: %v", err)
			}
			fn(b.stream)
			if err := stop.Record(b.stream); err != nil {
				b.tb.Fatalf("cudabench: record stop: %v", err)
			}
			if err := stop.Synchronize(nil); err != nil {
				b.tb.Fatalf("cudabench: stop synchronize: %v", err)
			}
			d, err := stop.ElapsedSince(start)
			if err != nil {
				b.tb.Fatalf("cudabench: elapsed time calculation: %v", err)
			}
			elapsed += d
		}
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
	b.reportLimits()
}

// reportLimits states what the reported numbers do not include, next to the
// numbers.
func (b *B) reportLimits() {
	for _, s := range b.limits() {
		b.tb.Logf("cudabench: %s", s)
	}
}

// limits returns the caveats that apply to this run's reported numbers. Each
// names a term a reader could otherwise take them to cover. It is separate from
// reporting so the set can be asserted without running a benchmark.
func (b *B) limits() []string {
	out := []string{"not included: host dispatch cost; these are device event spans, " +
		"so a workload bound by launch overhead will look faster here than it is"}
	if b.evict == nil {
		out = append(out, "cache state: uncontrolled across iterations, so a working set "+
			"that fits in cache is measured progressively warmer; see B.Evict and EvictL2")
		if b.bytesPerOp > 0 {
			// The declared-bytes caveat only bites when a rate is derived from it.
			out = append(out, "GB/s is SetBytes divided by measured time, so it is the traffic "+
				"the workload requires, not the traffic it performed, and it was measured against "+
				"an uncontrolled cache")
		}
	} else {
		out = append(out, fmt.Sprintf("cache state: %s between iterations, and the eviction is "+
			"outside the timed spans", b.evictName))
	}
	if b.spanPerIteration() {
		out = append(out, "timing is the sum of per-iteration spans, which excludes the "+
			"inter-launch gaps a single span around the loop includes; comparable only to "+
			"other per-iteration-span runs")
	}
	return out
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
	defer stream.Close()

	cb := &B{
		tb:     b,
		dev:    dev,
		stream: stream,
	}

	benchmarkFn(cb)
}
