package cudabench_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudabench"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// BenchmarkHarnessOverhead measures the harness itself, not a workload: the
// timed closure is empty, so the reported gpu-ns/op is the per-iteration cost
// of dispatching through Time on an idle stream. That is the floor beneath any
// benchmark built on this package, and the run exercises the whole plumbing
// path — stream creation, event record, synchronize, ElapsedSince — which
// fails the benchmark if any of it errors.
//
// It deliberately sets no bytes and no flops. An empty closure moves no memory
// and performs no arithmetic, so a GB/s or TFLOPS column here would be a rate
// invented from a size that nothing consumed.
func BenchmarkHarnessOverhead(b *testing.B) {
	dev := cudatest.Require(b)
	cudabench.Run(b, dev, func(cb *cudabench.B) {
		cb.Time(func(s *cuda.Stream) {})
	})
}
