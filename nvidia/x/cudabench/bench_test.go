package cudabench_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudabench"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestRun(t *testing.T) {
	// Verifies compilation and basic package entry points
}

func BenchmarkMock(b *testing.B) {

	dev := cudatest.Require(b)
	cudabench.Run(b, dev, func(cb *cudabench.B) {
		cb.SetBytes(1024)
		cb.SetFlops(2048)
		cb.Time(func(s *cuda.Stream) {
			// Benchmark workload executed on stream
		})
	})
}
