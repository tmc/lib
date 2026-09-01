//go:build darwin

package metalbench

import (
	"fmt"
	"runtime"
	"time"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
)

// BenchmarkResult records GPU execution metrics for an empirical search run.
type BenchmarkResult struct {
	KernelName  string
	MSL         string
	Duration    time.Duration
	OpsPerSec   float64
	BandwidthGB float64
}

// Harness manages hardware timing on Apple Metal command queues.
type Harness struct {
	dev   *metal.Device
	queue apple.MTLCommandQueue
}

// New creates a new Metal benchmark harness.
func New(dev *metal.Device) *Harness {
	return &Harness{
		dev:   dev,
		queue: dev.Queue(),
	}
}

// Time measures the GPU hardware execution latency of fn over iterations runs.
func (h *Harness) Time(iterations int, fn func(cb apple.MTLCommandBuffer)) time.Duration {
	if iterations <= 0 {
		iterations = 1
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Warmup
	cbWarm := h.queue.CommandBuffer()
	fn(cbWarm)
	cbWarm.Commit()
	cbWarm.WaitUntilCompleted()

	start := time.Now()
	for i := 0; i < iterations; i++ {
		cb := h.queue.CommandBuffer()
		fn(cb)
		cb.Commit()
		cb.WaitUntilCompleted()
	}
	total := time.Since(start)

	return total / time.Duration(iterations)
}

// Candidate defines a candidate compute shader kernel implementation.
type Candidate struct {
	Name string
	MSL  string
}

// AutoTune searches candidate MSL pipelines and selects the kernel with the lowest empirical execution latency.
func (h *Harness) AutoTune(candidates []Candidate, runs int, dispatch func(pso apple.MTLComputePipelineState, cb apple.MTLCommandBuffer)) (*BenchmarkResult, error) {
	if len(candidates) == 0 {
		return nil, fmt.Errorf("metalbench: no candidate kernels provided")
	}

	var best Candidate
	var bestDur time.Duration = time.Duration(1<<63 - 1)

	for _, cand := range candidates {
		lib, err := h.dev.CompileLibrary(cand.MSL)
		if err != nil {
			continue
		}
		pso, err := h.dev.NewComputePipeline(lib, cand.Name)
		if err != nil {
			continue
		}

		dur := h.Time(runs, func(cb apple.MTLCommandBuffer) {
			dispatch(pso, cb)
		})

		if dur < bestDur {
			bestDur = dur
			best = cand
		}
	}

	if best.Name == "" {
		return nil, fmt.Errorf("metalbench: all autotuning candidates failed to compile or run")
	}

	opsPerSec := 0.0
	if bestDur > 0 {
		opsPerSec = 1.0 / bestDur.Seconds()
	}

	return &BenchmarkResult{
		KernelName: best.Name,
		MSL:        best.MSL,
		Duration:   bestDur,
		OpsPerSec:  opsPerSec,
	}, nil
}
