//go:build darwin

package metalbench

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/apple/x/metal/metalprof"
	"github.com/tmc/lib/apple/x/metal/metalx"
)

// WorkloadPhase describes a segment of dynamic inference traffic.
type WorkloadPhase struct {
	Name      string
	BatchSize int
	SeqLen    int
	Steps     int
}

// PhaseMetrics captures execution telemetry for a phase.
type PhaseMetrics struct {
	PhaseName  string
	BatchSize  int
	TotalTime  time.Duration
	StepTokens int
	Throughput float64 // tok/s
	AvgStepUs  float64 // µs per step
}

// EngineBenchmarkSummary summarizes full-schedule execution.
type EngineBenchmarkSummary struct {
	EngineName      string
	Phases          []PhaseMetrics
	TotalDuration   time.Duration
	TotalTokens     int
	TotalThroughput float64 // aggregate tok/s
}

// StaticBaselineEngine represents a rigid graph engine that uses a fixed, pre-compiled
// kernel layout regardless of dynamic workload shifts.
type StaticBaselineEngine struct {
	dev       *metal.Device
	gemvPSO   apple.MTLComputePipelineState
	dim       int
	outDim    int
	weightBuf metalmem.Slice[float32]
	inBuf     metalmem.Slice[float32]
	outBuf    metalmem.Slice[float32]
}

// NewStaticBaselineEngine creates a static graph engine fixed to a single baseline kernel.
func NewStaticBaselineEngine(dev *metal.Device, dim, outDim int) (*StaticBaselineEngine, error) {
	msl := generateTiledGEMVMSL("Static_Baseline_GEMV", 1, 32)
	pso, err := metalx.Compile(dev, msl, "Static_Baseline_GEMV")
	if err != nil {
		return nil, fmt.Errorf("static engine: compile baseline: %w", err)
	}

	w, _ := metalmem.Alloc[float32](dev, dim*outDim)
	in, _ := metalmem.Alloc[float32](dev, 64*dim) // up to batch=64
	out, _ := metalmem.Alloc[float32](dev, 64*outDim)

	return &StaticBaselineEngine{
		dev:       dev,
		gemvPSO:   pso,
		dim:       dim,
		outDim:    outDim,
		weightBuf: w,
		inBuf:     in,
		outBuf:    out,
	}, nil
}

// ExecutePhase runs a phase using the static kernel. For B>1, it dispatches B sequential passes.
func (e *StaticBaselineEngine) ExecutePhase(phase WorkloadPhase) (PhaseMetrics, error) {
	start := time.Now()
	totalTokens := phase.Steps * phase.BatchSize

	for s := 0; s < phase.Steps; s++ {
		cb := e.dev.Queue().CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(e.gemvPSO)

		// Static engine repeats B single-vector dispatches sequentially
		for b := 0; b < phase.BatchSize; b++ {
			inOffset := uint(b * e.dim * 4)
			outOffset := uint(b * e.outDim * 4)

			enc.SetBufferWithOffsetAtIndex(e.inBuf.Buffer(), inOffset, 0)
			enc.SetBufferWithOffsetAtIndex(e.weightBuf.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(e.outBuf.Buffer(), outOffset, 2)
			enc.DispatchThreadsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(e.outDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: 128, Height: 1, Depth: 1},
			)
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
	}

	dur := time.Since(start)
	return PhaseMetrics{
		PhaseName:  phase.Name,
		BatchSize:  phase.BatchSize,
		TotalTime:  dur,
		StepTokens: totalTokens,
		Throughput: float64(totalTokens) / dur.Seconds(),
		AvgStepUs:  float64(dur.Microseconds()) / float64(phase.Steps),
	}, nil
}

// AdaptivePGOEngine dynamically shifts execution graphs (fused GEMV vs tiled batched GEMM)
// as workload dimensions cross crossover thresholds.
type AdaptivePGOEngine struct {
	dev          *metal.Device
	dim          int
	outDim       int
	activePSO    atomic.Pointer[apple.MTLComputePipelineState]
	psoGEMV      apple.MTLComputePipelineState
	psoBatched   apple.MTLComputePipelineState
	weightBuf    metalmem.Slice[float32]
	inBuf        metalmem.Slice[float32]
	outBuf       metalmem.Slice[float32]
	recorder     *metalprof.FlightRecorder
	mu           sync.Mutex
	currentBatch int
}

const batchedGEMMMSL = `
#include <metal_stdlib>
using namespace metal;

kernel void Tiled_Batched_GEMM(
    device const float* X       [[buffer(0)]],
    device const float* W       [[buffer(1)]],
    device float* Out           [[buffer(2)]],
    constant uint& Dim          [[buffer(3)]],
    constant uint& OutDim       [[buffer(4)]],
    constant uint& BatchSize    [[buffer(5)]],
    uint2 threadgroup_pos       [[threadgroup_position_in_grid]],
    uint2 tid_in_tg             [[thread_position_in_threadgroup]]
) {
    uint batch_idx = threadgroup_pos.y * 16 + tid_in_tg.y;
    uint out_col   = threadgroup_pos.x * 16 + tid_in_tg.x;

    if (batch_idx >= BatchSize || out_col >= OutDim) return;

    float acc = 0.0f;
    device const float* x_row = X + batch_idx * Dim;
    device const float* w_col = W + out_col * Dim;

    for (uint k = 0; k < Dim; k++) {
        acc += x_row[k] * w_col[k];
    }
    Out[batch_idx * OutDim + out_col] = acc;
}
`

// NewAdaptivePGOEngine constructs an adaptive dynamic engine.
func NewAdaptivePGOEngine(dev *metal.Device, dim, outDim int) (*AdaptivePGOEngine, error) {
	// 1. Compile Fused Unrolled GEMV (B=1 optimal)
	gemvMSL := generateTiledGEMVMSL("Adaptive_Fast_GEMV", 4, 128)
	psoGEMV, err := metalx.Compile(dev, gemvMSL, "Adaptive_Fast_GEMV")
	if err != nil {
		return nil, fmt.Errorf("adaptive engine: compile GEMV: %w", err)
	}

	// 2. Compile Tiled Batched GEMM (B>=8 optimal)
	psoBatched, err := metalx.Compile(dev, batchedGEMMMSL, "Tiled_Batched_GEMM")
	if err != nil {
		return nil, fmt.Errorf("adaptive engine: compile GEMM: %w", err)
	}

	w, _ := metalmem.Alloc[float32](dev, dim*outDim)
	in, _ := metalmem.Alloc[float32](dev, 64*dim)
	out, _ := metalmem.Alloc[float32](dev, 64*outDim)

	engine := &AdaptivePGOEngine{
		dev:        dev,
		dim:        dim,
		outDim:     outDim,
		psoGEMV:    psoGEMV,
		psoBatched: psoBatched,
		weightBuf:  w,
		inBuf:      in,
		outBuf:     out,
		recorder:   metalprof.NewFlightRecorder(256),
	}
	engine.activePSO.Store(&engine.psoGEMV)
	return engine, nil
}

// AdaptTopology updates the active execution graph based on incoming batch size.
func (e *AdaptivePGOEngine) AdaptTopology(batchSize int) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if batchSize >= 8 {
		e.activePSO.Store(&e.psoBatched)
		e.currentBatch = batchSize
	} else {
		e.activePSO.Store(&e.psoGEMV)
		e.currentBatch = 1
	}
}

// ExecutePhase runs a workload phase with dynamic graph selection.
func (e *AdaptivePGOEngine) ExecutePhase(phase WorkloadPhase) (PhaseMetrics, error) {
	// Dynamically shift graph topology to match current workload regime
	e.AdaptTopology(phase.BatchSize)

	start := time.Now()
	totalTokens := phase.Steps * phase.BatchSize

	for s := 0; s < phase.Steps; s++ {
		cb := e.dev.Queue().CommandBuffer()
		enc := cb.ComputeCommandEncoder()

		pso := *e.activePSO.Load()
		enc.SetComputePipelineState(pso)

		if phase.BatchSize >= 8 {
			// Fast 2D Grid Batched GEMM
			enc.SetBufferWithOffsetAtIndex(e.inBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(e.weightBuf.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(e.outBuf.Buffer(), 0, 2)

			gridWidth := (e.outDim + 15) / 16
			gridHeight := (phase.BatchSize + 15) / 16
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(gridWidth), Height: uint(gridHeight), Depth: 1},
				apple.MTLSize{Width: 16, Height: 16, Depth: 1},
			)
		} else {
			// Streaming Unrolled GEMV
			for b := 0; b < phase.BatchSize; b++ {
				inOffset := uint(b * e.dim * 4)
				outOffset := uint(b * e.outDim * 4)
				enc.SetBufferWithOffsetAtIndex(e.inBuf.Buffer(), inOffset, 0)
				enc.SetBufferWithOffsetAtIndex(e.weightBuf.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(e.outBuf.Buffer(), outOffset, 2)
				enc.DispatchThreadsThreadsPerThreadgroup(
					apple.MTLSize{Width: uint(e.outDim), Height: 1, Depth: 1},
					apple.MTLSize{Width: 128, Height: 1, Depth: 1},
				)
			}
		}

		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		e.recorder.Record("Adaptive_Step", time.Since(start)/time.Duration(s+1), [3]int{e.outDim, phase.BatchSize, 1}, [3]int{128, 1, 1}, nil)
	}

	dur := time.Since(start)
	return PhaseMetrics{
		PhaseName:  phase.Name,
		BatchSize:  phase.BatchSize,
		TotalTime:  dur,
		StepTokens: totalTokens,
		Throughput: float64(totalTokens) / dur.Seconds(),
		AvgStepUs:  float64(dur.Microseconds()) / float64(phase.Steps),
	}, nil
}
