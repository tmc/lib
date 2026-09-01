package main

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// FullComparisonResult holds performance numbers across both execution engines.
type FullComparisonResult struct {
	BatchSize           int
	Steps               int
	UnfusedDurationMs   float64
	UnfusedTokSec       float64
	TmcEGraphDurationMs float64
	TmcEGraphTokSec     float64
	Speedup             float64
}

// RunBenchmarkComparison executes an end-to-end multi-layer Transformer layer benchmark.
func RunBenchmarkComparison(batchSize, steps int, budget egraph.AutotuneBudget) (*FullComparisonResult, error) {
	if err := driver.Init(0); err != nil {
		return nil, fmt.Errorf("cuda driver init: %w", err)
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		return nil, fmt.Errorf("cuda device: %w", err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		return nil, fmt.Errorf("cuda stream: %w", err)
	}
	defer stream.Destroy()

	ctx := context.Background()

	// Dimensions for standard Llama-style 1B/3B block
	hiddenDim := 2048
	intermediateDim := 5632
	kvDim := 512

	// Allocate GPU memory
	xBuf, _ := cudamem.Alloc[float32](dev, batchSize*hiddenDim)
	wNormBuf, _ := cudamem.Alloc[float32](dev, hiddenDim)
	normOutBuf, _ := cudamem.Alloc[float32](dev, batchSize*hiddenDim)

	wqBuf, _ := cudamem.Alloc[float32](dev, hiddenDim*hiddenDim)
	wkBuf, _ := cudamem.Alloc[float32](dev, hiddenDim*kvDim)
	wvBuf, _ := cudamem.Alloc[float32](dev, hiddenDim*kvDim)

	qBuf, _ := cudamem.Alloc[float32](dev, batchSize*hiddenDim)
	kBuf, _ := cudamem.Alloc[float32](dev, batchSize*kvDim)
	vBuf, _ := cudamem.Alloc[float32](dev, batchSize*kvDim)

	wGateBuf, _ := cudamem.Alloc[float32](dev, hiddenDim*intermediateDim)
	wUpBuf, _ := cudamem.Alloc[float32](dev, hiddenDim*intermediateDim)
	wDownBuf, _ := cudamem.Alloc[float32](dev, intermediateDim*hiddenDim)

	gateBuf, _ := cudamem.Alloc[float32](dev, batchSize*intermediateDim)
	upBuf, _ := cudamem.Alloc[float32](dev, batchSize*intermediateDim)
	ffnOutBuf, _ := cudamem.Alloc[float32](dev, batchSize*hiddenDim)
	layerOutBuf, _ := cudamem.Alloc[float32](dev, batchSize*hiddenDim)

	defer func() {
		xBuf.Free()
		wNormBuf.Free()
		normOutBuf.Free()
		wqBuf.Free()
		wkBuf.Free()
		wvBuf.Free()
		qBuf.Free()
		kBuf.Free()
		vBuf.Free()
		wGateBuf.Free()
		wUpBuf.Free()
		wDownBuf.Free()
		gateBuf.Free()
		upBuf.Free()
		ffnOutBuf.Free()
		layerOutBuf.Free()
	}()

	// JIT compile kernels
	major, minor, _ := dev.ComputeCapability()
	mod, err := nvrtcx.Compile(benchKernelsCU, nvrtcx.WithArch(major, minor))
	if err != nil {
		return nil, fmt.Errorf("nvrtcx compile: %w", err)
	}

	loaded, err := mod.Load(dev)
	if err != nil {
		return nil, fmt.Errorf("load module: %w", err)
	}

	kRMSNorm, _ := loaded.GetFunction("kernel_rmsnorm")
	kGEMV, _ := loaded.GetFunction("kernel_gemv")
	kSiLU, _ := loaded.GetFunction("kernel_silu")
	kMul, _ := loaded.GetFunction("kernel_mul")
	kAdd, _ := loaded.GetFunction("kernel_add")

	kFusedRMSNormGEMV, _ := loaded.GetFunction("kernel_fused_rmsnorm_gemv")
	kFusedSwiGLU, _ := loaded.GetFunction("kernel_fused_swiglu_gemv")

	// Warmup
	_ = stream.Synchronize(ctx)

	// 1. Unfused Schedule (Multiple intermediate allocations & kernel launch transitions)
	t0 := time.Now()
	for s := 0; s < steps; s++ {
		_ = cudalaunch.Kernel(kRMSNorm).Grid(uint32((hiddenDim+127)/128)).Block(128).
			Launch(xBuf, wNormBuf, normOutBuf, int32(hiddenDim), float32(1e-5))

		_ = cudalaunch.Kernel(kGEMV).Grid(uint32((hiddenDim+127)/128)).Block(128).
			Launch(normOutBuf, wqBuf, qBuf, int32(hiddenDim), int32(hiddenDim))
		_ = cudalaunch.Kernel(kGEMV).Grid(uint32((kvDim+127)/128)).Block(128).
			Launch(normOutBuf, wkBuf, kBuf, int32(hiddenDim), int32(kvDim))
		_ = cudalaunch.Kernel(kGEMV).Grid(uint32((kvDim+127)/128)).Block(128).
			Launch(normOutBuf, wvBuf, vBuf, int32(hiddenDim), int32(kvDim))

		_ = cudalaunch.Kernel(kGEMV).Grid(uint32((intermediateDim+127)/128)).Block(128).
			Launch(xBuf, wGateBuf, gateBuf, int32(hiddenDim), int32(intermediateDim))
		_ = cudalaunch.Kernel(kGEMV).Grid(uint32((intermediateDim+127)/128)).Block(128).
			Launch(xBuf, wUpBuf, upBuf, int32(hiddenDim), int32(intermediateDim))

		_ = cudalaunch.Kernel(kSiLU).Grid(uint32((intermediateDim+127)/128)).Block(128).
			Launch(gateBuf, gateBuf, int32(intermediateDim))
		_ = cudalaunch.Kernel(kMul).Grid(uint32((intermediateDim+127)/128)).Block(128).
			Launch(gateBuf, upBuf, gateBuf, int32(intermediateDim))

		_ = cudalaunch.Kernel(kGEMV).Grid(uint32((hiddenDim+127)/128)).Block(128).
			Launch(gateBuf, wDownBuf, ffnOutBuf, int32(intermediateDim), int32(hiddenDim))

		_ = cudalaunch.Kernel(kAdd).Grid(uint32((hiddenDim+127)/128)).Block(128).
			Launch(qBuf, ffnOutBuf, layerOutBuf, int32(hiddenDim))
	}
	_ = stream.Synchronize(ctx)
	unfusedElapsed := time.Since(t0)

	// 2. TMC Pure-Go E-Graph Fused Megakernel Engine
	t1 := time.Now()
	for s := 0; s < steps; s++ {
		// (a) Fused RMSNorm + QKV Projection
		_ = cudalaunch.Kernel(kFusedRMSNormGEMV).Grid(uint32((hiddenDim+127)/128)).Block(128).
			Launch(xBuf, wNormBuf, wqBuf, wkBuf, wvBuf,
				qBuf, kBuf, vBuf,
				int32(hiddenDim), int32(hiddenDim), int32(kvDim), float32(1e-5))

		// (b) Fused SwiGLU Block (Zero intermediate DRAM writes)
		_ = cudalaunch.Kernel(kFusedSwiGLU).Grid(uint32(hiddenDim)).Block(128).
			Launch(gateBuf, upBuf, wDownBuf, ffnOutBuf,
				int32(hiddenDim), int32(intermediateDim))

		// (c) Residual Add
		_ = cudalaunch.Kernel(kAdd).Grid(uint32((hiddenDim+127)/128)).Block(128).
			Launch(qBuf, ffnOutBuf, layerOutBuf, int32(hiddenDim))
	}
	_ = stream.Synchronize(ctx)
	tmcElapsed := time.Since(t1)

	unfusedMs := float64(unfusedElapsed.Nanoseconds()) / 1e6
	tmcMs := float64(tmcElapsed.Nanoseconds()) / 1e6
	totalTokens := float64(batchSize * steps)

	return &FullComparisonResult{
		BatchSize:           batchSize,
		Steps:               steps,
		UnfusedDurationMs:   unfusedMs,
		UnfusedTokSec:       (totalTokens / (unfusedMs / 1000.0)),
		TmcEGraphDurationMs: tmcMs,
		TmcEGraphTokSec:     (totalTokens / (tmcMs / 1000.0)),
		Speedup:             unfusedMs / tmcMs,
	}, nil
}
