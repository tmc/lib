//go:build darwin

package metalbench

import (
	"testing"
	"time"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func BenchmarkAutoTunedDecoding(b *testing.B) {
	dev, err := metal.NewDevice()
	if err != nil {
		b.Skipf("Metal device unavailable: %v", err)
	}

	const dim = 2048
	const outDim = 2048

	xBuf, _ := metalmem.Alloc[float32](dev, dim)
	wBuf, _ := metalmem.Alloc[float32](dev, dim*outDim)
	outBuf, _ := metalmem.Alloc[float32](dev, outDim)
	defer xBuf.Free()
	defer wBuf.Free()
	defer outBuf.Free()
	for i := range xBuf.HostSlice() {
		xBuf.HostSlice()[i] = 1
	}
	for i := range wBuf.HostSlice() {
		wBuf.HostSlice()[i] = 1
	}
	for i := range outBuf.HostSlice() {
		outBuf.HostSlice()[i] = -1
	}

	dimVal := uint32(dim)
	outDimVal := uint32(outDim)
	dimBuf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&dimVal), 4, apple.MTLResourceStorageModeShared)
	outDimBuf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&outDimVal), 4, apple.MTLResourceStorageModeShared)

	// 1. Define competing candidate tile strategies from E-Graph
	candidates := []Candidate{
		{Name: "GEMV_Unroll_1", MSL: generateTiledGEMVMSL("GEMV_Unroll_1", 1)},
		{Name: "GEMV_Unroll_2", MSL: generateTiledGEMVMSL("GEMV_Unroll_2", 2)},
		{Name: "GEMV_Unroll_4", MSL: generateTiledGEMVMSL("GEMV_Unroll_4", 4)},
		{Name: "GEMV_Unroll_8", MSL: generateTiledGEMVMSL("GEMV_Unroll_8", 8)},
	}

	harness := New(dev)

	// 2. Run metalbench empirical search across candidate kernels
	dispatchFn := func(pso apple.MTLComputePipelineState, cb apple.MTLCommandBuffer) {
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(xBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(wBuf.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(outBuf.Buffer(), 0, 2)
		enc.SetBufferWithOffsetAtIndex(dimBuf, 0, 3)
		enc.SetBufferWithOffsetAtIndex(outDimBuf, 0, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			apple.MTLSize{Width: outDim, Height: 1, Depth: 1},
			apple.MTLSize{Width: 128, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
	}

	bestRes, err := harness.AutoTune(candidates, 10, dispatchFn)
	if err != nil {
		b.Fatalf("AutoTune failed: %v", err)
	}

	for i, got := range outBuf.HostSlice() {
		if got != dim {
			b.Fatalf("output[%d] = %g, want %d", i, got, dim)
		}
	}

	// 3. Compile Baseline vs AutoTuned PSO
	baselineLib, _ := dev.CompileLibrary(candidates[0].MSL)
	baselinePSO, _ := dev.NewComputePipeline(baselineLib, candidates[0].Name)

	bestLib, _ := dev.CompileLibrary(bestRes.MSL)
	bestPSO, _ := dev.NewComputePipeline(bestLib, bestRes.KernelName)

	b.Logf("=== metalbench E-Graph Autotuner Search Results ===")
	b.Logf("Selected Best Tile Kernel: %s", bestRes.KernelName)
	b.Logf("Empirical Search Latency:   %v", bestRes.Duration)

	b.Run("Baseline_Unoptimized_GEMV", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			cb := dev.Queue().CommandBuffer()
			dispatchFn(baselinePSO, cb)
			cb.Commit()
			cb.WaitUntilCompleted()
		}
		dur := time.Since(start)
		if dur > 0 {
			tokPerSec := float64(b.N) / dur.Seconds()
			b.ReportMetric(tokPerSec, "tok/s")
		}
	})

	b.Run("AutoTuned_EGraph_GEMV", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			cb := dev.Queue().CommandBuffer()
			dispatchFn(bestPSO, cb)
			cb.Commit()
			cb.WaitUntilCompleted()
		}
		dur := time.Since(start)
		if dur > 0 {
			tokPerSec := float64(b.N) / dur.Seconds()
			b.ReportMetric(tokPerSec, "tok/s")
		}
	})
}
