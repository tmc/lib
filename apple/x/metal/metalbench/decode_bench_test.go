//go:build darwin

package metalbench

import (
	"fmt"
	"testing"
	"time"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/mempool"
)

// Generates GEMV MSL shaders parameterized by unroll factor and tile size.
func generateTiledGEMVMSL(kernelName string, unrollFactor int, tileWidth int) string {
	return fmt.Sprintf(`
#include <metal_stdlib>
using namespace metal;

inline float simd_sum(float val) {
    #pragma unroll
    for (ushort offset = 16; offset > 0; offset /= 2) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

kernel void %s(
    device const float* x      [[buffer(0)]],
    device const float* W      [[buffer(1)]],
    device float* out          [[buffer(2)]],
    constant uint& dim         [[buffer(3)]],
    constant uint& out_dim     [[buffer(4)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint tid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]]
) {
    if (row >= out_dim) return;

    device const float* w_row = W + row * dim;
    float acc = 0.0f;

    uint i = tid * %d;
    for (; i + %d <= dim; i += 128 * %d) {
        #pragma unroll
        for (int k = 0; k < %d; k++) {
            acc += x[i + k] * w_row[i + k];
        }
    }
    for (; i < dim; i++) {
        acc += x[i] * w_row[i];
    }

    acc = simd_sum(acc);
    threadgroup float s_sum[4];
    if (simd_lane == 0) s_sum[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
    if (simd_id == 0) block_sum = simd_sum(block_sum);
    if (tid == 0) out[row] = block_sum;
}
`, kernelName, unrollFactor, unrollFactor, unrollFactor, unrollFactor)
}

func BenchmarkAutoTunedDecoding(b *testing.B) {
	dev, err := metal.NewDevice()
	if err != nil {
		b.Skipf("Metal device unavailable: %v", err)
	}

	const dim = 2048
	const outDim = 2048

	xBuf, _ := mempool.Alloc[float32](dev, dim)
	wBuf, _ := mempool.Alloc[float32](dev, dim*outDim)
	outBuf, _ := mempool.Alloc[float32](dev, outDim)

	dimVal := uint32(dim)
	outDimVal := uint32(outDim)
	dimBuf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&dimVal), 4, apple.MTLResourceStorageModeShared)
	outDimBuf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&outDimVal), 4, apple.MTLResourceStorageModeShared)

	// 1. Define competing candidate tile strategies from E-Graph
	candidates := []Candidate{
		{Name: "GEMV_Unroll_1", MSL: generateTiledGEMVMSL("GEMV_Unroll_1", 1, 32)},
		{Name: "GEMV_Unroll_2", MSL: generateTiledGEMVMSL("GEMV_Unroll_2", 2, 64)},
		{Name: "GEMV_Unroll_4", MSL: generateTiledGEMVMSL("GEMV_Unroll_4", 4, 128)},
		{Name: "GEMV_Unroll_8", MSL: generateTiledGEMVMSL("GEMV_Unroll_8", 8, 256)},
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
		enc.DispatchThreadsThreadsPerThreadgroup(
			apple.MTLSize{Width: outDim, Height: 1, Depth: 1},
			apple.MTLSize{Width: 128, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
	}

	bestRes, err := harness.AutoTune(candidates, 10, dispatchFn)
	if err != nil {
		b.Fatalf("AutoTune failed: %v", err)
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
