package cudablock_test

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudablock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// reduceKernelSource instantiates block_reduce_sum and block_reduce_max for a
// block of threads threads. Both kernels write from thread 0 only, which is the
// single thread the documented contract says holds the result.
func reduceKernelSource(threads int) string {
	return fmt.Sprintf(`#include "cudablock.cuh"

extern "C" __global__ void reduce_sum(const float* in, float* out) {
    __shared__ float scratch[32];
    float v = in[threadIdx.x];
    float r = block_reduce_sum<float, %d>(v, scratch);
    if (threadIdx.x == 0) *out = r;
}

extern "C" __global__ void reduce_max(const float* in, float* out) {
    __shared__ float scratch[32];
    float v = in[threadIdx.x];
    float r = block_reduce_max<float, %d>(v, scratch);
    if (threadIdx.x == 0) *out = r;
}
`, threads, threads)
}

// runReduction compiles the header for threads threads, launches kernel over
// in, and returns the single value thread 0 wrote.
func runReduction(t *testing.T, dev *cuda.Device, kernel string, in []float32) float32 {
	t.Helper()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatalf("ComputeCapability: %v", err)
	}
	cm, err := nvrtcx.Compile(reduceKernelSource(len(in)),
		nvrtcx.WithArch(int(major), int(minor)),
		nvrtcx.WithHeader("cudablock.cuh", cudablock.Header),
		nvrtcx.WithToolkitIncludes())
	if err != nil {
		t.Skipf("skipping: NVRTC cannot compile cudablock.Header: %v", err)
	}
	raw, err := cm.Load(dev)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	mod, err := cuda.OwnModule(dev, raw)
	if err != nil {
		t.Fatalf("OwnModule: %v", err)
	}
	defer mod.Close()

	fn, err := mod.Function(kernel)
	if err != nil {
		t.Fatalf("Function %s: %v", kernel, err)
	}

	src, err := cudamem.Alloc[float32](dev, len(in))
	if err != nil {
		t.Fatalf("alloc in: %v", err)
	}
	defer src.Free()
	if err := src.CopyFromHost(in); err != nil {
		t.Fatalf("upload: %v", err)
	}
	dst, err := cudamem.Alloc[float32](dev, 1)
	if err != nil {
		t.Fatalf("alloc out: %v", err)
	}
	defer dst.Free()

	srcPtr, dstPtr := src.Ptr(), dst.Ptr()
	if err := fn.Launch([3]uint32{1, 1, 1}, [3]uint32{uint32(len(in)), 1, 1}, 0, 0,
		[]unsafe.Pointer{unsafe.Pointer(&srcPtr), unsafe.Pointer(&dstPtr)}); err != nil {
		t.Fatalf("Launch %s: %v", kernel, err)
	}
	got := make([]float32, 1)
	if err := dst.CopyToHost(got); err != nil {
		t.Fatalf("download: %v", err)
	}
	return got[0]
}

// blockReduceCases covers block shapes the two-stage reduction treats
// differently: a single warp, a partial single warp, exact multiples of the
// warp size, a non-power-of-two block, and blocks whose thread count is not a
// multiple of 32 so the final warp is partial.
var blockReduceCases = []struct {
	name    string
	threads int
}{
	{"one_warp", 32},
	{"sub_warp", 20},
	{"two_warps", 64},
	{"non_power_of_two", 96},
	{"partial_final_warp", 100},
	{"partial_final_warp_odd", 77},
	{"full_block", 256},
	{"max_block", 1024},
}

func TestBlockReduceMax(t *testing.T) {
	dev := cudatest.Require(t)
	for _, tt := range blockReduceCases {
		t.Run(tt.name, func(t *testing.T) {
			// Values are distinct and the maximum sits at an interior index,
			// so a reduction that silently dropped the final partial warp, or
			// folded in a lane that holds no value, would not match.
			in := make([]float32, tt.threads)
			for i := range in {
				in[i] = float32((i*37)%tt.threads) - float32(tt.threads)/2
			}
			want := in[0]
			for _, v := range in {
				want = float32(math.Max(float64(want), float64(v)))
			}
			if got := runReduction(t, dev, "reduce_max", in); got != want {
				t.Errorf("block_reduce_max over %d threads = %v, want %v", tt.threads, got, want)
			}
		})
	}
}

// TestBlockReduceMaxAllNegative pins the case an identity of zero would get
// wrong: every input is below zero, so a maximum of 0 is a bug, not a rounding
// difference.
func TestBlockReduceMaxAllNegative(t *testing.T) {
	dev := cudatest.Require(t)
	for _, tt := range blockReduceCases {
		t.Run(tt.name, func(t *testing.T) {
			in := make([]float32, tt.threads)
			for i := range in {
				in[i] = -1 - float32((i*13)%tt.threads)
			}
			want := in[0]
			for _, v := range in {
				want = float32(math.Max(float64(want), float64(v)))
			}
			if want >= 0 {
				t.Fatalf("test setup: want %v is not negative", want)
			}
			if got := runReduction(t, dev, "reduce_max", in); got != want {
				t.Errorf("block_reduce_max over %d negative threads = %v, want %v", tt.threads, got, want)
			}
		})
	}
}

// TestBlockReduceSum covers the same block shapes as the max twin, because
// block_reduce_sum shares its partial-final-warp handling.
func TestBlockReduceSum(t *testing.T) {
	dev := cudatest.Require(t)
	for _, tt := range blockReduceCases {
		t.Run(tt.name, func(t *testing.T) {
			// Exact in float32: every value is a small integer and the total
			// stays well inside the 24-bit mantissa, so any difference from
			// the reference is a real reduction error.
			in := make([]float32, tt.threads)
			for i := range in {
				in[i] = float32(i%7) - 3
			}
			var want float32
			for _, v := range in {
				want += v
			}
			if got := runReduction(t, dev, "reduce_sum", in); got != want {
				t.Errorf("block_reduce_sum over %d threads = %v, want %v", tt.threads, got, want)
			}
		})
	}
}
