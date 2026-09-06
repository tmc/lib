package nvrtcx_test

import (
	"context"
	"crypto/sha256"
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// The partition/load/mma/store syntax follows NVIDIA's CUDA 13.3 example:
// https://docs.nvidia.com/cuda/nvrtc/index.html#example-tile-compilation-cuda-13-3
// One output tile and two reduction tiles exercise accumulation without padding.
const tileMatrixSource = `#include "cuda_tile.h"
using namespace cuda::tiles;
template<int M, int N, int K, int TM, int TN, int TK>
__tile_global__ void matrix(float* A, float* B, float* C) {
 auto a = partition_view{tensor_span{assume_aligned<16>(A), extents{M, K}}, shape<TM, TK>{}};
 auto b = partition_view{tensor_span{assume_aligned<16>(B), extents{K, N}}, shape<TK, TN>{}};
 auto c = partition_view{tensor_span{assume_aligned<16>(C), extents{M, N}}, shape<TM, TN>{}};
 auto sum = full<tile<float, shape<TM, TN>>>(0.0f);
 for (auto k : irange(0, K / TK)) {
  sum = mma(a.load(0, k), b.load(k, 0), sum);
 }
 c.store(sum, 0, 0);
}
extern "C" __global__ void transform(float* C) {
 unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i >= 256) return;
 C[i] = 2.0f * C[i] + 1.0f;
}`

const tileMatrixName = "matrix<16,16,32,16,16,16>"

// Keep the standalone numerical gate independent of template name retrieval.
// TestTileOnlyTemplateLoweredName covers that capability without a fallback.
const tileMatrixStandaloneSource = `#include "cuda_tile.h"
using namespace cuda::tiles;
extern "C" __tile_global__ void matrix(float* A, float* B, float* C) {
 constexpr int M = 16, N = 16, K = 32, TM = 16, TN = 16, TK = 16;
 auto a = partition_view{tensor_span{assume_aligned<16>(A), extents{M, K}}, shape<TM, TK>{}};
 auto b = partition_view{tensor_span{assume_aligned<16>(B), extents{K, N}}, shape<TK, TN>{}};
 auto c = partition_view{tensor_span{assume_aligned<16>(C), extents{M, N}}, shape<TM, TN>{}};
 auto sum = full<tile<float, shape<TM, TN>>>(0.0f);
 for (auto k : irange(0, K / TK)) {
  sum = mma(a.load(0, k), b.load(k, 0), sum);
 }
 c.store(sum, 0, 0);
}`

func compileTileMatrix(t *testing.T, major, minor int, mixed bool) *nvrtcx.CompiledModule {
	t.Helper()
	output := nvrtcx.OutputTileIR
	source := tileMatrixStandaloneSource
	expression := ""
	if mixed {
		output = nvrtcx.OutputPTXTileIR
		source = tileMatrixSource
		expression = tileMatrixName
	}
	t.Logf("source=%s\narch=compute_%d%d std=c++20 output=%v expression=%q cache=disabled", source, major, minor, output, expression)
	opts := []nvrtcx.Option{nvrtcx.WithArch(major, minor),
		nvrtcx.WithFlag("--std=c++20"), nvrtcx.WithOutput(output),
		nvrtcx.WithNoCache()}
	if mixed {
		opts = append(opts, nvrtcx.WithNameExpressions(expression))
	}
	cm, err := nvrtcx.Compile(source, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if len(cm.TileIR()) == 0 || (mixed && cm.PTX() == "") {
		t.Fatal("missing requested matrix artifact")
	}
	if mixed {
		if name, ok := cm.LoweredName(tileMatrixName); !ok || name == "" {
			t.Fatal("missing lowered matrix name")
		} else {
			t.Logf("lowered name=%s", name)
		}
	}
	t.Logf("compiler log=%q TileIR bytes=%d sha256=%x PTX bytes=%d sha256=%x", cm.Log(), len(cm.TileIR()), sha256.Sum256(cm.TileIR()), len(cm.PTX()), sha256.Sum256([]byte(cm.PTX())))
	return cm
}

// This gate compiles the numerical fixture without requiring a CUDA context.
// It does not establish that either artifact loads or executes.
func TestTileMatrixCompilation(t *testing.T) {
	if os.Getenv("CUDA_TILE_TEST") != "1" {
		t.Skip("CUDA_TILE_TEST=1 requires NVRTC supporting Tile compilation")
	}
	t.Run("standalone", func(t *testing.T) { compileTileMatrix(t, 12, 1, false) })
	t.Run("mixed", func(t *testing.T) { compileTileMatrix(t, 12, 1, true) })
}

// NVRTC 13.3 returned error 9 when retrieving the name with --tile-only on
// the qualified GB10 host. This strict gate retains that unresolved failure;
// neither mixed compilation nor an extern-C entry substitutes for this check.
func TestTileOnlyTemplateLoweredName(t *testing.T) {
	if os.Getenv("CUDA_TILE_TEST") != "1" {
		t.Skip("CUDA_TILE_TEST=1 requires NVRTC supporting Tile template names")
	}
	t.Logf("source=%s\narch=compute_121 std=c++20 output=TileIR expression=%s cache=disabled", tileMatrixSource, tileMatrixName)
	cm, err := nvrtcx.Compile(tileMatrixSource, nvrtcx.WithArch(12, 1),
		nvrtcx.WithFlag("--std=c++20"), nvrtcx.WithOutput(nvrtcx.OutputTileIR),
		nvrtcx.WithNameExpressions(tileMatrixName), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	name, ok := cm.LoweredName(tileMatrixName)
	if !ok || name == "" || len(cm.TileIR()) == 0 {
		t.Fatal("missing TileIR or lowered template name")
	}
	t.Logf("lowered name=%s compiler log=%q TileIR bytes=%d sha256=%x", name, cm.Log(), len(cm.TileIR()), sha256.Sum256(cm.TileIR()))
}

func runTileMatrix(t *testing.T, mixed bool) {
	t.Helper()
	if os.Getenv("CUDA_TILE_TEST") != "1" {
		t.Skip("CUDA_TILE_TEST=1 requires NVRTC 13.3 and a TileIR-capable driver")
	}
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	cleanup := func(fn func() error) {
		t.Helper()
		t.Cleanup(func() {
			if err := fn(); err != nil {
				t.Errorf("cleanup: %v", err)
			}
		})
	}
	dev, err := cuda.NewDevice(0)
	check(err)
	cleanup(dev.Close)
	major, minor, err := dev.ComputeCapability()
	check(err)
	name, err := dev.Name()
	check(err)
	driverVersion, err := driver.DriverGetVersion()
	check(err)
	compilerMajor, compilerMinor, err := nvrtc.Version()
	check(err)
	t.Logf("GPU=%s compute=%d.%d driver=%d NVRTC=%d.%d path=%s", name, major, minor, driverVersion, compilerMajor, compilerMinor, nvrtc.Path())
	cm := compileTileMatrix(t, int(major), int(minor), mixed)
	mod, log, err := cm.LoadOwnedOutput(dev, nvrtcx.OutputTileIR)
	t.Logf("TileIR JIT info=%q error=%q load=%v", log.Info, log.Error, err)
	check(err)
	cleanup(mod.Close)
	lowered := "matrix"
	if mixed {
		lowered, _ = cm.LoweredName(tileMatrixName)
	}
	kernel, err := mod.Function(lowered)
	check(err)
	var transform *cuda.Kernel
	if mixed {
		ptxMod, log, err := cm.LoadOwnedOutput(dev, nvrtcx.OutputPTX)
		t.Logf("PTX JIT info=%q error=%q load=%v", log.Info, log.Error, err)
		check(err)
		cleanup(ptxMod.Close)
		transform, err = ptxMod.Function("transform")
		check(err)
	}
	stream, err := dev.NewStream()
	check(err)
	cleanup(stream.Destroy)
	var a, b [16 * 32]float32
	var c [16 * 16]float32
	for i := range a {
		a[i] = float32((i*3+i/32)%17-8) / 4
		b[i] = float32((i*5+i/16)%19-9) / 4
	}
	for i := range c {
		c[i] = float32(math.NaN())
	}
	alloc := func(data []float32) driver.DevicePtr {
		t.Helper()
		var p driver.DevicePtr
		check(dev.WithContext(func() error {
			var err error
			p, err = driver.MemAlloc(uint(len(data) * 4))
			return err
		}))
		cleanup(func() error { return dev.WithContext(p.MemFree) })
		check(dev.WithContext(func() error { return p.MemcpyHtoD(unsafe.Pointer(&data[0]), uint(len(data)*4)) }))
		return p
	}
	pa, pb, pc := alloc(a[:]), alloc(b[:]), alloc(c[:])
	// Synchronize before freeing buffers or modules even after a launch failure.
	cleanup(func() error { return stream.Synchronize(context.Background()) })
	check(kernel.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, stream.Raw(), []unsafe.Pointer{unsafe.Pointer(&pa), unsafe.Pointer(&pb), unsafe.Pointer(&pc)}))
	if mixed {
		// Both modules use the same allocation and stream. No host barrier
		// intervenes between the producer and consumer launches.
		check(transform.Launch([3]uint32{1, 1, 1}, [3]uint32{256, 1, 1}, 0, stream.Raw(), []unsafe.Pointer{unsafe.Pointer(&pc)}))
	}
	check(stream.Synchronize(context.Background()))
	check(dev.WithContext(func() error { return pc.MemcpyDtoH(unsafe.Pointer(&c[0]), uint(len(c)*4)) }))
	// Inputs are exact binary fractions; use a float64 oracle and a fixed
	// tolerance chosen before execution, with explicit nonfinite rejection.
	const absTol, relTol = 1e-5, 1e-5
	var maxError float64
	for i, v := range c {
		var want float64
		for k := 0; k < 32; k++ {
			want += float64(a[(i/16)*32+k]) * float64(b[k*16+i%16])
		}
		if mixed {
			want = 2*want + 1
		}
		got := float64(v)
		diff := math.Abs(got - want)
		if math.IsNaN(got) || math.IsInf(got, 0) || diff > absTol+relTol*math.Abs(want) {
			t.Fatalf("matrix[%d,%d]=%g want %g abs error=%g tolerance=%g", i/16, i%16, got, want, diff, absTol+relTol*math.Abs(want))
		}
		maxError = max(maxError, diff)
	}
	t.Logf("verified 256 matrix outputs mixed=%v max absolute error=%g absTol=%g relTol=%g", mixed, maxError, absTol, relTol)
}
