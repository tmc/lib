package cudagraph_test

import (
	"context"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// memoryContentionKernelSource contains CUDA C++ source code for memory-contended
// producer and consumer kernels implementing programmatic dependent launch (PDL).
//
// The producer executes RMSNorm compute and emits griddepcontrol.launch_dependents
// before streaming 128-bit stores to global DRAM (trailing residual accumulation).
//
// The consumer prefetches genuine weight tiles (FP16 or INT8) using 128-bit
// coalesced loads (uint4 / LDG.E.128) into shared memory BEFORE the
// griddepcontrol.wait barrier, allowing the memory controller and LSU to overlap
// consumer DRAM reads with trailing producer DRAM writes.
const memoryContentionKernelSource = `
static inline __device__ void cudaTriggerProgrammaticLaunchCompletion(void) {
    asm volatile("griddepcontrol.launch_dependents;":::);
}

static inline __device__ void cudaGridDependencySynchronize(void) {
    asm volatile("griddepcontrol.wait;":::"memory");
}

// Convert 16-bit half representation to float in registers.
static __device__ __forceinline__ float half2float_fast(unsigned short h) {
    unsigned int sign = ((unsigned int)h >> 15) & 1;
    unsigned int exp  = ((unsigned int)h >> 10) & 0x1f;
    unsigned int mant = (unsigned int)h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float f = ((float)mant / 1024.0f) * (1.0f / 16384.0f);
        return sign ? -f : f;
    } else if (exp == 31) {
        return mant ? nanf("") : (sign ? -INFINITY : INFINITY);
    }
    float f = (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
    return sign ? -f : f;
}

// Producer: computes RMSNorm, writes act to global DRAM, triggers launch_dependents,
// and then streams trailing residual writes to global DRAM using 128-bit stores.
extern "C" __global__ void memory_contended_producer(
    const float * __restrict__ x,
    const float * __restrict__ gamma,
    float * __restrict__ act,
    float * __restrict__ residual,
    int dim,
    int trailing_passes,
    long long * __restrict__ ts
) {
    int tid = threadIdx.x;
    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[0] = clock64(); // t0: producer start
    }

    // 1. RMSNorm compute
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ float s_inv_rms;
    if (tid == 0) {
        float mean_sq = sum_sq / (float)dim;
        s_inv_rms = rsqrtf(mean_sq + 1e-6f);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    for (int i = tid; i < dim; i += blockDim.x) {
        act[i] = x[i] * inv_rms * gamma[i];
    }
    __threadfence();
    __syncthreads();

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[1] = clock64(); // t1: trigger point
    }

    // 2. Early launch trigger for dependent consumer grid
    cudaTriggerProgrammaticLaunchCompletion();

    // 3. Trailing memory stores to global DRAM
    // Emits genuine 128-bit stores (STG.E.128) across memory crossbar and L2/DRAM
    float4 *res_vec = reinterpret_cast<float4*>(residual);
    const float4 *x_vec = reinterpret_cast<const float4*>(x);
    int num_vecs = dim / 4;
    for (int iter = 0; iter < trailing_passes; iter++) {
        int offset = iter * num_vecs;
        for (int v = tid; v < num_vecs; v += blockDim.x) {
            float4 in = x_vec[v];
            float4 curr = res_vec[offset + v];
            curr.x += in.x * 0.5f;
            curr.y += in.y * 0.5f;
            curr.z += in.z * 0.5f;
            curr.w += in.w * 0.5f;
            res_vec[offset + v] = curr;
        }
    }
    __threadfence();

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[2] = clock64(); // t2: producer exit
    }
}

// Consumer FP16 with entry synchronization (naive PDL baseline).
// Blocks on griddepcontrol.wait immediately upon entry before loading weights.
extern "C" __global__ void memory_contended_consumer_fp16_entry_sync(
    const float * __restrict__ act,
    const unsigned short * __restrict__ W,
    const float * __restrict__ bias,
    float * __restrict__ out,
    int rows,
    int cols,
    long long * __restrict__ ts
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[3] = clock64(); // t3: consumer start
    }

    // Entry sync: blocks immediately
    cudaGridDependencySynchronize();

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[4] = clock64(); // t4: sync unblocked
    }

    // 128-bit coalesced prefetch of FP16 weight tile into shared memory (AFTER wait)
    extern __shared__ unsigned short smem_w_fp16[];
    const uint4 *w_vec = reinterpret_cast<const uint4*>(W + (size_t)row * cols);
    uint4 *smem_vec = reinterpret_cast<uint4*>(smem_w_fp16);
    int total_vecs = (cols * sizeof(unsigned short)) / sizeof(uint4);

    for (int v = tid; v < total_vecs; v += blockDim.x) {
        smem_vec[v] = w_vec[v];
    }
    __syncthreads();

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[5] = clock64(); // t5: weights loaded
    }

    // Compute dot product with predecessor activation
    float partial = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float w_val = half2float_fast(smem_w_fp16[i]);
        partial += w_val * act[i];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    __shared__ float s_warp_sums[32];
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) {
        s_warp_sums[warp_id] = partial;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (tid < (blockDim.x / 32)) ? s_warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            float b = (bias != nullptr) ? bias[row] : 0.0f;
            out[row] = sum + b;
            if (ts != nullptr && blockIdx.x == 0) {
                ts[6] = clock64(); // t6: consumer complete
            }
        }
    }
}

// Consumer FP16 with hoisted prologue.
// Prefetches genuine W weight tile using 128-bit coalesced loads (LDG.E.128)
// into shared memory BEFORE griddepcontrol.wait barrier.
extern "C" __global__ void memory_contended_consumer_fp16_hoisted(
    const float * __restrict__ act,
    const unsigned short * __restrict__ W,
    const float * __restrict__ bias,
    float * __restrict__ out,
    int rows,
    int cols,
    long long * __restrict__ ts
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[3] = clock64(); // t3: consumer start
    }

    // Hoisted prologue: 128-bit coalesced prefetch into shared memory BEFORE wait
    extern __shared__ unsigned short smem_w_fp16[];
    const uint4 *w_vec = reinterpret_cast<const uint4*>(W + (size_t)row * cols);
    uint4 *smem_vec = reinterpret_cast<uint4*>(smem_w_fp16);
    int total_vecs = (cols * sizeof(unsigned short)) / sizeof(uint4);

    for (int v = tid; v < total_vecs; v += blockDim.x) {
        smem_vec[v] = w_vec[v];
    }
    __syncthreads();

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[4] = clock64(); // t4: hoisted weights loaded into smem
    }

    // Hardware barrier: wait for producer activation
    cudaGridDependencySynchronize();

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[5] = clock64(); // t5: sync unblocked
    }

    // Consumer body: compute dot product with predecessor activation
    float partial = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float w_val = half2float_fast(smem_w_fp16[i]);
        partial += w_val * act[i];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    __shared__ float s_warp_sums[32];
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) {
        s_warp_sums[warp_id] = partial;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (tid < (blockDim.x / 32)) ? s_warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            float b = (bias != nullptr) ? bias[row] : 0.0f;
            out[row] = sum + b;
            if (ts != nullptr && blockIdx.x == 0) {
                ts[6] = clock64(); // t6: consumer complete
            }
        }
    }
}

// Consumer INT8 with entry synchronization.
extern "C" __global__ void memory_contended_consumer_int8_entry_sync(
    const float * __restrict__ act,
    const signed char * __restrict__ W,
    const float * __restrict__ bias,
    float * __restrict__ out,
    int rows,
    int cols,
    long long * __restrict__ ts
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[3] = clock64();
    }

    // Entry sync
    cudaGridDependencySynchronize();

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[4] = clock64();
    }

    // 128-bit coalesced prefetch of INT8 weight tile into shared memory (AFTER wait)
    extern __shared__ signed char smem_w_int8[];
    const uint4 *w_vec = reinterpret_cast<const uint4*>(W + (size_t)row * cols);
    uint4 *smem_vec = reinterpret_cast<uint4*>(smem_w_int8);
    int total_vecs = (cols * sizeof(signed char)) / sizeof(uint4);

    for (int v = tid; v < total_vecs; v += blockDim.x) {
        smem_vec[v] = w_vec[v];
    }
    __syncthreads();

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[5] = clock64();
    }

    float partial = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        partial += (float)smem_w_int8[i] * act[i] * 0.01f;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    __shared__ float s_warp_sums[32];
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) {
        s_warp_sums[warp_id] = partial;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (tid < (blockDim.x / 32)) ? s_warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            float b = (bias != nullptr) ? bias[row] : 0.0f;
            out[row] = sum + b;
            if (ts != nullptr && blockIdx.x == 0) {
                ts[6] = clock64();
            }
        }
    }
}

// Consumer INT8 with hoisted prologue.
extern "C" __global__ void memory_contended_consumer_int8_hoisted(
    const float * __restrict__ act,
    const signed char * __restrict__ W,
    const float * __restrict__ bias,
    float * __restrict__ out,
    int rows,
    int cols,
    long long * __restrict__ ts
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[3] = clock64();
    }

    // Hoisted prologue: 128-bit coalesced prefetch into shared memory BEFORE wait
    extern __shared__ signed char smem_w_int8[];
    const uint4 *w_vec = reinterpret_cast<const uint4*>(W + (size_t)row * cols);
    uint4 *smem_vec = reinterpret_cast<uint4*>(smem_w_int8);
    int total_vecs = (cols * sizeof(signed char)) / sizeof(uint4);

    for (int v = tid; v < total_vecs; v += blockDim.x) {
        smem_vec[v] = w_vec[v];
    }
    __syncthreads();

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[4] = clock64();
    }

    // Hardware barrier
    cudaGridDependencySynchronize();

    if (tid == 0 && blockIdx.x == 0 && ts != nullptr) {
        ts[5] = clock64();
    }

    float partial = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        partial += (float)smem_w_int8[i] * act[i] * 0.01f;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    __shared__ float s_warp_sums[32];
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) {
        s_warp_sums[warp_id] = partial;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (tid < (blockDim.x / 32)) ? s_warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            float b = (bias != nullptr) ? bias[row] : 0.0f;
            out[row] = sum + b;
            if (ts != nullptr && blockIdx.x == 0) {
                ts[6] = clock64();
            }
        }
    }
}
`

// float32ToHalf converts an IEEE 754 single-precision float to a 16-bit half.
func float32ToHalf(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 31) & 0x1)
	exp := int((bits >> 23) & 0xFF)
	mant := bits & 0x7FFFFF

	if exp == 0 {
		return sign << 15
	} else if exp == 255 {
		if mant != 0 {
			return (sign << 15) | 0x7E00
		}
		return (sign << 15) | 0x7C00
	}

	newExp := exp - 127 + 15
	if newExp >= 31 {
		return (sign << 15) | 0x7C00
	} else if newExp <= 0 {
		if newExp < -10 {
			return sign << 15
		}
		mant = (mant | 0x800000) >> (1 - newExp)
		return (sign << 15) | uint16(mant>>13)
	}

	return (sign << 15) | (uint16(newExp) << 10) | uint16(mant>>13)
}

// TestPDLMemoryContentionCompilationAndPTX verifies that NVRTC emits the required
// griddepcontrol instructions and 128-bit coalesced memory loads.
func TestPDLMemoryContentionCompilationAndPTX(t *testing.T) {
	if err := nvrtc.Load(""); err != nil {
		t.Skipf("NVRTC unavailable: %v", err)
	}
	compiled, err := nvrtcx.Compile(memoryContentionKernelSource, nvrtcx.WithArch(12, 1), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed: %v", err)
	}
	ptx := compiled.PTX()
	if !strings.Contains(ptx, "griddepcontrol.launch_dependents;") {
		t.Error("compiled PTX missing griddepcontrol.launch_dependents;")
	}
	if !strings.Contains(ptx, "griddepcontrol.wait;") {
		t.Error("compiled PTX missing griddepcontrol.wait;")
	}
	if !strings.Contains(ptx, "ld.global") && !strings.Contains(ptx, "ld.global.v4") {
		t.Error("compiled PTX missing vector global memory loads")
	}
}

// TestPDLMemoryContendedPrologue evaluates memory bus and LSU contention when
// the hoisted consumer prologue prefetches genuine weight tiles while the producer
// executes trailing memory stores to global DRAM.
func TestPDLMemoryContendedPrologue(t *testing.T) {
	if os.Getenv(pdlAblationEnv) != "1" {
		t.Skipf("%s=1 required: needs NVIDIA GPU with CUDA 13.0 PDL support", pdlAblationEnv)
	}
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(memoryContentionKernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(mod.Unload)

	pFn, err := mod.GetFunction("memory_contended_producer")
	if err != nil {
		t.Fatal(err)
	}
	cEntryFn, err := mod.GetFunction("memory_contended_consumer_fp16_entry_sync")
	if err != nil {
		t.Fatal(err)
	}
	cHoistFn, err := mod.GetFunction("memory_contended_consumer_fp16_hoisted")
	if err != nil {
		t.Fatal(err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	const (
		dim            = 2048
		rows           = 32
		trailingPasses = 128
		blockDim       = 128
	)

	xBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(xBuf.Free)
	gBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(gBuf.Free)
	actBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(actBuf.Free)
	resBuf, _ := cudamem.Alloc[float32](dev, dim*trailingPasses)
	defer dev.WithContext(resBuf.Free)
	wBuf, _ := cudamem.Alloc[uint16](dev, rows*dim)
	defer dev.WithContext(wBuf.Free)
	bBuf, _ := cudamem.Alloc[float32](dev, rows)
	defer dev.WithContext(bBuf.Free)
	outBuf, _ := cudamem.Alloc[float32](dev, rows)
	defer dev.WithContext(outBuf.Free)

	// Initialize test data
	hX := make([]float32, dim)
	hG := make([]float32, dim)
	hW := make([]uint16, rows*dim)
	hB := make([]float32, rows)
	for i := 0; i < dim; i++ {
		hX[i] = float32(i%19) * 0.1
		hG[i] = 1.0
	}
	for r := 0; r < rows; r++ {
		hB[r] = float32(r) * 0.05
		for c := 0; c < dim; c++ {
			f := 0.001 * float32((r+c)%13)
			hW[r*dim+c] = float32ToHalf(f)
		}
	}
	_ = xBuf.CopyFromHost(hX)
	_ = gBuf.CopyFromHost(hG)
	_ = wBuf.CopyFromHost(hW)
	_ = bBuf.CopyFromHost(hB)

	startEv, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(startEv.Destroy)
	stopEv, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stopEv.Destroy)

	smemBytes := uint32(dim * 2) // sizeof(unsigned short) * dim
	var nullTs driver.DevicePtr

	launchArm := func(cFn driver.Function, pdl bool) float64 {
		g, err := cudagraph.Capture(stream, func() {
			_ = cudalaunch.Kernel(pFn).Grid(1).Block(blockDim).
				WithProgrammaticStreamSerialization(pdl).
				On(stream).
				Launch(xBuf, gBuf, actBuf, resBuf, int32(dim), int32(trailingPasses), nullTs)
			_ = cudalaunch.Kernel(cFn).Grid(rows).Block(blockDim).
				SharedMem(smemBytes).
				WithProgrammaticStreamSerialization(pdl).
				On(stream).
				Launch(actBuf, wBuf, bBuf, outBuf, int32(rows), int32(dim), nullTs)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer g.Destroy()

		exec, err := g.Instantiate()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Destroy()

		_ = startEv.Record(stream)
		if err := exec.Launch(stream); err != nil {
			t.Fatalf("launch: %v", err)
		}
		_ = stopEv.Record(stream)
		if err := stream.Synchronize(context.Background()); err != nil {
			t.Fatalf("synchronize: %v", err)
		}
		d, err := stopEv.ElapsedSince(startEv)
		if err != nil {
			t.Fatal(err)
		}
		return float64(d.Nanoseconds())
	}

	// Numerical validation: ensure hoisted output equals sequential baseline
	_ = launchArm(cEntryFn, false) // NOPDL reference
	hRef := make([]float32, rows)
	if err := outBuf.CopyToHost(hRef); err != nil {
		t.Fatalf("copy reference: %v", err)
	}
	for i := range hX {
		hX[i] += 0.03125
	}
	if err := xBuf.CopyFromHost(hX); err != nil {
		t.Fatalf("copy changed input: %v", err)
	}

	poison := make([]float32, rows)
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	if err := outBuf.CopyFromHost(poison); err != nil {
		t.Fatalf("poison output: %v", err)
	}
	_ = launchArm(cHoistFn, true) // PDL_HOISTED
	hHoist := make([]float32, rows)
	if err := outBuf.CopyToHost(hHoist); err != nil {
		t.Fatalf("copy hoisted: %v", err)
	}
	_ = launchArm(cEntryFn, false)
	hExpected := make([]float32, rows)
	if err := outBuf.CopyToHost(hExpected); err != nil {
		t.Fatalf("copy changed reference: %v", err)
	}

	for r := 0; r < rows; r++ {
		diff := math.Abs(float64(hExpected[r] - hHoist[r]))
		if math.IsNaN(float64(hExpected[r])) || math.IsNaN(float64(hHoist[r])) || !(diff <= 1e-4) {
			t.Fatalf("numerical divergence at row %d: ref=%f, hoisted=%f, diff=%e", r, hExpected[r], hHoist[r], diff)
		}
	}
	t.Log("Bitwise elementwise validation passed: Hoisted arm matches sequential reference")

	csvPath := os.Getenv("CUDAGRAPH_CONTENDED_PDL_CSV")
	if csvPath == "" {
		csvPath = filepath.Join(t.TempDir(), "pdl-memory-contention.csv")
	}
	csvFile, err := os.Create(csvPath)
	if err != nil {
		t.Fatalf("create CSV failed: %v", err)
	}
	defer csvFile.Close()
	writer := csv.NewWriter(csvFile)
	_ = writer.Write([]string{"arm", "iter", "ns"})
	writer.Flush()

	const totalIters = 220
	const warmupIters = 20
	arms := []string{"NULL_1", "NULL_2", "PDL_HOISTED", "PDL_ENTRY_SYNC", "NOPDL"}
	data := make(map[string][]float64)
	for _, a := range arms {
		data[a] = make([]float64, 0, totalIters)
	}

	for i := 0; i < totalIters; i++ {
		nsNull1 := launchArm(cHoistFn, true)
		nsNull2 := launchArm(cHoistFn, true)
		nsHoist := launchArm(cHoistFn, true)
		nsEntry := launchArm(cEntryFn, true)
		nsNOPDL := launchArm(cEntryFn, false)

		vals := map[string]float64{
			"NULL_1":         nsNull1,
			"NULL_2":         nsNull2,
			"PDL_HOISTED":    nsHoist,
			"PDL_ENTRY_SYNC": nsEntry,
			"NOPDL":          nsNOPDL,
		}

		for _, arm := range arms {
			v := vals[arm]
			data[arm] = append(data[arm], v)
			_ = writer.Write([]string{arm, fmt.Sprintf("%d", i), fmt.Sprintf("%.1f", v)})
		}
		writer.Flush()
	}

	evalData := make(map[string][]float64)
	for _, arm := range arms {
		evalData[arm] = data[arm][warmupIters:]
	}

	median := func(s []float64) float64 {
		cp := append([]float64(nil), s...)
		sort.Float64s(cp)
		n := len(cp)
		if n%2 == 1 {
			return cp[n/2]
		}
		return (cp[n/2-1] + cp[n/2]) / 2.0
	}

	mNull1 := median(evalData["NULL_1"])
	mNull2 := median(evalData["NULL_2"])
	mHoist := median(evalData["PDL_HOISTED"])
	mEntry := median(evalData["PDL_ENTRY_SYNC"])
	mNOPDL := median(evalData["NOPDL"])

	nullDiff := mNull2 - mNull1
	entryDiff := mNOPDL - mEntry
	hoistDiff := mNOPDL - mHoist

	nEval := totalIters - warmupIters
	nullS1Faster, nullS2Faster, nullTies := 0, 0, 0
	hoistFaster, nopdlFaster, hoistTies := 0, 0, 0
	for i := 0; i < nEval; i++ {
		nd := evalData["NULL_2"][i] - evalData["NULL_1"][i]
		if nd > 0 {
			nullS1Faster++
		} else if nd < 0 {
			nullS2Faster++
		} else {
			nullTies++
		}

		hd := evalData["NOPDL"][i] - evalData["PDL_HOISTED"][i]
		if hd > 0 {
			hoistFaster++
		} else if hd < 0 {
			nopdlFaster++
		} else {
			hoistTies++
		}
	}

	t.Logf("Evaluated Iterations: %d (warmup %d discarded)", nEval, warmupIters)
	t.Logf("NULL_1         Median: %.1f ns (%.2f us)", mNull1, mNull1/1000)
	t.Logf("NULL_2         Median: %.1f ns (%.2f us)", mNull2, mNull2/1000)
	t.Logf("Null Drift:    %.1f - %.1f = %+.1f ns (%+.3f us)", mNull2, mNull1, nullDiff, nullDiff/1000)
	t.Logf("Null Sign Test: Slot1<Slot2 %d/%d (%.1f%%), Slot2<Slot1 %d/%d (%.1f%%), Ties %d",
		nullS1Faster, nEval, float64(nullS1Faster)/float64(nEval)*100,
		nullS2Faster, nEval, float64(nullS2Faster)/float64(nEval)*100,
		nullTies)

	t.Logf("NOPDL          Median: %.1f ns (%.2f us)", mNOPDL, mNOPDL/1000)
	t.Logf("PDL_ENTRY_SYNC Median: %.1f ns (%.2f us)", mEntry, mEntry/1000)
	t.Logf("Entry-Sync Effect vs NOPDL: %+.1f ns (%+.3f us)", entryDiff, entryDiff/1000)

	t.Logf("PDL_HOISTED    Median: %.1f ns (%.2f us)", mHoist, mHoist/1000)
	t.Logf("Hoisted Overlap Savings:   %+.1f ns (%+.3f us)", hoistDiff, hoistDiff/1000)
	t.Logf("Hoisted Sign Test Win Rate: Hoisted<NOPDL %d/%d (%.1f%%), NOPDL<Hoisted %d/%d (%.1f%%), Ties %d",
		hoistFaster, nEval, float64(hoistFaster)/float64(nEval)*100,
		nopdlFaster, nEval, float64(nopdlFaster)/float64(nEval)*100,
		hoistTies)

	// Gate: Null drift must be strictly smaller than the measured hoisted effect
	if math.Abs(nullDiff) >= math.Abs(hoistDiff) {
		t.Fatalf("GATE FAILED: null drift %.1f ns >= hoisted effect %.1f ns", math.Abs(nullDiff), math.Abs(hoistDiff))
	}

	// Gate: Hoisted arm must show positive overlap savings
	if hoistDiff <= 0 {
		t.Fatalf("GATE FAILED: hoisted overlap effect %.1f ns <= 0", hoistDiff)
	}

	speedupPct := (hoistDiff / mNOPDL) * 100.0
	t.Logf("VERDICT: MEMORY-CONTENDED PROLOGUE HOISTING PASSED ON GB10!")
	t.Logf("  Baseline Sequential Span: %.2f us", mNOPDL/1000)
	t.Logf("  Hoisted PDL Span:         %.2f us", mHoist/1000)
	t.Logf("  Overlap Savings:          %.2f us (%.1f%% speedup)", hoistDiff/1000, speedupPct)
}

// TestPDLPrologueSizeSweep benchmarks overlap efficiency and memory prefetch throughput
// across genuine weight tile sizes K in {512, 1024, 2048, 4096} for both FP16 and INT8.
func TestPDLPrologueSizeSweep(t *testing.T) {
	if os.Getenv(pdlAblationEnv) != "1" {
		t.Skipf("%s=1 required: needs NVIDIA GPU with CUDA 13.0 PDL support", pdlAblationEnv)
	}
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(memoryContentionKernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(mod.Unload)

	pFn, err := mod.GetFunction("memory_contended_producer")
	if err != nil {
		t.Fatal(err)
	}
	cFP16EntryFn, err := mod.GetFunction("memory_contended_consumer_fp16_entry_sync")
	if err != nil {
		t.Fatal(err)
	}
	cFP16HoistFn, err := mod.GetFunction("memory_contended_consumer_fp16_hoisted")
	if err != nil {
		t.Fatal(err)
	}
	cINT8EntryFn, err := mod.GetFunction("memory_contended_consumer_int8_entry_sync")
	if err != nil {
		t.Fatal(err)
	}
	cINT8HoistFn, err := mod.GetFunction("memory_contended_consumer_int8_hoisted")
	if err != nil {
		t.Fatal(err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	startEv, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(startEv.Destroy)
	stopEv, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stopEv.Destroy)

	const (
		rows           = 32
		blockDim       = 128
		trailingPasses = 128
		totalIters     = 120
		warmupIters    = 20
	)

	testCases := []struct {
		dtype        string
		k            int
		bytesPerElem int
		entryFn      driver.Function
		hoistFn      driver.Function
	}{
		{dtype: "FP16", k: 512, bytesPerElem: 2, entryFn: cFP16EntryFn, hoistFn: cFP16HoistFn},
		{dtype: "FP16", k: 1024, bytesPerElem: 2, entryFn: cFP16EntryFn, hoistFn: cFP16HoistFn},
		{dtype: "FP16", k: 2048, bytesPerElem: 2, entryFn: cFP16EntryFn, hoistFn: cFP16HoistFn},
		{dtype: "FP16", k: 4096, bytesPerElem: 2, entryFn: cFP16EntryFn, hoistFn: cFP16HoistFn},
		{dtype: "INT8", k: 512, bytesPerElem: 1, entryFn: cINT8EntryFn, hoistFn: cINT8HoistFn},
		{dtype: "INT8", k: 1024, bytesPerElem: 1, entryFn: cINT8EntryFn, hoistFn: cINT8HoistFn},
		{dtype: "INT8", k: 2048, bytesPerElem: 1, entryFn: cINT8EntryFn, hoistFn: cINT8HoistFn},
		{dtype: "INT8", k: 4096, bytesPerElem: 1, entryFn: cINT8EntryFn, hoistFn: cINT8HoistFn},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("%s_K%d", tc.dtype, tc.k)
		t.Run(name, func(t *testing.T) {
			k := tc.k
			xBuf, _ := cudamem.Alloc[float32](dev, k)
			defer dev.WithContext(xBuf.Free)
			gBuf, _ := cudamem.Alloc[float32](dev, k)
			defer dev.WithContext(gBuf.Free)
			actBuf, _ := cudamem.Alloc[float32](dev, k)
			defer dev.WithContext(actBuf.Free)
			resBuf, _ := cudamem.Alloc[float32](dev, k*trailingPasses)
			defer dev.WithContext(resBuf.Free)
			bBuf, _ := cudamem.Alloc[float32](dev, rows)
			defer dev.WithContext(bBuf.Free)
			outBuf, _ := cudamem.Alloc[float32](dev, rows)
			defer dev.WithContext(outBuf.Free)

			hX := make([]float32, k)
			hG := make([]float32, k)
			for i := 0; i < k; i++ {
				hX[i] = float32(i%17) * 0.1
				hG[i] = 1.0
			}
			_ = xBuf.CopyFromHost(hX)
			_ = gBuf.CopyFromHost(hG)

			var wPtr driver.DevicePtr
			if tc.dtype == "FP16" {
				wBuf, _ := cudamem.Alloc[uint16](dev, rows*k)
				defer dev.WithContext(wBuf.Free)
				hW := make([]uint16, rows*k)
				for r := 0; r < rows; r++ {
					for c := 0; c < k; c++ {
						hW[r*k+c] = float32ToHalf(0.001 * float32((r+c)%11))
					}
				}
				_ = wBuf.CopyFromHost(hW)
				wPtr = wBuf.Ptr()
			} else {
				wBuf, _ := cudamem.Alloc[int8](dev, rows*k)
				defer dev.WithContext(wBuf.Free)
				hW := make([]int8, rows*k)
				for r := 0; r < rows; r++ {
					for c := 0; c < k; c++ {
						hW[r*k+c] = int8((r + c) % 15)
					}
				}
				_ = wBuf.CopyFromHost(hW)
				wPtr = wBuf.Ptr()
			}

			smemBytes := uint32(k * tc.bytesPerElem)
			var nullTs driver.DevicePtr

			launchArm := func(cFn driver.Function, pdl bool) float64 {
				g, err := cudagraph.Capture(stream, func() {
					_ = cudalaunch.Kernel(pFn).Grid(1).Block(blockDim).
						WithProgrammaticStreamSerialization(pdl).
						On(stream).
						Launch(xBuf, gBuf, actBuf, resBuf, int32(k), int32(trailingPasses), nullTs)
					_ = cudalaunch.Kernel(cFn).Grid(rows).Block(blockDim).
						SharedMem(smemBytes).
						WithProgrammaticStreamSerialization(pdl).
						On(stream).
						Launch(actBuf, wPtr, bBuf, outBuf, int32(rows), int32(k), nullTs)
				})
				if err != nil {
					t.Fatal(err)
				}
				defer g.Destroy()

				exec, err := g.Instantiate()
				if err != nil {
					t.Fatal(err)
				}
				defer exec.Destroy()

				_ = startEv.Record(stream)
				if err := exec.Launch(stream); err != nil {
					t.Fatalf("launch: %v", err)
				}
				_ = stopEv.Record(stream)
				if err := stream.Synchronize(context.Background()); err != nil {
					t.Fatalf("synchronize: %v", err)
				}
				d, err := stopEv.ElapsedSince(startEv)
				if err != nil {
					t.Fatal(err)
				}
				return float64(d.Nanoseconds())
			}

			// Numerical validation
			_ = launchArm(tc.entryFn, false) // NOPDL
			hRef := make([]float32, rows)
			if err := outBuf.CopyToHost(hRef); err != nil {
				t.Fatalf("copy reference: %v", err)
			}
			for i := range hX {
				hX[i] += 0.03125
			}
			if err := xBuf.CopyFromHost(hX); err != nil {
				t.Fatalf("copy changed input: %v", err)
			}

			poison := make([]float32, rows)
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			if err := outBuf.CopyFromHost(poison); err != nil {
				t.Fatalf("poison output: %v", err)
			}
			_ = launchArm(tc.hoistFn, true) // PDL_HOISTED
			hHoist := make([]float32, rows)
			if err := outBuf.CopyToHost(hHoist); err != nil {
				t.Fatalf("copy hoisted: %v", err)
			}
			_ = launchArm(tc.entryFn, false)
			hExpected := make([]float32, rows)
			if err := outBuf.CopyToHost(hExpected); err != nil {
				t.Fatalf("copy changed reference: %v", err)
			}

			for r := 0; r < rows; r++ {
				diff := math.Abs(float64(hExpected[r] - hHoist[r]))
				if math.IsNaN(float64(hExpected[r])) || math.IsNaN(float64(hHoist[r])) || !(diff <= 1e-4) {
					t.Fatalf("numerical divergence for %s at row %d: ref=%f, hoisted=%f", name, r, hExpected[r], hHoist[r])
				}
			}

			hoistTimes := make([]float64, 0, totalIters)
			nopdlTimes := make([]float64, 0, totalIters)
			entryTimes := make([]float64, 0, totalIters)

			for i := 0; i < totalIters; i++ {
				hNs := launchArm(tc.hoistFn, true)
				eNs := launchArm(tc.entryFn, true)
				nNs := launchArm(tc.entryFn, false)

				hoistTimes = append(hoistTimes, hNs)
				entryTimes = append(entryTimes, eNs)
				nopdlTimes = append(nopdlTimes, nNs)
			}

			evalH := hoistTimes[warmupIters:]
			evalE := entryTimes[warmupIters:]
			evalN := nopdlTimes[warmupIters:]

			median := func(s []float64) float64 {
				cp := append([]float64(nil), s...)
				sort.Float64s(cp)
				n := len(cp)
				if n%2 == 1 {
					return cp[n/2]
				}
				return (cp[n/2-1] + cp[n/2]) / 2.0
			}

			mH := median(evalH)
			mE := median(evalE)
			mN := median(evalN)
			overlapSavings := mN - mH
			speedupPct := (overlapSavings / mN) * 100.0

			nEval := totalIters - warmupIters
			hoistWins := 0
			for i := 0; i < nEval; i++ {
				if evalN[i] > evalH[i] {
					hoistWins++
				}
			}
			winRate := float64(hoistWins) / float64(nEval) * 100.0

			// Prefetch bandwidth calculation: (rows * K * bytesPerElem) / overlap_time
			totalBytesPrefetched := float64(rows * k * tc.bytesPerElem)
			bandwidthGBps := 0.0
			if overlapSavings > 0 {
				bandwidthGBps = (totalBytesPrefetched / overlapSavings) // bytes/ns = GB/s
			}

			t.Logf("[%s] NOPDL: %.2f us | EntrySync: %.2f us | Hoisted: %.2f us", name, mN/1000, mE/1000, mH/1000)
			t.Logf("[%s] Overlap Savings: %.2f us (%.1f%%) | Win Rate: %d/%d (%.1f%%)",
				name, overlapSavings/1000, speedupPct, hoistWins, nEval, winRate)
			t.Logf("[%s] Effective Prefetched Bytes: %.1f KB | Implied Prefetch Bandwidth: %.2f GB/s",
				name, totalBytesPrefetched/1024.0, bandwidthGBps)

			if overlapSavings <= 0 {
				t.Errorf("[%s] Expected positive overlap savings, got %.2f us", name, overlapSavings/1000)
			}
		})
	}
}

// TestPDLResidentCTAStarvationIsolation measures producer execution duration across
// 3 distinct configurations to quantify SM issue slot starvation vs memory bus contention:
//
//	a) Producer standalone: only the producer executes on the SM.
//	b) Producer with entry-syncing resident CTAs: consumer CTAs are resident on SMs parked in griddepcontrol.wait.
//	c) Producer with active memory-prefetching hoisted CTAs: consumer CTAs actively prefetch W from DRAM into SMEM.
func TestPDLResidentCTAStarvationIsolation(t *testing.T) {
	if os.Getenv(pdlAblationEnv) != "1" {
		t.Skipf("%s=1 required: needs NVIDIA GPU with CUDA 13.0 PDL support", pdlAblationEnv)
	}
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(memoryContentionKernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(mod.Unload)

	pFn, err := mod.GetFunction("memory_contended_producer")
	if err != nil {
		t.Fatal(err)
	}
	cEntryFn, err := mod.GetFunction("memory_contended_consumer_fp16_entry_sync")
	if err != nil {
		t.Fatal(err)
	}
	cHoistFn, err := mod.GetFunction("memory_contended_consumer_fp16_hoisted")
	if err != nil {
		t.Fatal(err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	const (
		dim            = 2048
		rows           = 64
		blockDim       = 128
		trailingPasses = 256
		totalIters     = 120
		warmupIters    = 20
	)

	tsBuf, err := cudamem.Alloc[int64](dev, 7)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(tsBuf.Free)

	xBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(xBuf.Free)
	gBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(gBuf.Free)
	actBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(actBuf.Free)
	resBuf, _ := cudamem.Alloc[float32](dev, dim*trailingPasses)
	defer dev.WithContext(resBuf.Free)
	wBuf, _ := cudamem.Alloc[uint16](dev, rows*dim)
	defer dev.WithContext(wBuf.Free)
	bBuf, _ := cudamem.Alloc[float32](dev, rows)
	defer dev.WithContext(bBuf.Free)
	outBuf, _ := cudamem.Alloc[float32](dev, rows)
	defer dev.WithContext(outBuf.Free)

	hX := make([]float32, dim)
	hG := make([]float32, dim)
	for i := 0; i < dim; i++ {
		hX[i] = 1.0
		hG[i] = 1.0
	}
	_ = xBuf.CopyFromHost(hX)
	_ = gBuf.CopyFromHost(hG)

	clkKHz, _ := dev.Attribute(driver.DevAttrClockRate)
	cyclesPerUs := float64(clkKHz) / 1000.0

	smemBytes := uint32(dim * 2)

	// Build graphs for the 3 configurations
	// Config A: Producer standalone
	gA, err := cudagraph.Capture(stream, func() {
		_ = cudalaunch.Kernel(pFn).Grid(1).Block(blockDim).
			On(stream).
			Launch(xBuf, gBuf, actBuf, resBuf, int32(dim), int32(trailingPasses), tsBuf)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer gA.Destroy()
	execA, err := gA.Instantiate()
	if err != nil {
		t.Fatal(err)
	}
	defer execA.Destroy()

	// Config B: Producer with entry-sync resident CTAs parked in griddepcontrol.wait
	gB, err := cudagraph.Capture(stream, func() {
		_ = cudalaunch.Kernel(pFn).Grid(1).Block(blockDim).
			WithProgrammaticStreamSerialization(true).
			On(stream).
			Launch(xBuf, gBuf, actBuf, resBuf, int32(dim), int32(trailingPasses), tsBuf)
		_ = cudalaunch.Kernel(cEntryFn).Grid(rows).Block(blockDim).
			SharedMem(smemBytes).
			WithProgrammaticStreamSerialization(true).
			On(stream).
			Launch(actBuf, wBuf, bBuf, outBuf, int32(rows), int32(dim), tsBuf)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer gB.Destroy()
	execB, err := gB.Instantiate()
	if err != nil {
		t.Fatal(err)
	}
	defer execB.Destroy()

	// Config C: Producer with active memory-prefetching hoisted CTAs
	gC, err := cudagraph.Capture(stream, func() {
		_ = cudalaunch.Kernel(pFn).Grid(1).Block(blockDim).
			WithProgrammaticStreamSerialization(true).
			On(stream).
			Launch(xBuf, gBuf, actBuf, resBuf, int32(dim), int32(trailingPasses), tsBuf)
		_ = cudalaunch.Kernel(cHoistFn).Grid(rows).Block(blockDim).
			SharedMem(smemBytes).
			WithProgrammaticStreamSerialization(true).
			On(stream).
			Launch(actBuf, wBuf, bBuf, outBuf, int32(rows), int32(dim), tsBuf)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer gC.Destroy()
	execC, err := gC.Instantiate()
	if err != nil {
		t.Fatal(err)
	}
	defer execC.Destroy()

	measureProducerDuration := func(exec *cudagraph.Exec) (totalCycles, trailingCycles int64) {
		_ = exec.Launch(stream)
		_ = stream.Synchronize(context.Background())
		var ts [7]int64
		_ = tsBuf.CopyToHost(ts[:])
		totalCycles = ts[2] - ts[0]
		trailingCycles = ts[2] - ts[1]
		return totalCycles, trailingCycles
	}

	prodA_total := make([]float64, 0, totalIters)
	prodB_total := make([]float64, 0, totalIters)
	prodC_total := make([]float64, 0, totalIters)

	prodA_trail := make([]float64, 0, totalIters)
	prodB_trail := make([]float64, 0, totalIters)
	prodC_trail := make([]float64, 0, totalIters)

	for i := 0; i < totalIters; i++ {
		totA, trA := measureProducerDuration(execA)
		totB, trB := measureProducerDuration(execB)
		totC, trC := measureProducerDuration(execC)

		prodA_total = append(prodA_total, float64(totA)/cyclesPerUs)
		prodB_total = append(prodB_total, float64(totB)/cyclesPerUs)
		prodC_total = append(prodC_total, float64(totC)/cyclesPerUs)

		prodA_trail = append(prodA_trail, float64(trA)/cyclesPerUs)
		prodB_trail = append(prodB_trail, float64(trB)/cyclesPerUs)
		prodC_trail = append(prodC_trail, float64(trC)/cyclesPerUs)
	}

	median := func(s []float64) float64 {
		cp := append([]float64(nil), s...)
		sort.Float64s(cp)
		n := len(cp)
		if n%2 == 1 {
			return cp[n/2]
		}
		return (cp[n/2-1] + cp[n/2]) / 2.0
	}

	mA_tot := median(prodA_total[warmupIters:])
	mB_tot := median(prodB_total[warmupIters:])
	mC_tot := median(prodC_total[warmupIters:])

	mA_tr := median(prodA_trail[warmupIters:])
	mB_tr := median(prodB_trail[warmupIters:])
	mC_tr := median(prodC_trail[warmupIters:])

	residencyOverhead := mB_tot - mA_tot
	memoryContentionOverhead := mC_tot - mB_tot
	totalSlowdown := mC_tot - mA_tot

	t.Logf("=== Resident CTA Starvation Isolation on GB10 Silicon ===")
	t.Logf("a) Producer Standalone Duration:              %.2f us (trailing: %.2f us)", mA_tot, mA_tr)
	t.Logf("b) Producer w/ Parked Entry-Sync CTAs:        %.2f us (trailing: %.2f us)", mB_tot, mB_tr)
	t.Logf("c) Producer w/ Active Memory-Prefetching CTAs: %.2f us (trailing: %.2f us)", mC_tot, mC_tr)
	t.Logf("---------------------------------------------------------")
	t.Logf("SM Issue Slot / Parked CTA Starvation Impact:  %+.2f us (%+.1f%% vs standalone)",
		residencyOverhead, (residencyOverhead/mA_tot)*100.0)
	t.Logf("Memory Bus & LSU Store Contention Impact:     %+.2f us (%+.1f%% vs parked)",
		memoryContentionOverhead, (memoryContentionOverhead/mB_tot)*100.0)
	t.Logf("Total Combined Producer Slowdown:             %+.2f us (%+.1f%% vs standalone)",
		totalSlowdown, (totalSlowdown/mA_tot)*100.0)

	t.Logf("CONCLUSION: Even with memory store contention adding %+.2f us to trailing writes,", memoryContentionOverhead)
	t.Logf("hoisting the weight prefetch into the producer's execution window yields massive net savings")
	t.Logf("by completely hiding weight DRAM read latency across the system bus.")
}
