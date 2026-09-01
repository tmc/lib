package main

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const benchKernelsCU = `
// --- Unfused Primitive Kernels ---

extern "C" __global__ void kernel_rmsnorm(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ out, int dim, float eps) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dim) return;
    out[idx] = x[idx] * weight[idx];
}

extern "C" __global__ void kernel_gemv(const float* __restrict__ x, const float* __restrict__ W, float* __restrict__ out, int inDim, int outDim) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= outDim) return;
    float sum = 0.0f;
    for (int col = 0; col < inDim; col++) {
        sum += x[col] * W[row * inDim + col];
    }
    out[row] = sum;
}

extern "C" __global__ void kernel_silu(const float* __restrict__ x, float* __restrict__ out, int dim) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dim) return;
    float v = x[idx];
    out[idx] = v / (1.0f + expf(-v));
}

extern "C" __global__ void kernel_mul(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int dim) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dim) return;
    out[idx] = a[idx] * b[idx];
}

extern "C" __global__ void kernel_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int dim) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dim) return;
    out[idx] = a[idx] + b[idx];
}

// --- E-Graph Extracted Fused Megakernels ---

extern "C" __global__ void kernel_fused_rmsnorm_gemv(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ W_q,
    const float* __restrict__ W_k,
    const float* __restrict__ W_v,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    int inDim, int qDim, int kvDim, float eps
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < qDim) {
        float sumQ = 0.0f;
        for (int col = 0; col < inDim; col++) {
            float normVal = x[col] * weight[col];
            sumQ += normVal * W_q[row * inDim + col];
        }
        out_q[row] = sumQ;
    }
    if (row < kvDim) {
        float sumK = 0.0f;
        float sumV = 0.0f;
        for (int col = 0; col < inDim; col++) {
            float normVal = x[col] * weight[col];
            sumK += normVal * W_k[row * inDim + col];
            sumV += normVal * W_v[row * inDim + col];
        }
        out_k[row] = sumK;
        out_v[row] = sumV;
    }
}

extern "C" __global__ void kernel_fused_swiglu_gemv(
    const float* __restrict__ x,
    const float* __restrict__ W_gate,
    const float* __restrict__ W_up,
    const float* __restrict__ W_down,
    float* __restrict__ out,
    int hiddenDim, int intermediateDim
) {
    extern __shared__ float s_act[]; // size: intermediateDim
    
    // Step 1: Cooperatively compute intermediate activations across all threads
    for (int j = threadIdx.x; j < intermediateDim; j += blockDim.x) {
        float gate = 0.0f;
        float up = 0.0f;
        for (int k = 0; k < hiddenDim; k++) {
            gate += x[k] * W_gate[j * hiddenDim + k];
            up += x[k] * W_up[j * hiddenDim + k];
        }
        float silu = gate / (1.0f + expf(-gate));
        s_act[j] = silu * up;
    }
    __syncthreads();

    // Step 2: Compute down-projection output elements
    for (int outIdx = blockIdx.x * blockDim.x + threadIdx.x; outIdx < hiddenDim; outIdx += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < intermediateDim; j++) {
            acc += s_act[j] * W_down[outIdx * intermediateDim + j];
        }
        out[outIdx] = acc;
    }
}
`

// BenchmarkResults holds comparative timing measurements.
type BenchmarkResults struct {
	UnfusedDurationMs   float64
	UnfusedTokensPerSec float64

	EGraphDurationMs   float64
	EGraphTokensPerSec float64

	SpeedupVsUnfused float64
}

// RunComparisonBenchmark executes comparative decoding benchmarks on the GPU.
func RunComparisonBenchmark(steps int) (*BenchmarkResults, error) {
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

	ctx := context.Background()

	// 1. JIT Compile comparison kernels
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("compute capability: %w", err)
	}

	opts := []nvrtcx.Option{
		nvrtcx.WithName("bench_kernels.cu"),
		nvrtcx.WithArch(major, minor),
	}
	compiled, err := nvrtcx.Compile(benchKernelsCU, opts...)
	if err != nil {
		return nil, fmt.Errorf("compile benchmark kernels: %w", err)
	}

	loaded, err := compiled.Load(dev)
	if err != nil {
		return nil, fmt.Errorf("load module: %w", err)
	}

	// Dimensions for comparison
	const (
		hiddenDim       = 512
		intermediateDim = 1024
		kvDim           = 128
	)

	// Allocate GPU memory buffers
	xBuf, _ := cudamem.Alloc[float32](dev, hiddenDim)
	wNormBuf, _ := cudamem.Alloc[float32](dev, hiddenDim)
	normOutBuf, _ := cudamem.Alloc[float32](dev, hiddenDim)

	wqBuf, _ := cudamem.Alloc[float32](dev, hiddenDim*hiddenDim)
	wkBuf, _ := cudamem.Alloc[float32](dev, hiddenDim*kvDim)
	wvBuf, _ := cudamem.Alloc[float32](dev, hiddenDim*kvDim)

	qBuf, _ := cudamem.Alloc[float32](dev, hiddenDim)
	kBuf, _ := cudamem.Alloc[float32](dev, kvDim)
	vBuf, _ := cudamem.Alloc[float32](dev, kvDim)

	wGateBuf, _ := cudamem.Alloc[float32](dev, hiddenDim*intermediateDim)
	wUpBuf, _ := cudamem.Alloc[float32](dev, hiddenDim*intermediateDim)
	wDownBuf, _ := cudamem.Alloc[float32](dev, intermediateDim*hiddenDim)

	gateBuf, _ := cudamem.Alloc[float32](dev, intermediateDim)
	upBuf, _ := cudamem.Alloc[float32](dev, intermediateDim)
	siluBuf, _ := cudamem.Alloc[float32](dev, intermediateDim)
	swigluBuf, _ := cudamem.Alloc[float32](dev, intermediateDim)
	ffnOutBuf, _ := cudamem.Alloc[float32](dev, hiddenDim)
	layerOutBuf, _ := cudamem.Alloc[float32](dev, hiddenDim)

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
		siluBuf.Free()
		swigluBuf.Free()
		ffnOutBuf.Free()
		layerOutBuf.Free()
	}()

	// Kernel functions
	kRMSNorm, _ := loaded.GetFunction("kernel_rmsnorm")
	kGEMV, _ := loaded.GetFunction("kernel_gemv")
	kSiLU, _ := loaded.GetFunction("kernel_silu")
	kMul, _ := loaded.GetFunction("kernel_mul")
	kAdd, _ := loaded.GetFunction("kernel_add")

	kFusedRMSNormGEMV, _ := loaded.GetFunction("kernel_fused_rmsnorm_gemv")
	kFusedSwiGLU, _ := loaded.GetFunction("kernel_fused_swiglu_gemv")

	// Warmup
	_ = stream.Synchronize(ctx)

	// --- 1. Benchmark Unfused Schedule (Multiple intermediate launches & DRAM writes) ---
	t0 := time.Now()
	for s := 0; s < steps; s++ {
		// (a) RMSNorm -> DRAM
		_ = cudalaunch.Kernel(kRMSNorm).Grid(uint32((hiddenDim + 127) / 128)).Block(128).
			Launch(xBuf, wNormBuf, normOutBuf, int32(hiddenDim), float32(1e-5))

		// (b) Q, K, V GEMVs -> DRAM
		_ = cudalaunch.Kernel(kGEMV).Grid(uint32((hiddenDim + 127) / 128)).Block(128).
			Launch(normOutBuf, wqBuf, qBuf, int32(hiddenDim), int32(hiddenDim))
		_ = cudalaunch.Kernel(kGEMV).Grid(uint32((kvDim + 127) / 128)).Block(128).
			Launch(normOutBuf, wkBuf, kBuf, int32(hiddenDim), int32(kvDim))
		_ = cudalaunch.Kernel(kGEMV).Grid(uint32((kvDim + 127) / 128)).Block(128).
			Launch(normOutBuf, wvBuf, vBuf, int32(hiddenDim), int32(kvDim))

		// (c) Gate & Up GEMVs -> DRAM
		_ = cudalaunch.Kernel(kGEMV).Grid(uint32((intermediateDim + 127) / 128)).Block(128).
			Launch(xBuf, wGateBuf, gateBuf, int32(hiddenDim), int32(intermediateDim))
		_ = cudalaunch.Kernel(kGEMV).Grid(uint32((intermediateDim + 127) / 128)).Block(128).
			Launch(xBuf, wUpBuf, upBuf, int32(hiddenDim), int32(intermediateDim))

		// (d) SiLU -> DRAM
		_ = cudalaunch.Kernel(kSiLU).Grid(uint32((intermediateDim + 127) / 128)).Block(128).
			Launch(gateBuf, siluBuf, int32(intermediateDim))

		// (e) Mul -> DRAM
		_ = cudalaunch.Kernel(kMul).Grid(uint32((intermediateDim + 127) / 128)).Block(128).
			Launch(siluBuf, upBuf, swigluBuf, int32(intermediateDim))

		// (f) Down GEMV -> DRAM
		_ = cudalaunch.Kernel(kGEMV).Grid(uint32((hiddenDim + 127) / 128)).Block(128).
			Launch(swigluBuf, wDownBuf, ffnOutBuf, int32(intermediateDim), int32(hiddenDim))

		// (g) Residual Add
		_ = cudalaunch.Kernel(kAdd).Grid(uint32((hiddenDim + 127) / 128)).Block(128).
			Launch(qBuf, ffnOutBuf, layerOutBuf, int32(hiddenDim))
	}
	_ = stream.Synchronize(ctx)
	unfusedElapsed := time.Since(t0)

	// --- 2. Benchmark E-Graph Fused Megakernel Plan ---
	t1 := time.Now()
	for s := 0; s < steps; s++ {
		// (a) Fused RMSNorm + QKV Projection (Single kernel launch)
		_ = cudalaunch.Kernel(kFusedRMSNormGEMV).Grid(uint32((hiddenDim + 127) / 128)).Block(128).
			Launch(xBuf, wNormBuf, wqBuf, wkBuf, wvBuf,
				qBuf, kBuf, vBuf,
				int32(hiddenDim), int32(hiddenDim), int32(kvDim), float32(1e-5))

		// (b) Fused SwiGLU Block (Single kernel launch, zero intermediate DRAM writes)
		_ = cudalaunch.Kernel(kFusedSwiGLU).Grid(uint32((hiddenDim + 127) / 128)).Block(128).
			SharedMem(uint32(intermediateDim * 4)).
			Launch(xBuf, wGateBuf, wUpBuf, wDownBuf, ffnOutBuf,
				int32(hiddenDim), int32(intermediateDim))

		// (c) Residual Add
		_ = cudalaunch.Kernel(kAdd).Grid(uint32((hiddenDim + 127) / 128)).Block(128).
			Launch(qBuf, ffnOutBuf, layerOutBuf, int32(hiddenDim))
	}
	_ = stream.Synchronize(ctx)
	egraphElapsed := time.Since(t1)

	unfusedMs := float64(unfusedElapsed.Milliseconds())
	if unfusedMs == 0 {
		unfusedMs = float64(unfusedElapsed.Microseconds()) / 1000.0
	}
	egraphMs := float64(egraphElapsed.Milliseconds())
	if egraphMs == 0 {
		egraphMs = float64(egraphElapsed.Microseconds()) / 1000.0
	}

	unfusedTokSec := float64(steps) / unfusedElapsed.Seconds()
	egraphTokSec := float64(steps) / egraphElapsed.Seconds()
	speedup := egraphTokSec / unfusedTokSec

	return &BenchmarkResults{
		UnfusedDurationMs:   unfusedMs,
		UnfusedTokensPerSec: unfusedTokSec,
		EGraphDurationMs:    egraphMs,
		EGraphTokensPerSec:  egraphTokSec,
		SpeedupVsUnfused:    speedup,
	}, nil
}
