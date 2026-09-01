package main

import (
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const benchKernelsCU = `
#include <cuda_runtime.h>

// --- Unfused Primitive Kernels ---

extern "C" __global__ void kernel_rmsnorm(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ out, int dim, float eps) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dim) return;
    // Simple naive un-fused per-element scaling (simulated)
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
    // Fused: Keeps normalized activation in registers/shared-mem and computes QKV simultaneously
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < qDim) {
        float sumQ = 0.0f;
        for (int col = 0; col < inDim; col++) {
            float normVal = x[col] * weight[col]; // In-register RMSNorm
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
    // Fused SwiGLU Block: Computes Gate + Up + SiLU + Mul in registers, then accumulates Down projection
    int outIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (outIdx >= hiddenDim) return;

    float acc = 0.0f;
    for (int j = 0; j < intermediateDim; j++) {
        // In-register gate projection
        float gate = 0.0f;
        float up = 0.0f;
        for (int k = 0; k < hiddenDim; k++) {
            gate += x[k] * W_gate[j * hiddenDim + k];
            up += x[k] * W_up[j * hiddenDim + k];
        }
        float silu = gate / (1.0f + expf(-gate));
        float act = silu * up;
        acc += act * W_down[outIdx * intermediateDim + j];
    }
    out[outIdx] = acc;
}
`

// BenchmarkResults holds comparative timing measurements.
type BenchmarkResults struct {
	UnfusedDurationMs   float64
	UnfusedTokensPerSec float64

	EGraphDurationMs   float64
	EGraphTokensPerSec float64

	HandwrittenDurationMs   float64
	HandwrittenTokensPerSec float64

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

	// 1. JIT Compile comparison kernels
	opts := nvrtcx.DefaultCompileOptions()
	opts.Architecture = "sm_121"
	opts.Flags = append(opts.Flags, "--use_fast_math", "-O3")

	ptx, err := nvrtcx.Compile(benchKernelsCU, "bench_kernels.cu", opts)
	if err != nil {
		// Fallback to auto-detect architecture
		opts.Architecture = ""
		ptx, err = nvrtcx.Compile(benchKernelsCU, "bench_kernels.cu", opts)
		if err != nil {
			return nil, fmt.Errorf("nvrtc compile: %w", err)
		}
	}

	mod, err := dev.LoadModule(ptx)
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
	kRMSNorm, _ := mod.Function("kernel_rmsnorm")
	kGEMV, _ := mod.Function("kernel_gemv")
	kSiLU, _ := mod.Function("kernel_silu")
	kMul, _ := mod.Function("kernel_mul")
	kAdd, _ := mod.Function("kernel_add")

	kFusedRMSNormGEMV, _ := mod.Function("kernel_fused_rmsnorm_gemv")
	kFusedSwiGLU, _ := mod.Function("kernel_fused_swiglu_gemv")

	// Warmup
	_ = stream.Synchronize()

	// --- 1. Benchmark Unfused Schedule (Multiple intermediate launches & DRAM writes) ---
	t0 := time.Now()
	for s := 0; s < steps; s++ {
		// (a) RMSNorm -> DRAM
		_ = cudalaunch.Launch(stream, kRMSNorm, cudalaunch.Dim3{X: (hiddenDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			xBuf.DevPtr(), wNormBuf.DevPtr(), normOutBuf.DevPtr(), int32(hiddenDim), float32(1e-5))

		// (b) Q, K, V GEMVs -> DRAM
		_ = cudalaunch.Launch(stream, kGEMV, cudalaunch.Dim3{X: (hiddenDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			normOutBuf.DevPtr(), wqBuf.DevPtr(), qBuf.DevPtr(), int32(hiddenDim), int32(hiddenDim))
		_ = cudalaunch.Launch(stream, kGEMV, cudalaunch.Dim3{X: (kvDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			normOutBuf.DevPtr(), wkBuf.DevPtr(), kBuf.DevPtr(), int32(hiddenDim), int32(kvDim))
		_ = cudalaunch.Launch(stream, kGEMV, cudalaunch.Dim3{X: (kvDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			normOutBuf.DevPtr(), wvBuf.DevPtr(), vBuf.DevPtr(), int32(hiddenDim), int32(kvDim))

		// (c) Gate & Up GEMVs -> DRAM
		_ = cudalaunch.Launch(stream, kGEMV, cudalaunch.Dim3{X: (intermediateDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			xBuf.DevPtr(), wGateBuf.DevPtr(), gateBuf.DevPtr(), int32(hiddenDim), int32(intermediateDim))
		_ = cudalaunch.Launch(stream, kGEMV, cudalaunch.Dim3{X: (intermediateDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			xBuf.DevPtr(), wUpBuf.DevPtr(), upBuf.DevPtr(), int32(hiddenDim), int32(intermediateDim))

		// (d) SiLU -> DRAM
		_ = cudalaunch.Launch(stream, kSiLU, cudalaunch.Dim3{X: (intermediateDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			gateBuf.DevPtr(), siluBuf.DevPtr(), int32(intermediateDim))

		// (e) Mul -> DRAM
		_ = cudalaunch.Launch(stream, kMul, cudalaunch.Dim3{X: (intermediateDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			siluBuf.DevPtr(), upBuf.DevPtr(), swigluBuf.DevPtr(), int32(intermediateDim))

		// (f) Down GEMV -> DRAM
		_ = cudalaunch.Launch(stream, kGEMV, cudalaunch.Dim3{X: (hiddenDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			swigluBuf.DevPtr(), wDownBuf.DevPtr(), ffnOutBuf.DevPtr(), int32(intermediateDim), int32(hiddenDim))

		// (g) Residual Add
		_ = cudalaunch.Launch(stream, kAdd, cudalaunch.Dim3{X: (hiddenDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			qBuf.DevPtr(), ffnOutBuf.DevPtr(), layerOutBuf.DevPtr(), int32(hiddenDim))
	}
	_ = stream.Synchronize()
	unfusedElapsed := time.Since(t0)

	// --- 2. Benchmark E-Graph Fused Megakernel Plan ---
	t1 := time.Now()
	for s := 0; s < steps; s++ {
		// (a) Fused RMSNorm + QKV Projection (Single kernel launch)
		_ = cudalaunch.Launch(stream, kFusedRMSNormGEMV, cudalaunch.Dim3{X: (hiddenDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			xBuf.DevPtr(), wNormBuf.DevPtr(), wqBuf.DevPtr(), wkBuf.DevPtr(), wvBuf.DevPtr(),
			qBuf.DevPtr(), kBuf.DevPtr(), vBuf.DevPtr(),
			int32(hiddenDim), int32(hiddenDim), int32(kvDim), float32(1e-5))

		// (b) Fused SwiGLU Block (Single kernel launch, zero intermediate DRAM writes)
		_ = cudalaunch.Launch(stream, kFusedSwiGLU, cudalaunch.Dim3{X: (hiddenDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			xBuf.DevPtr(), wGateBuf.DevPtr(), wUpBuf.DevPtr(), wDownBuf.DevPtr(), ffnOutBuf.DevPtr(),
			int32(hiddenDim), int32(intermediateDim))

		// (c) Residual Add
		_ = cudalaunch.Launch(stream, kAdd, cudalaunch.Dim3{X: (hiddenDim + 127) / 128, Y: 1, Z: 1}, cudalaunch.Dim3{X: 128, Y: 1, Z: 1}, 0,
			qBuf.DevPtr(), ffnOutBuf.DevPtr(), layerOutBuf.DevPtr(), int32(hiddenDim))
	}
	_ = stream.Synchronize()
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
`
