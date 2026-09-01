package mxfp4

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const mxfp4KernelsCU = `
// --- OCP Microscaling FP4 (E2M1) CUDA Dequantization and Fused GEMV ---

__constant__ float c_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// MXFP4Dequantize unpacks packed uint8 pairs and scales into full float32 output.
extern "C" __global__ void MXFP4Dequantize(
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    float* __restrict__ out,
    int totalElements,
    int groupSize
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; // index of pair (2 elements)
    if (idx * 2 >= totalElements) return;

    unsigned char byteVal = packed[idx];
    int groupIdx = (idx * 2) / groupSize;
    float scale = scales[groupIdx];

    int n0 = byteVal & 0x0F;
    int n1 = (byteVal >> 4) & 0x0F;

    out[idx * 2]     = c_e2m1_lut[n0] * scale;
    out[idx * 2 + 1] = c_e2m1_lut[n1] * scale;
}

// MXFP4LinearGEMV computes y = W_mxfp4 * x in a single high-throughput fused pass.
extern "C" __global__ void MXFP4LinearGEMV(
    const float* __restrict__ x,
    const unsigned char* __restrict__ packedW,
    const float* __restrict__ scales,
    float* __restrict__ outY,
    int inDim,
    int outDim,
    int groupSize
) {
    int row = blockIdx.x; // each thread block handles one output row
    if (row >= outDim) return;

    int tid = threadIdx.x;
    int pairsPerRow = inDim / 2;
    int groupsPerRow = inDim / groupSize;
    const unsigned char* rowPacked = packedW + row * pairsPerRow;
    const float* rowScales = scales + row * groupsPerRow;

    float localSum = 0.0f;
    for (int p = tid; p < pairsPerRow; p += blockDim.x) {
        unsigned char b = rowPacked[p];
        int g = (p * 2) / groupSize;
        float scale = rowScales[g];

        int n0 = b & 0x0F;
        int n1 = (b >> 4) & 0x0F;

        float w0 = c_e2m1_lut[n0] * scale;
        float w1 = c_e2m1_lut[n1] * scale;

        localSum += x[p * 2] * w0 + x[p * 2 + 1] * w1;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    }

    __shared__ float s_warp_sums[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) {
        s_warp_sums[wid] = localSum;
    }
    __syncthreads();

    if (wid == 0) {
        float val = (tid < (blockDim.x / 32)) ? s_warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            outY[row] = val;
        }
    }
}
`

// Engine manages compiled GPU functions for MXFP4 operations.
type Engine struct {
	mod          driver.Module
	dequantFn    driver.Function
	fusedGEMVFn  driver.Function
}

// NewEngine compiles and loads MXFP4 GPU kernels on dev.
func NewEngine(dev *cuda.Device) (*Engine, error) {
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("mxfp4: compute capability: %w", err)
	}

	opts := []nvrtcx.Option{
		nvrtcx.WithName("mxfp4_kernels.cu"),
		nvrtcx.WithArch(major, minor),
	}
	compiled, err := nvrtcx.Compile(mxfp4KernelsCU, opts...)
	if err != nil {
		return nil, fmt.Errorf("mxfp4: compile kernels: %w", err)
	}

	loaded, err := compiled.Load(dev)
	if err != nil {
		return nil, fmt.Errorf("mxfp4: load module: %w", err)
	}

	dequantFn, _ := loaded.GetFunction("MXFP4Dequantize")
	fusedGEMVFn, _ := loaded.GetFunction("MXFP4LinearGEMV")

	return &Engine{
		mod:         loaded,
		dequantFn:   dequantFn,
		fusedGEMVFn: fusedGEMVFn,
	}, nil
}

// LinearGEMV launches fused MXFP4 matrix-vector multiplication on the GPU.
func (e *Engine) LinearGEMV(
	x cudamem.Slice[float32],
	packed cudamem.Slice[uint8],
	scales cudamem.Slice[float32],
	outY cudamem.Slice[float32],
	inDim, outDim, groupSize int,
) error {
	if groupSize <= 0 {
		groupSize = DefaultGroupSize
	}
	return cudalaunch.Kernel(e.fusedGEMVFn).
		Grid(uint32(outDim)).Block(128).
		Launch(x, packed, scales, outY, int32(inDim), int32(outDim), int32(groupSize))
}

// Dequantize launches GPU-side dequantization from packed uint8 + scales into float32 destination.
func (e *Engine) Dequantize(
	packed cudamem.Slice[uint8],
	scales cudamem.Slice[float32],
	out cudamem.Slice[float32],
	totalElements, groupSize int,
) error {
	if groupSize <= 0 {
		groupSize = DefaultGroupSize
	}
	numPairs := (totalElements + 1) / 2
	return cudalaunch.Kernel(e.dequantFn).
		Grid(uint32((numPairs + 127) / 128)).Block(128).
		Launch(packed, scales, out, int32(totalElements), int32(groupSize))
}

// Close releases the loaded CUDA module.
func (e *Engine) Close() error {
	if e.mod != 0 {
		return e.mod.Unload()
	}
	return nil
}
