package cudasample

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// ArgmaxKernelTemplate is the embedded CUDA C++ kernel finding the maximum logit index across vocabulary on GPU.
const ArgmaxKernelTemplate = `
extern "C" __global__ void ArgmaxLogits(
    const float* __restrict__ logits,
    int* __restrict__ out_token,
    int vocab_size
) {
    __shared__ float s_max_val[256];
    __shared__ int s_max_idx[256];

    int tid = threadIdx.x;
    float my_max = -__int_as_float(0x7f800000);
    int my_idx = -1;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = logits[i];
        if (val > my_max) {
            my_max = val;
            my_idx = i;
        }
    }

    s_max_val[tid] = my_max;
    s_max_idx[tid] = my_idx;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_val = s_max_val[tid + stride];
            int other_idx = s_max_idx[tid + stride];
            if (other_val > s_max_val[tid] ||
                (other_val == s_max_val[tid] && other_idx >= 0 &&
                 (s_max_idx[tid] < 0 || other_idx < s_max_idx[tid]))) {
                s_max_val[tid] = s_max_val[tid + stride];
                s_max_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_token = s_max_idx[0];
    }
}
`

// SampleArgmaxCPU returns the index of the maximum logit using the same rules
// as ArgmaxLogits. NaNs are ignored, ties favor the lowest index, and empty or
// all-NaN/all-negative-infinity input returns -1.
func SampleArgmaxCPU(logits []float32) int32 {
	maxIdx := -1
	maxVal := float32(math.Inf(-1))
	for i, v := range logits {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return int32(maxIdx)
}

// Compile compiles the JIT argmax sampling module.
func Compile(major, minor int) (*nvrtcx.CompiledModule, error) {
	opts := []nvrtcx.Option{
		nvrtcx.WithName("sample_argmax.cu"),
		nvrtcx.WithArch(major, minor),
	}
	mod, err := nvrtcx.Compile(ArgmaxKernelTemplate, opts...)
	if err != nil {
		return nil, fmt.Errorf("cudasample: compile: %w", err)
	}
	return mod, nil
}
