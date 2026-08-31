package cudasample

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Config configures token sampling parameters.
type Config struct {
	Temperature float32
	TopP        float32
	TopK        int
}

// ArgmaxKernelTemplate is the embedded CUDA C++ kernel finding the maximum logit index across vocabulary on GPU.
const ArgmaxKernelTemplate = `
#include <cuda_runtime.h>

extern "C" __global__ void ArgmaxLogits(
    const float* __restrict__ logits,
    int* __restrict__ out_token,
    int vocab_size
) {
    __shared__ float s_max_val[256];
    __shared__ int s_max_idx[256];

    int tid = threadIdx.x;
    float my_max = -1e20f;
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
            if (s_max_val[tid + stride] > s_max_val[tid]) {
                s_max_val[tid] = s_max_val[tid + stride];
                s_max_idx[tid] = s_max_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_token = s_max_idx[0];
    }
}
`

// SampleArgmaxCPU provides a CPU reference implementation of greedy argmax token selection.
func SampleArgmaxCPU(logits []float32) int32 {
	if len(logits) == 0 {
		return -1
	}
	maxIdx := 0
	maxVal := logits[0]
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
