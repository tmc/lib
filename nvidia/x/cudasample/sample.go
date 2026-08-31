package cudasample

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Sampler defines an interface for sampling a token index from a distribution of logits.
type Sampler interface {
	Sample(logits []float32) int32
}

// Config configures token sampling parameters.
type Config struct {
	Temperature float32
	TopP        float32
	TopK        int
}

// ArgmaxSampler selects the token with the highest logit value (greedy decoding).
type ArgmaxSampler struct{}

func (ArgmaxSampler) Sample(logits []float32) int32 {
	return SampleArgmaxCPU(logits)
}

// TemperatureSampler scales logits and performs categorical sampling.
type TemperatureSampler struct {
	Temperature float32
	TopK        int
	TopP        float32
}

func (s TemperatureSampler) Sample(logits []float32) int32 {
	if len(logits) == 0 {
		return -1
	}
	if s.Temperature <= 0 || (s.TopK == 1) {
		return SampleArgmaxCPU(logits)
	}

	type item struct {
		idx int32
		val float32
	}
	items := make([]item, len(logits))
	for i, v := range logits {
		items[i] = item{idx: int32(i), val: v / s.Temperature}
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].val > items[j].val
	})

	if s.TopK > 0 && s.TopK < len(items) {
		items = items[:s.TopK]
	}

	// Softmax
	maxVal := items[0].val
	var sumExp float32
	for i := range items {
		e := float32(math.Exp(float64(items[i].val - maxVal)))
		items[i].val = e
		sumExp += e
	}

	// Top-P filter
	if s.TopP > 0 && s.TopP < 1.0 {
		var cumSum float32
		cutoff := len(items)
		for i, it := range items {
			cumSum += it.val / sumExp
			if cumSum >= s.TopP {
				cutoff = i + 1
				break
			}
		}
		items = items[:cutoff]
		sumExp = 0
		for _, it := range items {
			sumExp += it.val
		}
	}

	// Sample from categorical distribution
	r := rand.Float32() * sumExp
	var acc float32
	for _, it := range items {
		acc += it.val
		if r <= acc {
			return it.idx
		}
	}
	return items[0].idx
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
