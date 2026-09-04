package cudalayer

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// ErrInvalidConfig reports a decoder-layer shape the kernels cannot execute.
var ErrInvalidConfig = errors.New("cudalayer: invalid configuration")

// ProjectionNormConfig describes a projection, residual addition, and RMS
// normalization over packed token rows.
type ProjectionNormConfig struct {
	Dim     int
	Threads int
	Epsilon float32
}

// Validate checks the kernel shape and numerical invariants.
func (c ProjectionNormConfig) Validate() error {
	if c.Dim <= 0 || c.Threads <= 0 || c.Threads > 1024 || c.Threads&(c.Threads-1) != 0 ||
		!(c.Epsilon > 0) || math.IsNaN(float64(c.Epsilon)) || math.IsInf(float64(c.Epsilon), 0) {
		return ErrInvalidConfig
	}
	return nil
}

// SharedMemoryBytes returns the reduction storage required by ProjectionResidualRMSNorm.
func (c ProjectionNormConfig) SharedMemoryBytes() (int, error) {
	if err := c.Validate(); err != nil {
		return 0, err
	}
	return 4 * c.Threads, nil
}

// ProjectionResidualRMSNormTemplate is the CUDA source for one complete
// attention-output projection, residual addition, and RMS normalization stage.
const ProjectionResidualRMSNormTemplate = `
extern "C" __global__ void ProjectionResidualRMSNorm(
    const float* __restrict__ attention,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    const float* __restrict__ norm_weight,
    const int* __restrict__ token_rows,
    const int* __restrict__ row_status,
    float* __restrict__ output,
    float* __restrict__ normalized,
    int dim,
    int num_tokens,
    int metadata_rows,
    float epsilon
) {
    int token = blockIdx.x;
    if (token >= num_tokens) return;
    int row = token_rows[token];
    bool valid = row >= 0 && row < metadata_rows && row_status[row] == 0 &&
        dim > 0 && epsilon > 0.0f;
    if (!valid) {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            output[token * dim + i] = nanf("");
            normalized[token * dim + i] = nanf("");
        }
        return;
    }

    const float* a = attention + token * dim;
    const float* x = residual + token * dim;
    float* y = output + token * dim;
    for (int out = threadIdx.x; out < dim; out += blockDim.x) {
        const float* w = weight + out * dim;
        float sum = x[out];
        for (int in = 0; in < dim; in++) sum += a[in] * w[in];
        y[out] = sum;
    }
    __syncthreads();

    extern __shared__ float reduce[];
    float squares = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) squares += y[i] * y[i];
    reduce[threadIdx.x] = squares;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(reduce[0] / (float)dim + epsilon);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        normalized[token * dim + i] = y[i] * scale * norm_weight[i];
    }
}
`

// Compile builds the decoder-layer JIT module for a CUDA architecture.
func Compile(major, minor int) (*nvrtcx.CompiledModule, error) {
	if major <= 0 || minor < 0 {
		return nil, ErrInvalidConfig
	}
	module, err := nvrtcx.Compile(ProjectionResidualRMSNormTemplate,
		nvrtcx.WithName("decoder_layer.cu"), nvrtcx.WithArch(major, minor))
	if err != nil {
		return nil, fmt.Errorf("cudalayer: compile: %w", err)
	}
	return module, nil
}
