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

// ProjectionResidualRMSNormTemplate is the CUDA source for the shared-weight
// projections and normalization stages of one packed decoder layer.
const ProjectionResidualRMSNormTemplate = `
extern "C" __global__ void RMSNormQKV(
    const float* __restrict__ input,
    const float* __restrict__ norm_weight,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ v_weight,
    const int* __restrict__ token_rows,
    const unsigned long long* __restrict__ row_generations,
    const unsigned long long* __restrict__ token_generations,
    int* __restrict__ row_status,
    float* __restrict__ normalized,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int dim,
    int q_dim,
    int kv_dim,
    int num_tokens,
    int metadata_rows,
    float epsilon
) {
    int token = blockIdx.x;
    if (token >= num_tokens) return;
    int row = token_rows[token];
    bool valid_row = row >= 0 && row < metadata_rows;
    bool valid = valid_row && dim > 0 && q_dim > 0 && kv_dim > 0 && epsilon > 0.0f &&
        token_generations[token] != 0 && token_generations[token] == row_generations[row];
    if (!valid) {
        if (threadIdx.x == 0 && metadata_rows > 0) atomicExch(row_status + (valid_row ? row : 0), 1);
        for (int i = threadIdx.x; i < dim; i += blockDim.x) normalized[token * dim + i] = nanf("");
        for (int i = threadIdx.x; i < q_dim; i += blockDim.x) q[token * q_dim + i] = nanf("");
        for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
            k[token * kv_dim + i] = nanf("");
            v[token * kv_dim + i] = nanf("");
        }
        return;
    }

    extern __shared__ float reduce[];
    const float* x = input + token * dim;
    float squares = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) squares += x[i] * x[i];
    reduce[threadIdx.x] = squares;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(reduce[0] / (float)dim + epsilon);
    float* n = normalized + token * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) n[i] = x[i] * scale * norm_weight[i];
    __syncthreads();
    for (int out = threadIdx.x; out < q_dim; out += blockDim.x) {
        float sum = 0.0f;
        for (int in = 0; in < dim; in++) sum += n[in] * q_weight[out * dim + in];
        q[token * q_dim + out] = sum;
    }
    for (int out = threadIdx.x; out < kv_dim; out += blockDim.x) {
        float sk = 0.0f, sv = 0.0f;
        for (int in = 0; in < dim; in++) {
            float value = n[in];
            sk += value * k_weight[out * dim + in];
            sv += value * v_weight[out * dim + in];
        }
        k[token * kv_dim + out] = sk;
        v[token * kv_dim + out] = sv;
    }
}

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

extern "C" __global__ void GateUpSiLU(
    const float* __restrict__ normalized,
    const float* __restrict__ gate_weight,
    const float* __restrict__ up_weight,
    const int* __restrict__ token_rows,
    const int* __restrict__ row_status,
    float* __restrict__ gate,
    float* __restrict__ up,
    float* __restrict__ activation,
    int dim,
    int intermediate_dim,
    int num_tokens,
    int metadata_rows
) {
    int token = blockIdx.x;
    if (token >= num_tokens) return;
    int row = token_rows[token];
    bool valid = row >= 0 && row < metadata_rows && row_status[row] == 0 &&
        dim > 0 && intermediate_dim > 0;
    const float* x = normalized + token * dim;
    for (int out = threadIdx.x; out < intermediate_dim; out += blockDim.x) {
        float g = 0.0f, u = 0.0f;
        if (valid) {
            const float* gw = gate_weight + out * dim;
            const float* uw = up_weight + out * dim;
            for (int in = 0; in < dim; in++) {
                g += x[in] * gw[in];
                u += x[in] * uw[in];
            }
        } else {
            g = nanf("");
            u = nanf("");
        }
        gate[token * intermediate_dim + out] = g;
        up[token * intermediate_dim + out] = u;
        activation[token * intermediate_dim + out] = g / (1.0f + expf(-g)) * u;
    }
}

extern "C" __global__ void DownResidual(
    const float* __restrict__ activation,
    const float* __restrict__ residual,
    const float* __restrict__ down_weight,
    const int* __restrict__ token_rows,
    const int* __restrict__ row_status,
    float* __restrict__ output,
    int dim,
    int intermediate_dim,
    int num_tokens,
    int metadata_rows
) {
    int token = blockIdx.x;
    if (token >= num_tokens) return;
    int row = token_rows[token];
    bool valid = row >= 0 && row < metadata_rows && row_status[row] == 0 &&
        dim > 0 && intermediate_dim > 0;
    for (int out = threadIdx.x; out < dim; out += blockDim.x) {
        float sum = valid ? residual[token * dim + out] : nanf("");
        if (valid) {
            const float* w = down_weight + out * intermediate_dim;
            const float* a = activation + token * intermediate_dim;
            for (int in = 0; in < intermediate_dim; in++) sum += a[in] * w[in];
        }
        output[token * dim + out] = sum;
    }
}
`

// Compile builds the decoder-layer JIT module for a CUDA architecture.
func Compile(major, minor int) (*nvrtcx.CompiledModule, error) {
	if major <= 0 || minor < 0 {
		return nil, ErrInvalidConfig
	}
	module, err := nvrtcx.Compile(ProjectionResidualRMSNormTemplate,
		nvrtcx.WithName("decoder_layer.cu"), nvrtcx.WithPhysicalArch(major, minor))
	if err != nil {
		return nil, fmt.Errorf("cudalayer: compile: %w", err)
	}
	return module, nil
}
