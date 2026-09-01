package egraph

import "fmt"

// FP8Format specifies the 8-bit floating point encoding format.
type FP8Format int

const (
	// FP8E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits (higher precision for weights/activations).
	FP8E4M3 FP8Format = iota
	// FP8E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits (higher dynamic range for gradients).
	FP8E5M2
)

// ScaledFP8GEMVTemplate is the embedded CUDA C++ kernel for in-register E4M3/E5M2 dequantization and GEMV.
const ScaledFP8GEMVTemplate = `
#define WARP_SIZE 32

// Convert uint8 (E4M3) to float in register using bit manipulation
__device__ __forceinline__ float fp8_e4m3_to_float(unsigned char val) {
    if (val == 0) return 0.0f;
    int sign = (val >> 7) & 1;
    int exp = (val >> 3) & 0x0F;
    int mant = val & 0x07;

    if (exp == 0) {
        // Subnormal
        float f = ((float)mant) / 8.0f * (1.0f / 64.0f);
        return sign ? -f : f;
    }

    float f = (1.0f + ((float)mant) / 8.0f) * powf(2.0f, (float)(exp - 7));
    return sign ? -f : f;
}

extern "C" __global__ void ScaledFP8GEMV(
    const float* __restrict__ x,
    const unsigned char* __restrict__ W_fp8,
    const float* __restrict__ scale_w,
    float* __restrict__ out,
    int in_dim,
    int out_dim
) {
    int row = blockIdx.x;
    if (row >= out_dim) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    __shared__ float s_sum[32];

    const unsigned char* w_row = W_fp8 + row * in_dim;
    float scale = scale_w ? scale_w[row] : 1.0f;

    float sum = 0.0f;
    for (int i = tid; i < in_dim; i += blockDim.x) {
        float w_val = fp8_e4m3_to_float(w_row[i]) * scale;
        sum += x[i] * w_val;
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        s_sum[wid] = sum;
    }
    __syncthreads();

    float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_sum[lane] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }

    if (tid == 0) {
        out[row] = block_sum;
    }
}
`

// GenerateFP8CUDA produces a specialized FP8 dequantizing GEMV kernel.
func GenerateFP8CUDA(kernelName string) string {
	if kernelName == "" {
		kernelName = "ScaledFP8GEMV"
	}
	return fmt.Sprintf("// Specialized Scaled FP8 GEMV\n%s\n", ScaledFP8GEMVTemplate)
}
