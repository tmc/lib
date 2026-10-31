#include <cuda_bf16.h>

// These baseline kernels store BF16-valued activations in float32 buffers.
// A scalar/tree reduction is not a bitwise identity claim for a library GEMM.
__device__ float afmoe_core_round(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ float afmoe_core_weight(unsigned short x) {
    return __uint_as_float(unsigned(x) << 16);
}

extern "C" __global__ void AfmoeEmbedding(
    const unsigned short* table, float* out, int token, int hidden, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden)
        out[i] = afmoe_core_round(__fmul_rn(afmoe_core_weight(table[(unsigned long long)token * hidden + i]), scale));
}

extern "C" __global__ void AfmoeLinear(
    const unsigned short* weight, const float* x, float* out,
    int rows, int cols, int round_output) {
    int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        sum = __fadd_rn(sum, __fmul_rn(afmoe_core_weight(weight[(unsigned long long)row * cols + col]), x[col]));
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[row] = round_output ? afmoe_core_round(partial[0]) : partial[0];
}

extern "C" __global__ void AfmoeResidual(
    float* out, const float* x, const float* y, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) out[i] = afmoe_core_round(__fadd_rn(x[i], y[i]));
}

extern "C" __global__ void AfmoeArgmax(
    const float* logits, int* token, int width) {
    __shared__ float values[256];
    __shared__ int indices[256];
    __shared__ int invalid[256];
    float best = -__int_as_float(0x7f800000);
    int index = width;
    int bad = 0;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = logits[i];
        bad |= !isfinite(v);
        if (v > best || (v == best && i < index)) {
            best = v;
            index = i;
        }
    }
    values[threadIdx.x] = best;
    indices[threadIdx.x] = index;
    invalid[threadIdx.x] = bad;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            int other = threadIdx.x + stride;
            if (values[other] > values[threadIdx.x] ||
                (values[other] == values[threadIdx.x] && indices[other] < indices[threadIdx.x])) {
                values[threadIdx.x] = values[other];
                indices[threadIdx.x] = indices[other];
            }
            invalid[threadIdx.x] |= invalid[other];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) token[0] = invalid[0] ? -1 : indices[0];
}
