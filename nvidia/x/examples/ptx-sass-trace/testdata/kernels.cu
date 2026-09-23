// NVRTC compiles device code without system headers, so under NVRTC the
// fixed-width types used below are defined here. Offline nvcc builds of this
// file, which the capture bundle supports, still get the real <stdint.h>.
#if defined(__CUDACC_RTC__)
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef short int16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#endif

#define FULL_WARP_MASK 0xffffffff

// Device helper: half-to-float conversion using native PTX instruction.
__device__ __forceinline__ float half_to_float(uint16_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// Device helper: SIMD 4-way 8-bit dot product using native PTX instruction.
__device__ __forceinline__ int dp4a_simd(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#else
    return __dp4a(a, b, c);
#endif
}

// Q8_0 block: 32 signed 8-bit quantized values with a 16-bit half scale.
struct block_q8_0 {
    uint16_t d;        // float16 scale
    int8_t qs[32];     // 32 int8 quantized values
};

// 1. Quantized Matvec with SIMD & Warp Reductions (ds4_q8_0_matvec).
extern "C" __global__ void ds4_q8_0_matvec(
    const block_q8_0 *W,
    const float *x,
    float *y,
    int rows,
    int cols
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= rows) return;

    int row = warp_id;
    int blocks_per_row = cols / 32;
    float row_sum = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        const block_q8_0 *blk = &W[row * blocks_per_row + b];
        float scale = half_to_float(blk->d);
        float q = (float)blk->qs[lane];
        float xv = x[b * 32 + lane];
        float prod = q * xv;

        for (int offset = 16; offset > 0; offset /= 2) {
            prod += __shfl_down_sync(FULL_WARP_MASK, prod, offset);
        }
        if (lane == 0) {
            row_sum += prod * scale;
        }
    }
    if (lane == 0) {
        y[row] = row_sum;
    }
}

// 2a. Sub-byte dot product with SIMD __dp4a (ds4_iq2_simd).
extern "C" __global__ void ds4_iq2_simd(
    const uint32_t *q,
    const int8_t *a,
    float *out,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t qval = q[idx];
    const uint32_t *a_u32 = reinterpret_cast<const uint32_t*>(a);
    int sum = 0;

    #pragma unroll
    for (int g = 0; g < 4; g++) {
        uint32_t byte_g = (qval >> (8 * g)) & 0xFF;
        uint32_t rep = byte_g * 0x01010101;
        uint32_t mask_shifts = ((rep >> 0) & 0x03) |
                              (((rep >> 2) & 0x03) << 8) |
                              (((rep >> 4) & 0x03) << 16) |
                              (((rep >> 6) & 0x03) << 24);
        sum = dp4a_simd(mask_shifts, a_u32[idx * 4 + g], sum);
    }
    out[idx] = (float)sum * scale;
}

// 2b. Sub-byte dot product with scalar bit-shift loop (ds4_iq2_scalar).
extern "C" __global__ void ds4_iq2_scalar(
    const uint32_t *q,
    const int8_t *a,
    float *out,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t qval = q[idx];
    int sum = 0;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int shift = i * 2;
        int q_i = (int)((qval >> shift) & 0x3);
        int a_i = (int)a[idx * 16 + i];
        sum += q_i * a_i;
    }
    out[idx] = (float)sum * scale;
}

// 3. Sparse MoE Gather Matvec (ds4_moe_expert_dispatch).
extern "C" __global__ void ds4_moe_expert_dispatch(
    const float *expert_weights,
    const float *x,
    float *y,
    const int *expert_ids,
    int slot,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    int expert = expert_ids[slot];
    int64_t offset = ((int64_t)expert * rows + row) * cols;
    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
        sum += expert_weights[offset + c] * x[c];
    }
    y[slot * rows + row] = sum;
}

// 4. Shared-Memory Cooperative Staging (ds4_smem_stage_matvec).
extern "C" __global__ void ds4_smem_stage_matvec(
    const float *a,
    const float *x,
    float *y,
    int rows,
    int cols
) {
    extern __shared__ float s_stage[];

    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Cooperatively stage x into __shared__ memory using float4 vectorized loads
    int num_vec = cols / 4;
    const float4 *x_vec = reinterpret_cast<const float4*>(x);
    float4 *s_vec = reinterpret_cast<float4*>(s_stage);

    for (int i = tid; i < num_vec; i += bdim) {
        s_vec[i] = x_vec[i];
    }
    __syncthreads();

    // Reduction using staged shared memory
    float sum = 0.0f;
    for (int c = tid; c < cols; c += bdim) {
        sum += a[(int64_t)row * cols + c] * s_stage[c];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, offset);
    }

    __shared__ float s_warp_sums[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) {
        s_warp_sums[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        float bsum = (lane < (bdim / 32)) ? s_warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            bsum += __shfl_down_sync(FULL_WARP_MASK, bsum, offset);
        }
        if (lane == 0) {
            y[row] = bsum;
        }
    }
}

// Auxiliary: RMSNorm for transformer layer normalization.
extern "C" __global__ void ds4_rmsnorm(
    const float *x,
    const float *gamma,
    float *out,
    int rows,
    int cols,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    float sumsq = 0.0f;
    for (int c = tid; c < cols; c += bdim) {
        float v = x[(int64_t)row * cols + c];
        sumsq += v * v;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sumsq += __shfl_down_sync(FULL_WARP_MASK, sumsq, offset);
    }
    __shared__ float s_warp_sq[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) s_warp_sq[wid] = sumsq;
    __syncthreads();

    __shared__ float s_inv_rms;
    if (wid == 0) {
        float bsum = (lane < (bdim / 32)) ? s_warp_sq[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            bsum += __shfl_down_sync(FULL_WARP_MASK, bsum, offset);
        }
        if (lane == 0) {
            s_inv_rms = rsqrtf(bsum / (float)cols + eps);
        }
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    for (int c = tid; c < cols; c += bdim) {
        int64_t idx = (int64_t)row * cols + c;
        out[idx] = x[idx] * inv_rms * gamma[c];
    }
}

// Auxiliary: SwiGLU activation.
extern "C" __global__ void ds4_swiglu(
    const float *gate,
    const float *up,
    float *out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu = g / (1.0f + __expf(-g));
        out[i] = silu * up[i];
    }
}

// Baseline vector addition.
extern "C" __global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Baseline compute loop.
extern "C" __global__ void compute(float *x, int n, int rounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    for (int j = 0; j < rounds; j++) v = fmaf(v, 1.000001f, 0.0001f);
    x[i] = v;
}
