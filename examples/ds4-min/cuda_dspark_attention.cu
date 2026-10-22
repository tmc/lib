#include <math.h>
#include <cuda_fp16.h>

// E4M3FN nearest-even round trip, including the finite exponent-15 values.
// This is the pinned fp8_kv_quantize_row policy, not an FP8 cache encoding.
__device__ static float dspark_e4m3_value(int i) {
    int exponent = (i >> 3) & 15, mantissa = i & 7;
    if (exponent == 0) return (float)mantissa * 0.001953125f;
    return (1.0f + (float)mantissa * 0.125f) * exp2f((float)exponent - 7.0f);
}

__device__ static float dspark_e4m3_round(float value) {
    float magnitude = fminf(fabsf(value), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dspark_e4m3_value(mid) <= magnitude) lo = mid;
        else hi = mid - 1;
    }
    if (lo < 126) {
        float a = fabsf(magnitude - dspark_e4m3_value(lo));
        float b = fabsf(magnitude - dspark_e4m3_value(lo + 1));
        if (b < a || (b == a && (lo & 1))) lo++;
    }
    return (value < 0.0f ? -1.0f : 1.0f) * dspark_e4m3_value(lo);
}

// Mutate each normalized/rotated KV row before half-storing it in the private
// ring: first448 FP8 round trip, rotary64 untouched, then half all512.
// The caller supplies contiguous feature+draft rows, not target rawKV.
extern "C" __global__ void ds4_dspark_store_raw(
    float *kv, float *raw, unsigned int capacity, unsigned int start) {
    __shared__ float scratch[64];
    unsigned int tid = threadIdx.x;
    float *x = kv + (unsigned long long)blockIdx.x * 512;
    for (unsigned int offset = 0; offset < 448; offset += 64) {
        float value = x[offset + tid];
        scratch[tid] = fabsf(value);
        __syncthreads();
        for (unsigned int stride = 32; stride != 0; stride /= 2) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        x[offset + tid] = dspark_e4m3_round(value / scale) * scale;
        __syncthreads();
    }
    float *destination = raw + (unsigned long long)((start + blockIdx.x) % capacity) * 512;
    for (unsigned int d = tid; d < 512; d += 64)
        destination[d] = __half2float(__float2half_rn(x[d]));
}

// Support attention includes all visible draft keys, unlike target attention.
// Each block evaluates one draft/query head. The sink has zero V but remains
// in the denominator (pinned ds4_cuda.cu:16442). No target cache is modified.
extern "C" __global__ void ds4_dspark_attention_noncausal(
    const float *queries, const float *raw, const float *sinks, float *output,
    unsigned int capacity, unsigned int start, unsigned int count) {
    __shared__ float partial[256];
    __shared__ float maximum, denominator, alpha, beta;
    unsigned int head = blockIdx.x;
    unsigned int query_head = blockIdx.y * 64 + head;
    unsigned int d = threadIdx.x;
    const float *q = queries + (unsigned long long)query_head * 512;
    float value0 = 0.0f, value1 = 0.0f;
    if (d == 0) {
        maximum = sinks[head];
        denominator = 1.0f;
    }
    __syncthreads();
    for (unsigned int row = 0; row < count; row++) {
        const float *kv = raw + (unsigned long long)((start + row) % capacity) * 512;
        partial[d] = q[d] * kv[d] + q[d + 256] * kv[d + 256];
        __syncthreads();
        for (unsigned int stride = 128; stride != 0; stride /= 2) {
            if (d < stride) partial[d] += partial[d + stride];
            __syncthreads();
        }
        if (d == 0) {
            float score = partial[0] * rsqrtf(512.0f);
            float next = fmaxf(maximum, score);
            alpha = expf(maximum - next);
            beta = expf(score - next);
            denominator = denominator * alpha + beta;
            maximum = next;
        }
        __syncthreads();
        value0 = value0 * alpha + kv[d] * beta;
        value1 = value1 * alpha + kv[d + 256] * beta;
        __syncthreads();
    }
    output[(unsigned long long)query_head * 512 + d] = value0 / denominator;
    output[(unsigned long long)query_head * 512 + d + 256] = value1 / denominator;
}

// DSpark always uses dense RoPE, base10000, without target compressed YaRN.
extern "C" __global__ void ds4_dspark_rope_tail(
    float *values, unsigned int heads, unsigned int position,
    unsigned int inverse) {
    unsigned int pair = threadIdx.x;
    if (pair >= 32) return;
    unsigned int row = blockIdx.x / heads;
    unsigned int d = 448 + pair * 2;
    float theta = (float)(position + row) * powf(10000.0f, -(float)(pair * 2) / 64.0f);
    float c = cosf(theta), s = sinf(theta);
    if (inverse) s = -s;
    float *x = values + (unsigned long long)blockIdx.x * 512;
    float a = x[d], b = x[d + 1];
    x[d] = a * c - b * s;
    x[d + 1] = a * s + b * c;
}
