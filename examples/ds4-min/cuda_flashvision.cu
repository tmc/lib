// Flash Vision BF16 primitives. The small-row route consumes F32 activations;
// the large-row route first converts them to BF16 for cuBLAS GemmEx.
// Source contract: antirez/ds4 11e8cad, ds4_cuda.cu:27823-27948.
#include <cuda_bf16.h>
#include <stdint.h>

__device__ static float vision_bf16(const uint16_t *p) {
    return __uint_as_float((uint32_t)*p << 16);
}

__device__ static float vision_round(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) == 0x7f800000u) return x;
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

extern "C" __global__ void ds4_vision_matvec_bf16(
        float *out, const uint16_t *weights, const float *x,
        uint32_t in_dim, uint32_t out_dim) {
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t col = blockIdx.x * 8u + warp;
    const uint32_t row = blockIdx.y;
    float sum = 0.0f;
    if (col < out_dim) {
        const uint16_t *w = weights + (uint64_t)col * in_dim;
        const float *a = x + (uint64_t)row * in_dim;
        for (uint32_t i = lane; i < in_dim; i += 32u)
            sum = fmaf(__uint_as_float((uint32_t)w[i] << 16), a[i], sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0u && col < out_dim)
        out[(uint64_t)row * out_dim + col] = sum;
}

extern "C" __global__ void ds4_vision_f32_to_bf16(
        __nv_bfloat16 *out, const float *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2bfloat16(x[i]);
}

extern "C" __global__ void ds4_vision_round_bf16(float *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t bits = __float_as_uint(x[i]);
    if ((bits & 0x7f800000u) == 0x7f800000u) return;
    bits += 0x00007fffu + ((bits >> 16u) & 1u);
    x[i] = __uint_as_float(bits & 0xffff0000u);
}

extern "C" __global__ void ds4_vision_bias(
        float *x, const uint16_t *bias, const float *residual,
        uint64_t count, uint32_t width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = x[i] + vision_bf16(bias + i % width);
    if (residual) v += residual[i];
    x[i] = v;
}

extern "C" __global__ void ds4_vision_rms(
        float *out, const float *x, const uint16_t *weight,
        uint32_t width, float eps) {
    __shared__ float partial[256];
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const float *xr = x + (uint64_t)row * width;
    float *yr = out + (uint64_t)row * width;
    float sum = 0;
    for (uint32_t d = tid; d < width; d += blockDim.x)
        sum = fmaf(xr[d], xr[d], sum);
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)width + eps);
    for (uint32_t d = tid; d < width; d += blockDim.x)
        yr[d] = xr[d] * inv * vision_bf16(weight + d);
}

extern "C" __global__ void ds4_vision_qkv_rope(
        float *q, float *k, float *v, const float *qkv,
        const uint16_t *bias, uint32_t rows, uint32_t grid_w) {
    const uint32_t row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    if (row >= rows || head >= 16 || lane >= 32) return;
    const uint64_t qb = (uint64_t)row * 3072 + head * 64;
    const uint64_t ob = (uint64_t)row * 1024 + head * 64;
    const uint64_t bb = head * 64;
    const float q0 = vision_round(qkv[qb+lane] + vision_bf16(bias+bb+lane));
    const float q1 = vision_round(qkv[qb+lane+32] + vision_bf16(bias+bb+lane+32));
    const float k0 = vision_round(qkv[qb+1024+lane] + vision_bf16(bias+1024+bb+lane));
    const float k1 = vision_round(qkv[qb+1024+lane+32] + vision_bf16(bias+1024+bb+lane+32));
    const uint32_t y = row / grid_w, x = row - y * grid_w;
    const uint32_t pos = lane < 16 ? y : x, freq = lane & 15;
    const float angle = (float)pos * powf(10000.0f, -(float)freq / 16.0f);
    const float cs = cosf(angle), sn = sinf(angle);
    q[ob+lane] = vision_round(q0*cs - q1*sn);
    q[ob+lane+32] = vision_round(q1*cs + q0*sn);
    k[ob+lane] = vision_round(k0*cs - k1*sn);
    k[ob+lane+32] = vision_round(k1*cs + k0*sn);
    v[ob+lane] = vision_round(qkv[qb+2048+lane] + vision_bf16(bias+2048+bb+lane));
    v[ob+lane+32] = vision_round(qkv[qb+2048+lane+32] + vision_bf16(bias+2048+bb+lane+32));
}

extern "C" __global__ void ds4_vision_attention(
        float *out, const float *q, const float *k, const float *v, uint32_t rows) {
    __shared__ float dot[32];
    const uint32_t row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    if (row >= rows || head >= 16 || lane >= 32) return;
    const uint64_t base = (uint64_t)row * 1024 + head * 64;
    const float q0 = q[base+lane], q1 = q[base+lane+32];
    float acc0 = 0, acc1 = 0, max_score = -INFINITY, denom = 0;
    for (uint32_t r = 0; r < rows; r++) {
        const uint64_t kb = (uint64_t)r * 1024 + head * 64;
        dot[lane] = q0*k[kb+lane] + q1*k[kb+lane+32];
        __syncthreads();
        for (uint32_t s = 16; s; s >>= 1) {
            if (lane < s) dot[lane] += dot[lane+s];
            __syncthreads();
        }
        const float score = dot[0] * 0.125f;
        const float next_max = fmaxf(max_score, score);
        const float old_scale = r == 0 ? 0 : expf(max_score-next_max);
        const float new_scale = expf(score-next_max);
        denom = denom*old_scale + new_scale;
        acc0 = acc0*old_scale + new_scale*v[kb+lane];
        acc1 = acc1*old_scale + new_scale*v[kb+lane+32];
        max_score = next_max;
        __syncthreads();
    }
    out[base+lane] = acc0/denom;
    out[base+lane+32] = acc1/denom;
}

extern "C" __global__ void ds4_vision_swiglu(
        float *out, const float *gate_up, uint64_t count, uint32_t width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint64_t source = (i/width)*width*2 + i%width;
    const float gate = gate_up[source], up = gate_up[source+width];
    out[i] = vision_round((gate/(1.0f+expf(-gate)))*up);
}

extern "C" __global__ void ds4_vision_add_residual(
        float *x, const float *residual, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = vision_round(x[i] + residual[i]);
}

extern "C" __global__ void ds4_vision_aligner_reorder(
        float *out, const float *x, uint32_t grid_h, uint32_t grid_w, uint32_t rows) {
    const uint32_t d = blockIdx.x*blockDim.x+threadIdx.x, row = blockIdx.y;
    if (d >= 9216 || row >= rows) return;
    const uint32_t ow = (grid_w+2)/3, by = row/ow, bx = row-by*ow;
    const uint32_t channel = d/9, within = d-channel*9;
    const uint32_t sy = by*3+within/3, sx = bx*3+within%3;
    float value = 0;
    if (sy < grid_h && sx < grid_w) value = x[((uint64_t)sy*grid_w+sx)*1024+channel];
    out[(uint64_t)row*9216+d] = value;
}

__device__ static float vision_erf(float x) {
    const float sign = x < 0 ? -1.0f : 1.0f, a = fabsf(x);
    const float t = 1.0f/(1.0f+0.3275911f*a);
    const float p = (((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t;
    return sign*(1.0f-p*expf(-a*a));
}

extern "C" __global__ void ds4_vision_gelu_bias(
        float *out, const float *x, const uint16_t *bias, uint64_t count, uint32_t width) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    const float value = vision_round(x[i]+vision_bf16(bias+i%width));
    out[i] = vision_round(0.5f*value*(1.0f+vision_erf(value*0.7071067811865475f)));
}
