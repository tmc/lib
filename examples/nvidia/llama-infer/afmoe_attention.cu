#include <cuda_bf16.h>
#include <math.h>

// Storage is float32; every activation boundary explicitly rounds to BF16.
// These baseline kernels follow eager AFMoE operation boundaries. Their scalar
// reductions are not a bit-identity claim for a fused SDPA/GEMM implementation.
__device__ float afmoe_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

extern "C" __global__ void AfmoeHeadNorm(
    float* x, const float* weight, int heads, int head_dim, float eps) {
    int head = blockIdx.x;
    if (head >= heads) return;
    __shared__ float reduction[256];
    float sum = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float v = x[head * head_dim + d];
        sum = __fadd_rn(sum, __fmul_rn(v, v));
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x / 2; n; n /= 2) {
        if (threadIdx.x < n)
            reduction[threadIdx.x] = __fadd_rn(reduction[threadIdx.x], reduction[threadIdx.x + n]);
        __syncthreads();
    }
    float scale = rsqrtf(__fadd_rn(reduction[0] / head_dim, eps));
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        int i = head * head_dim + d;
        float normalized = afmoe_bf16(__fmul_rn(x[i], scale));
        x[i] = afmoe_bf16(__fmul_rn(normalized, weight[d]));
    }
}

extern "C" __global__ void AfmoeRoPE(
    float* x, int heads, int head_dim, int pos, float theta) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    if (pair >= heads * half) return;
    int h = pair / half;
    int d = pair % half;
    float inv = 1.0f / powf(theta, float(2 * d) / head_dim);
    float angle = __fmul_rn(float(pos), inv);
    float c = afmoe_bf16(cosf(angle));
    float s = afmoe_bf16(sinf(angle));
    int i = h * head_dim + d;
    int j = i + half;
    float a = x[i], b = x[j];
    x[i] = afmoe_bf16(__fadd_rn(afmoe_bf16(__fmul_rn(a, c)), afmoe_bf16(__fmul_rn(-b, s))));
    x[j] = afmoe_bf16(__fadd_rn(afmoe_bf16(__fmul_rn(b, c)), afmoe_bf16(__fmul_rn(a, s))));
}

extern "C" __global__ void AfmoeAttentionDecode(
    const float* q, const float* keys, const float* values, float* out,
    int heads, int kv_heads, int head_dim, int pos, int window) {
    int h = blockIdx.x;
    if (h >= heads) return;
    int start = window > 0 && pos >= window ? pos - window + 1 : 0;
    int count = pos - start + 1;
    int kvh = h / (heads / kv_heads);
    int stride = kv_heads * head_dim;
    extern __shared__ float shared[];
    float* scores = shared;
    float* reduction = shared + count;
    float max_score = -INFINITY;
    float scale = 1.0f / sqrtf(float(head_dim));
    for (int t = threadIdx.x; t < count; t += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot = __fadd_rn(dot, __fmul_rn(q[h * head_dim + d], keys[(start + t) * stride + kvh * head_dim + d]));
        float score = afmoe_bf16(__fmul_rn(afmoe_bf16(dot), scale));
        scores[t] = score;
        max_score = fmaxf(max_score, score);
    }
    reduction[threadIdx.x] = max_score;
    __syncthreads();
    for (int n = blockDim.x / 2; n; n /= 2) {
        if (threadIdx.x < n)
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + n]);
        __syncthreads();
    }
    max_score = reduction[0];
    float sum = 0.0f;
    for (int t = threadIdx.x; t < count; t += blockDim.x) {
        float p = expf(scores[t] - max_score);
        scores[t] = p;
        sum = __fadd_rn(sum, p);
    }
    __syncthreads();
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x / 2; n; n /= 2) {
        if (threadIdx.x < n)
            reduction[threadIdx.x] = __fadd_rn(reduction[threadIdx.x], reduction[threadIdx.x + n]);
        __syncthreads();
    }
    sum = reduction[0];
    for (int t = threadIdx.x; t < count; t += blockDim.x)
        scores[t] = afmoe_bf16(scores[t] / sum);
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float value = 0.0f;
        for (int t = 0; t < count; ++t)
            value = __fadd_rn(value, __fmul_rn(scores[t], values[(start + t) * stride + kvh * head_dim + d]));
        out[h * head_dim + d] = afmoe_bf16(value);
    }
}

extern "C" __global__ void AfmoeAttentionGate(float* out, const float* gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sigmoid = afmoe_bf16(1.0f / (1.0f + expf(-gate[i])));
    out[i] = afmoe_bf16(__fmul_rn(out[i], sigmoid));
}
