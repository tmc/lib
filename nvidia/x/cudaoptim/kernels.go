package cudaoptim

// blockThreads is the block size of every kernel in kernelSource.
const blockThreads = 256

// kernelSource holds the three kernels a step launches. The struct layouts
// must match tensorEntry, chunkEntry and State.
//
// Both reductions are deterministic: each block sums its chunk in a fixed
// thread order and a fixed shuffle tree, and a single block then sums the
// per-chunk partials in the same way. No atomics touch a value that is read
// back.
const kernelSource = `#include "cudablock.cuh"

struct cudaoptim_tensor {
    float* p;
    const float* g;
    float* m;
    float* v;
    long long n;
    float lr_scale;
    float weight_decay;
};

struct cudaoptim_chunk {
    int tensor;
    int pad;
    long long start;
};

struct cudaoptim_state {
    int step;
    int nonfinite;
    float lr;
    float norm;
    float clip_scale;
    float bias1;
    float inv_sqrt_bias2;
    int pad;
};

#define THREADS 256

// Sum of squares of one chunk of one gradient, in float64.
extern "C" __global__ void cudaoptim_norm_partials(
    const cudaoptim_tensor* __restrict__ tensors,
    const cudaoptim_chunk* __restrict__ chunks,
    long long chunk_size,
    double* __restrict__ partials)
{
    __shared__ double scratch[THREADS / WARP_SIZE];
    cudaoptim_chunk c = chunks[blockIdx.x];
    cudaoptim_tensor t = tensors[c.tensor];
    long long end = c.start + chunk_size;
    if (end > t.n) end = t.n;
    double sum = 0;
    for (long long i = c.start + threadIdx.x; i < end; i += THREADS) {
        double x = t.g[i];
        sum += x * x;
    }
    sum = block_reduce_sum<double, THREADS>(sum, scratch);
    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}

// Folds the partials into the global norm, the clip scale and the
// non-finite flag. When advance is nonzero and the norm is finite it also
// counts the step and computes its bias corrections.
extern "C" __global__ void cudaoptim_finalize(
    const double* __restrict__ partials,
    int nchunks,
    float max_norm,
    float beta1,
    float beta2,
    int advance,
    cudaoptim_state* __restrict__ state)
{
    __shared__ double scratch[THREADS / WARP_SIZE];
    double sum = 0;
    for (int i = threadIdx.x; i < nchunks; i += THREADS) sum += partials[i];
    sum = block_reduce_sum<double, THREADS>(sum, scratch);
    if (threadIdx.x != 0) return;
    double norm = sqrt(sum);
    double scale = 1.0;
    if (max_norm > 0) {
        scale = (double)max_norm / (norm + 1e-6);
        if (scale > 1.0) scale = 1.0;
    }
    state->norm = (float)norm;
    state->clip_scale = (float)scale;
    if (!isfinite(norm)) {
        state->nonfinite = 1;
    }
    if (advance && !state->nonfinite) {
        int step = state->step + 1;
        state->step = step;
        state->bias1 = (float)(1.0 - pow((double)beta1, (double)step));
        state->inv_sqrt_bias2 = (float)(1.0 / sqrt(1.0 - pow((double)beta2, (double)step)));
    }
}

// Applies one AdamW update to one chunk of one tensor. It does nothing once
// the non-finite flag is set.
extern "C" __global__ void cudaoptim_adamw(
    const cudaoptim_tensor* __restrict__ tensors,
    const cudaoptim_chunk* __restrict__ chunks,
    long long chunk_size,
    const cudaoptim_state* __restrict__ state,
    float beta1,
    float beta2,
    float eps)
{
    if (state->nonfinite) return;
    cudaoptim_chunk c = chunks[blockIdx.x];
    cudaoptim_tensor t = tensors[c.tensor];
    long long end = c.start + chunk_size;
    if (end > t.n) end = t.n;
    const float lr = state->lr * t.lr_scale;
    const float clip = state->clip_scale;
    // The parameter update runs in double and rounds once. In float it
    // rounds twice, and the resulting 1-ulp differences from the float64
    // reference add up as a random walk over steps. The moments stay in
    // float: their rounding moves the update by about 1e-7 of lr.
    const double lr_wd = (double)lr * (double)t.weight_decay;
    const double step_size = (double)lr / (double)state->bias1;
    const float inv_sqrt_bias2 = state->inv_sqrt_bias2;
    for (long long i = c.start + threadIdx.x; i < end; i += THREADS) {
        float g = t.g[i] * clip;
        float m = beta1 * t.m[i] + (1.0f - beta1) * g;
        float v = beta2 * t.v[i] + (1.0f - beta2) * g * g;
        float denom = sqrtf(v) * inv_sqrt_bias2 + eps;
        t.m[i] = m;
        t.v[i] = v;
        double p = t.p[i];
        t.p[i] = (float)(p - lr_wd * p - step_size * (double)(m / denom));
    }
}
`
