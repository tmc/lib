// A small single-token inference pipeline. These are readable reference kernels,
// not tuned implementations or a complete transformer layer.
extern "C" __global__ void rmsnorm(const float *input, float *norm, int n) {
    __shared__ float sums[256];
    unsigned t = threadIdx.x;
    float sum = 0;
    for (unsigned i = t; i < n; i += blockDim.x)
        sum += input[i] * input[i];
    sums[t] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (t < stride) sums[t] += sums[t + stride];
        __syncthreads();
    }
    float inv = rsqrtf(sums[0] / n + 1e-5f);
    for (unsigned i = t; i < n; i += blockDim.x)
        norm[i] = input[i] * inv;
}
extern "C" __global__ void project(const float *weights, const float *norm,
                                   float *projection, int n) {
    __shared__ float sums[128];
    unsigned t = threadIdx.x, row = blockIdx.x;
    float sum = 0;
    for (unsigned i = t; i < n; i += blockDim.x)
        sum += weights[row * n + i] * norm[i];
    sums[t] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (t < stride) sums[t] += sums[t + stride];
        __syncthreads();
    }
    if (t == 0) projection[row] = sums[0];
}
extern "C" __global__ void silu(const float *projection, float *output) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = projection[i];
    output[i] = x / (1 + expf(-x));
}
