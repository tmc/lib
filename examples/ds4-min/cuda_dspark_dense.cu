// Float support projections. CUDA value qualification is required before
// production admission; this reduction does not claim cuBLAS bit identity.
extern "C" __global__ void ds4_dspark_matvec_f32(
    const float *weights, const float *input, float *output,
    unsigned int width) {
    __shared__ float partial[256];
    unsigned int row = blockIdx.x;
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) {
        sum += weights[(unsigned long long)row * width + i] * input[i];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = 128; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = partial[0];
}
