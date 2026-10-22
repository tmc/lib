// Reduction/order matches antirez/ds4 11e8cad ds4_cuda.cu:12451.
extern "C" __global__ void ds4_directional_project(float *x, const float *directions,
        unsigned layer, unsigned width, unsigned rows, float scale) {
    unsigned row = blockIdx.x;
    if (row >= rows || width == 0) return;
    float *xr = x + (unsigned long long)row * width;
    const float *dir = directions + (unsigned long long)layer * width;
    float sum = 0;
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) sum += xr[i] * dir[i];
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float coefficient = scale * partial[0];
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) xr[i] -= coefficient * dir[i];
}
