extern "C" __global__ void project(const unsigned *input, unsigned *mid) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    mid[i] = input[i] + 3;
}
extern "C" __global__ void scale(const unsigned *mid, unsigned *output) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = 2 * mid[i];
}
