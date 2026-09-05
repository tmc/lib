extern "C" __global__ void addone(float *p) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  p[i] += 1.0f;
}
