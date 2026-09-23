// Compiles, but the entry point is C++-mangled: no extern "C". The module
// loads and does not define the symbol the task requires.
__global__ void axpby(float *out, const float *a, const float *b, float alpha,
                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride) out[i] = alpha * a[i] + b[i];
}
