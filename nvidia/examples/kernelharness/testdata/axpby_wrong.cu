// Deliberately wrong: subtracts b instead of adding it. This is the null
// control for the comparator -- a harness whose checker cannot fail would
// report this as correct.
extern "C" __global__ void axpby(float *out, const float *a, const float *b,
                                 float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride) out[i] = alpha * a[i] - b[i];
}
