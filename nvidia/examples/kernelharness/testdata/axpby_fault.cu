// Deliberately faults: indexes far past the end of every buffer. Running this
// poisons the CUDA context for the whole process, so it is never run in the
// same process as another fixture. See the README.
extern "C" __global__ void axpby(float *out, const float *a, const float *b,
                                 float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i + (1 << 28)] = alpha * a[i + (1 << 28)] + b[i];
}
