// Deliberately does not compile: alpha is undeclared.
extern "C" __global__ void axpby(float *out, const float *a, const float *b,
                                 float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = alpha_typo * a[i] + b[i];
}
