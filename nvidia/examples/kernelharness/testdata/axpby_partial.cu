// Deliberately incomplete: no grid-stride loop, so every element beyond the
// first grid's worth is never written. This is the control for the NaN
// prefill -- with a zero prefill the untouched tail would often read as
// correct, because the reference is near zero there only by luck.
extern "C" __global__ void axpby(float *out, const float *a, const float *b,
                                 float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = alpha * a[i] + b[i];
}
