// A correct candidate: vectorized by four, with a scalar tail.
extern "C" __global__ void axpby(float *out, const float *a, const float *b,
                                 float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int quads = n / 4;
  const float4 *a4 = reinterpret_cast<const float4 *>(a);
  const float4 *b4 = reinterpret_cast<const float4 *>(b);
  float4 *o4 = reinterpret_cast<float4 *>(out);
  for (int q = i; q < quads; q += stride) {
    float4 x = a4[q], y = b4[q], r;
    r.x = alpha * x.x + y.x;
    r.y = alpha * x.y + y.y;
    r.z = alpha * x.z + y.z;
    r.w = alpha * x.w + y.w;
    o4[q] = r;
  }
  for (int k = quads * 4 + i; k < n; k += stride) out[k] = alpha * a[k] + b[k];
}
