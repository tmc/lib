
extern "C" __global__ void ds4_q8_0_matvec_quantized(
    const unsigned char *weights, const ds4_q8_activation *x, float *out,
    unsigned blocks, unsigned rows) {
  const unsigned warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned row = blockIdx.x * 8 + warp_id;
  if (row >= rows) return;
  const unsigned char *weight_row = weights + (unsigned long long)row * blocks * 34;
  float sum = 0.0f;
  for (unsigned b = lane; b < blocks; b += 32) {
    const unsigned char *p = weight_row + (unsigned long long)b * 34;
    const unsigned short *w16 = reinterpret_cast<const unsigned short *>(p + 2);
    const int *x4 = reinterpret_cast<const int *>(x[b].qs);
    int dot = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int w4 = int(unsigned(w16[i * 2]) | (unsigned(w16[i * 2 + 1]) << 16));
      dot = __dp4a(w4, x4[i], dot);
    }
    const float d = ds4_half_to_float(unsigned(p[0]) | unsigned(p[1]) << 8);
    sum += d * x[b].d * float(dot);
  }
  for (unsigned stride = 16; stride; stride >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  if (lane == 0) out[row] = sum;
}

extern "C" __global__ void ds4_q8_0_matvec_quantized_blockdiag(
    const unsigned char *weights, const ds4_q8_activation *x, float *out,
    unsigned blocks, unsigned rows) {
  const unsigned warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned row = blockIdx.x * 8 + warp_id, diag = blockIdx.y;
  if (row >= rows) return;
  const unsigned char *weight_row = weights + ((unsigned long long)diag * rows + row) * blocks * 34;
  const ds4_q8_activation *xb = x + (unsigned long long)diag * blocks;
  float sum = 0.0f;
  for (unsigned b = lane; b < blocks; b += 32) {
    const unsigned char *p = weight_row + (unsigned long long)b * 34;
    const unsigned short *w16 = reinterpret_cast<const unsigned short *>(p + 2);
    const int *x4 = reinterpret_cast<const int *>(xb[b].qs);
    int dot = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int w4 = int(unsigned(w16[i * 2]) | (unsigned(w16[i * 2 + 1]) << 16));
      dot = __dp4a(w4, x4[i], dot);
    }
    const float d = ds4_half_to_float(unsigned(p[0]) | unsigned(p[1]) << 8);
    sum += d * xb[b].d * float(dot);
  }
  for (unsigned stride = 16; stride; stride >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  if (lane == 0) out[(unsigned long long)diag * rows + row] = sum;
}
