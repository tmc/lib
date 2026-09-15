
// DS4_Q80_MAX_BATCH is the compile-time maximum batch width of
// ds4_q8_0_matvec_quantized_batch. cudaQ80MaxBatch in cuda_q80_linux.go must
// hold the same value; TestCUDAQualificationQ80MaxBatchConstantSync asserts it.
#define DS4_Q80_MAX_BATCH 8

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

// ds4_q8_0_matvec_quantized_batch is ds4_q8_0_matvec_quantized with the weight
// row read once for batch token-major activation rows instead of once per
// token. Each token keeps its own accumulator and its own shfl_down reduction
// over the same lane-strided block order, so every output float32 is bit
// identical to the unbatched kernel's; batching changes only which loop is
// outer. Activations are small enough to stay in L2, so no shared memory.
extern "C" __global__ void ds4_q8_0_matvec_quantized_batch(
    const unsigned char *weights, const ds4_q8_activation *x, float *out,
    unsigned blocks, unsigned rows, unsigned batch) {
  const unsigned warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned row = blockIdx.x * 8 + warp_id;
  if (row >= rows) return;
  const unsigned char *weight_row = weights + (unsigned long long)row * blocks * 34;
  float sum[DS4_Q80_MAX_BATCH] = {0.0f};
  for (unsigned b = lane; b < blocks; b += 32) {
    const unsigned char *p = weight_row + (unsigned long long)b * 34;
    const unsigned short *w16 = reinterpret_cast<const unsigned short *>(p + 2);
    int w4[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      w4[i] = int(unsigned(w16[i * 2]) | (unsigned(w16[i * 2 + 1]) << 16));
    }
    const float d = ds4_half_to_float(unsigned(p[0]) | unsigned(p[1]) << 8);
    #pragma unroll
    for (unsigned t = 0; t < DS4_Q80_MAX_BATCH; t++) {
      if (t < batch) {
        const ds4_q8_activation *tx = x + (unsigned long long)t * blocks;
        const int *x4 = reinterpret_cast<const int *>(tx[b].qs);
        int dot = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
          dot = __dp4a(w4[i], x4[i], dot);
        }
        sum[t] += d * tx[b].d * float(dot);
      }
    }
  }
  #pragma unroll
  for (unsigned t = 0; t < DS4_Q80_MAX_BATCH; t++) {
    if (t < batch) {
      float s = sum[t];
      #pragma unroll
      for (unsigned stride = 16; stride; stride >>= 1)
        s += __shfl_down_sync(0xffffffff, s, stride);
      if (lane == 0) out[(unsigned long long)t * rows + row] = s;
    }
  }
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

extern "C" __global__ void ds4_swiglu_quant_q8_0(
    const float *gate, const float *up, ds4_q8_activation *out,
    unsigned int blocks, float weight) {
  const unsigned warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned block_idx = blockIdx.x * 8 + warp_id;
  if (block_idx >= blocks) return;
  const unsigned long long i = (unsigned long long)block_idx * 32 + lane;
  const float x = fminf(gate[i], 10.0f);
  const float y = fminf(fmaxf(up[i], -10.0f), 10.0f);
  const float val = (x / (1.0f + expf(-x))) * y * weight;
  float mag = fabsf(val);
  #pragma unroll
  for (unsigned stride = 16; stride; stride >>= 1) {
    mag = fmaxf(mag, __shfl_xor_sync(0xffffffff, mag, stride));
  }
  const float d = mag / 127.0f;
  const float inverse = d != 0.0f ? 1.0f / d : 0.0f;
  int qval = (int)lrintf(val * inverse);
  qval = qval > 127 ? 127 : qval < -128 ? -128 : qval;
  ds4_q8_activation *q = out + block_idx;
  q->qs[lane] = (signed char)qval;
  if (lane == 0) q->d = d;
}
