
extern "C" __global__ void ds4_f16_matvec_half_input(
    const unsigned short *weights, const float *x, float *out, unsigned cols) {
  const unsigned row = blockIdx.x, tid = threadIdx.x;
  const unsigned long long offset = (unsigned long long)row * cols;
  double sum = 0.0;
  for (unsigned i = tid; i < cols; i += blockDim.x)
    sum += double(ds4_half_to_float(weights[offset + i])) * double(ds4_half_round(x[i]));
  __shared__ double partial[256];
  partial[tid] = sum;
  __syncthreads();
  for (unsigned stride = 128; stride; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[row] = float(partial[0]);
}
extern "C" __global__ void ds4_f16_pair_ordered_chunks(
    const unsigned short *a, const unsigned short *b, const float *x,
    float *out_a, float *out_b, unsigned cols) {
  unsigned row = blockIdx.x, tid = threadIdx.x;
  unsigned long long offset = (unsigned long long)row * cols;

  __shared__ unsigned short s_a[32][130];
  __shared__ unsigned short s_b[32][130];
  __shared__ float partial_a[32], partial_b[32];

  if (cols == 4096) {
    const uint4 *a_vec = (const uint4 *)(a + offset);
    const uint4 *b_vec = (const uint4 *)(b + offset);
    for (unsigned idx = tid; idx < 512; idx += blockDim.x) {
      uint4 va = a_vec[idx];
      uint4 vb = b_vec[idx];
      unsigned elem = idx << 3;
      unsigned c = elem >> 7;
      unsigned k = elem & 127;
      *(unsigned int *)&s_a[c][k]     = va.x;
      *(unsigned int *)&s_a[c][k + 2] = va.y;
      *(unsigned int *)&s_a[c][k + 4] = va.z;
      *(unsigned int *)&s_a[c][k + 6] = va.w;
      *(unsigned int *)&s_b[c][k]     = vb.x;
      *(unsigned int *)&s_b[c][k + 2] = vb.y;
      *(unsigned int *)&s_b[c][k + 4] = vb.z;
      *(unsigned int *)&s_b[c][k + 6] = vb.w;
    }
    __syncthreads();

    if (tid < 32) {
      const float *xk = x + (tid << 7);
      float sum_a = 0.0f, sum_b = 0.0f;
      for (unsigned k = 0; k < 128; k++) {
        sum_a += ds4_half_to_float(s_a[tid][k]) * xk[k];
        sum_b += ds4_half_to_float(s_b[tid][k]) * xk[k];
      }
      partial_a[tid] = sum_a;
      partial_b[tid] = sum_b;
    }
    __syncthreads();

    if (tid == 0) {
      float total_a = 0.0f, total_b = 0.0f;
      for (unsigned i = 0; i < 32; i++) {
        total_a += partial_a[i];
        total_b += partial_b[i];
      }
      out_a[row] = total_a;
      out_b[row] = total_b;
    }
  } else if (cols <= 4096) {
    const unsigned chunk = (cols + 31) / 32;
    for (unsigned i = tid; i < cols; i += blockDim.x) {
      unsigned c = i / chunk;
      unsigned k = i - c * chunk;
      if (c < 32 && k < 130) {
        s_a[c][k] = a[offset + i];
        s_b[c][k] = b[offset + i];
      }
    }
    __syncthreads();

    if (tid < 32) {
      unsigned start = tid * chunk;
      unsigned count = (cols > start) ? (cols - start) : 0;
      if (count > chunk) count = chunk;
      const float *xk = x + start;
      float sum_a = 0.0f, sum_b = 0.0f;
      for (unsigned k = 0; k < count; k++) {
        sum_a += ds4_half_to_float(s_a[tid][k]) * xk[k];
        sum_b += ds4_half_to_float(s_b[tid][k]) * xk[k];
      }
      partial_a[tid] = sum_a;
      partial_b[tid] = sum_b;
    }
    __syncthreads();

    if (tid == 0) {
      float total_a = 0.0f, total_b = 0.0f;
      for (unsigned i = 0; i < 32; i++) {
        total_a += partial_a[i];
        total_b += partial_b[i];
      }
      out_a[row] = total_a;
      out_b[row] = total_b;
    }
  } else {
    unsigned long long chunk = ((unsigned long long)cols + 31) / 32;
    unsigned long long start = tid * chunk, end = start + chunk;
    if (end > cols) end = cols;
    float sum_a = 0.0f, sum_b = 0.0f;
    for (unsigned long long i = start; i < end; i++) {
      sum_a += ds4_half_to_float(a[offset + i]) * x[i];
      sum_b += ds4_half_to_float(b[offset + i]) * x[i];
    }
    partial_a[tid] = sum_a;
    partial_b[tid] = sum_b;
    __syncthreads();

    if (tid == 0) {
      float total_a = 0.0f, total_b = 0.0f;
      for (unsigned i = 0; i < 32; i++) {
        total_a += partial_a[i];
        total_b += partial_b[i];
      }
      out_a[row] = total_a;
      out_b[row] = total_b;
    }
  }
}

