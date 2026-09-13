
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
  unsigned long long chunk = ((unsigned long long)cols + 31) / 32;
  unsigned long long start = tid * chunk, end = start + chunk;
  if (end > cols) end = cols;
  float sum_a = 0.0f, sum_b = 0.0f;
  for (unsigned long long i = start; i < end; i++) {
    sum_a += ds4_half_to_float(a[offset + i]) * x[i];
    sum_b += ds4_half_to_float(b[offset + i]) * x[i];
  }
  __shared__ float partial_a[32], partial_b[32];
  partial_a[tid] = sum_a; partial_b[tid] = sum_b;
  __syncthreads();
  if (tid == 0) {
    float total_a = 0.0f, total_b = 0.0f;
    for (unsigned i = 0; i < 32; i++) { total_a += partial_a[i]; total_b += partial_b[i]; }
    out_a[row] = total_a; out_b[row] = total_b;
  }
}

