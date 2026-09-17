
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
    if (tid < 32) {
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
  }
}


// DS4_F16_PAIR_MAX_BATCH bounds the batched pair kernel below. The Go side
// mirrors it as cudaF16PairMaxBatch; a test asserts the two agree.
#define DS4_F16_PAIR_MAX_BATCH 16

// ds4_f16_pair_ordered_chunks_batch projects batch token-major activation rows
// against the same weight pair with one launch, reading each weight element
// once for all tokens. Every token keeps the chunk partition, the ascending
// within-chunk order, and the ascending chunk-partial sum of the single-token
// kernel above, so each output is bit-identical to a single-token launch.
//
// The launcher passes Block(256), so 8 warps process tokens in strided pairs
// (tok0 = warp, tok1 = warp + 8). Grid is (rows, ceil(batch/16)), so
// gridDim.x is the row count and gridDim.y is the batch tiles; out_a and out_b
// are token-major, row-minor.
extern "C" __global__ void ds4_f16_pair_ordered_chunks_batch(
    const unsigned short *a, const unsigned short *b, const float *x,
    float *out_a, float *out_b, unsigned cols, unsigned batch) {
  const unsigned row = blockIdx.x, tid = threadIdx.x;
  const unsigned lane = tid & 31, warp = tid >> 5;
  const unsigned t_base = blockIdx.y * DS4_F16_PAIR_MAX_BATCH;

  if (t_base >= batch) return;
  const unsigned cur_batch = (batch - t_base < DS4_F16_PAIR_MAX_BATCH) ? (batch - t_base) : DS4_F16_PAIR_MAX_BATCH;

  const unsigned long long offset = (unsigned long long)row * cols;
  const unsigned long long rows = gridDim.x;
  const float *batch_x = x + (unsigned long long)t_base * cols;
  float *batch_out_a = out_a + (unsigned long long)t_base * rows;
  float *batch_out_b = out_b + (unsigned long long)t_base * rows;

  __shared__ float s_raw_a[32 * 130 / 2], s_raw_b[32 * 130 / 2];
  unsigned short (*s_a)[130] = (unsigned short (*)[130])s_raw_a;
  unsigned short (*s_b)[130] = (unsigned short (*)[130])s_raw_b;
  float (*s_af)[33] = (float (*)[33])s_raw_a;
  float (*s_bf)[33] = (float (*)[33])s_raw_b;
  __shared__ float partial_a[DS4_F16_PAIR_MAX_BATCH][32];
  __shared__ float partial_b[DS4_F16_PAIR_MAX_BATCH][32];

  const unsigned tok0 = warp;
  const unsigned tok1 = warp + 8;

  if (cols == 4096) {
    float sum0_a = 0.0f, sum0_b = 0.0f;
    float sum1_a = 0.0f, sum1_b = 0.0f;
    const float *xk0 = (tok0 < cur_batch) ? (batch_x + (unsigned long long)tok0 * cols + (lane << 7)) : nullptr;
    const float *xk1 = (tok1 < cur_batch) ? (batch_x + (unsigned long long)tok1 * cols + (lane << 7)) : nullptr;

    for (unsigned t0 = 0; t0 < 128; t0 += 32) {
      __syncthreads();
      for (unsigned idx = tid; idx < 32 * 32; idx += blockDim.x) {
        const unsigned c = idx >> 5, j = idx & 31;
        const unsigned long long e = offset + ((unsigned long long)c << 7) + t0 + j;
        s_af[c][j] = ds4_half_to_float(a[e]);
        s_bf[c][j] = ds4_half_to_float(b[e]);
      }
      __syncthreads();
      for (unsigned j = 0; j < 32; j++) {
        const float wa = s_af[lane][j];
        const float wb = s_bf[lane][j];
        if (xk0) {
          const float x0 = xk0[t0 + j];
          sum0_a += wa * x0;
          sum0_b += wb * x0;
        }
        if (xk1) {
          const float x1 = xk1[t0 + j];
          sum1_a += wa * x1;
          sum1_b += wb * x1;
        }
      }
    }
    if (tok0 < cur_batch) {
      partial_a[tok0][lane] = sum0_a;
      partial_b[tok0][lane] = sum0_b;
    }
    if (tok1 < cur_batch) {
      partial_a[tok1][lane] = sum1_a;
      partial_b[tok1][lane] = sum1_b;
    }
    __syncthreads();

    if (lane == 0) {
      if (tok0 < cur_batch) {
        float total_a = 0.0f, total_b = 0.0f;
        for (unsigned i = 0; i < 32; i++) {
          total_a += partial_a[tok0][i];
          total_b += partial_b[tok0][i];
        }
        batch_out_a[(unsigned long long)tok0 * rows + row] = total_a;
        batch_out_b[(unsigned long long)tok0 * rows + row] = total_b;
      }
      if (tok1 < cur_batch) {
        float total_a = 0.0f, total_b = 0.0f;
        for (unsigned i = 0; i < 32; i++) {
          total_a += partial_a[tok1][i];
          total_b += partial_b[tok1][i];
        }
        batch_out_a[(unsigned long long)tok1 * rows + row] = total_a;
        batch_out_b[(unsigned long long)tok1 * rows + row] = total_b;
      }
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

    for (unsigned tok = warp; tok < cur_batch; tok += 8) {
      unsigned start = lane * chunk;
      unsigned count = (cols > start) ? (cols - start) : 0;
      if (count > chunk) count = chunk;
      const float *xk = batch_x + (unsigned long long)tok * cols + start;
      float sum_a = 0.0f, sum_b = 0.0f;
      for (unsigned k = 0; k < count; k++) {
        sum_a += ds4_half_to_float(s_a[lane][k]) * xk[k];
        sum_b += ds4_half_to_float(s_b[lane][k]) * xk[k];
      }
      partial_a[tok][lane] = sum_a;
      partial_b[tok][lane] = sum_b;
    }
    __syncthreads();

    if (lane == 0) {
      for (unsigned tok = warp; tok < cur_batch; tok += 8) {
        float total_a = 0.0f, total_b = 0.0f;
        for (unsigned i = 0; i < 32; i++) {
          total_a += partial_a[tok][i];
          total_b += partial_b[tok][i];
        }
        batch_out_a[(unsigned long long)tok * rows + row] = total_a;
        batch_out_b[(unsigned long long)tok * rows + row] = total_b;
      }
    }
  } else {
    for (unsigned tok = warp; tok < cur_batch; tok += 8) {
      unsigned long long chunk = ((unsigned long long)cols + 31) / 32;
      unsigned long long start = lane * chunk, end = start + chunk;
      if (end > cols) end = cols;
      float sum_a = 0.0f, sum_b = 0.0f;
      const float *xk = batch_x + (unsigned long long)tok * cols;
      for (unsigned long long i = start; i < end; i++) {
        sum_a += ds4_half_to_float(a[offset + i]) * xk[i];
        sum_b += ds4_half_to_float(b[offset + i]) * xk[i];
      }
      partial_a[tok][lane] = sum_a;
      partial_b[tok][lane] = sum_b;
    }
    __syncthreads();

    if (lane == 0) {
      for (unsigned tok = warp; tok < cur_batch; tok += 8) {
        float total_a = 0.0f, total_b = 0.0f;
        for (unsigned i = 0; i < 32; i++) {
          total_a += partial_a[tok][i];
          total_b += partial_b[tok][i];
        }
        batch_out_a[(unsigned long long)tok * rows + row] = total_a;
        batch_out_b[(unsigned long long)tok * rows + row] = total_b;
      }
    }
  }
}

extern "C" __global__ void ds4_f16_matvec_half_input_batch(
    const unsigned short *weights, const float *x, float *out,
    unsigned cols, unsigned stride, unsigned rows, unsigned batch) {
  const unsigned row = blockIdx.x;
  const unsigned t = blockIdx.y;
  const unsigned tid = threadIdx.x;
  if (row >= rows || t >= batch) return;

  const unsigned long long offset = (unsigned long long)row * cols;
  const float *token_x = x + (unsigned long long)t * stride;
  double sum = 0.0;
  for (unsigned i = tid; i < cols; i += blockDim.x)
    sum += double(ds4_half_to_float(weights[offset + i])) * double(ds4_half_round(token_x[i]));
  __shared__ double partial[256];
  partial[tid] = sum;
  __syncthreads();
  for (unsigned s = 128; s; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[(unsigned long long)t * rows + row] = float(partial[0]);
}
