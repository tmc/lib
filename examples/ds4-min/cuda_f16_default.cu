
extern "C" __global__ void ds4_f16_matvec_half_input(
    const unsigned short *weights, const float *x, float *out, unsigned cols) {
  const unsigned row = blockIdx.x, tid = threadIdx.x;
  const unsigned long long offset = (unsigned long long)row * cols;
  // Single precision with contraction pinned, for the reasons given at
  // ds4_f16_matvec_half_input_batch below; decode runs this 84 times a token.
  float sum = 0.0f;
  for (unsigned i = tid; i < cols; i += blockDim.x)
    sum = __fmaf_rn(ds4_half_to_float(weights[offset + i]), ds4_half_round(x[i]), sum);
  __shared__ float partial[256];
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


// ds4_f32_to_f16 narrows a batch of activation rows so the library GEMM path can
// read them. The kernels it replaces multiply half weights by float activations;
// a cuBLAS GemmEx wants both operands in one type, and half is the side that
// makes the weights readable without widening them, which would double the
// traffic the GEMM exists to reduce.
//
// Source rows are src_stride elements apart and the destination is packed, so a
// caller projecting one half of an interleaved pre block passes the block stride
// and a pointer to its own half. n is the packed element count, rows * cols.
// The grid is two-dimensional -- one row per blockIdx.y -- so that a thread
// recovers its row and column from the launch geometry. A flat index would have
// to divide a 64-bit element number by cols, and at several million elements per
// launch that division costs more than the conversion it indexes.
extern "C" __global__ void ds4_f32_to_f16(const float *in, unsigned short *out,
                                          unsigned cols, unsigned src_stride) {
  const float *src = in + (unsigned long long)blockIdx.y * src_stride;
  unsigned short *dst = out + (unsigned long long)blockIdx.y * cols;
  for (unsigned c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x)
    dst[c] = ds4_q81_half_bits(src[c]);
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
// The launcher passes Block(256), so 8 warps stride over the tile's tokens,
// warp + 8s for s in [0, DS4_F16_PAIR_MAX_BATCH/8). Grid is
// (rows, ceil(batch/DS4_F16_PAIR_MAX_BATCH)), so gridDim.x is the row count and
// gridDim.y is the batch tiles; out_a and out_b are token-major, row-minor.
//
// The tile decides how often the weights are re-read: a prefill group of 2048
// tokens reads every weight element ceil(2048/tile) times, so the tile and not
// the group width is this kernel's redundancy term. Raising it does not pay.
// Measured at prefill width 2048, this kernel's own device time was 4.40 s at
// tile 16 and 7.30 s at tile 64 -- 66% worse for a quarter of the weight
// traffic. Shared memory per block goes 20.7 KB to 33.0 KB across that range
// and halves the blocks resident per SM, and the activation reads, which are
// what the tile does not reduce, are the traffic that matters here. 16 stays.
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

  // One accumulator pair per token slot this warp owns. Each token's addends
  // keep the order the single-token kernel uses -- ascending j inside a 32-wide
  // lane chunk, then ascending chunks -- so widening the tile adds slots
  // without touching any token's summation order. The accumulate is pinned to
  // __fmaf_rn instead of left as a * b + c: nvcc chooses contraction per site
  // under the register pressure of the moment, a wider tile changes that
  // pressure, and an unpinned version therefore changes the logits when the
  // tile moves -- which is what an earlier attempt at this hit. Pinned, every
  // tile from 16 to 64 gives the same bits.
  constexpr unsigned SLOTS = DS4_F16_PAIR_MAX_BATCH / 8;

  if (cols == 4096) {
    float sum_a[SLOTS], sum_b[SLOTS];
    const float *xk[SLOTS];
#pragma unroll
    for (unsigned s = 0; s < SLOTS; s++) {
      const unsigned tok = warp + s * 8;
      sum_a[s] = 0.0f;
      sum_b[s] = 0.0f;
      xk[s] = (tok < cur_batch) ? (batch_x + (unsigned long long)tok * cols + (lane << 7)) : nullptr;
    }

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
#pragma unroll
        for (unsigned s = 0; s < SLOTS; s++) {
          if (xk[s]) {
            const float xv = xk[s][t0 + j];
            sum_a[s] = __fmaf_rn(wa, xv, sum_a[s]);
            sum_b[s] = __fmaf_rn(wb, xv, sum_b[s]);
          }
        }
      }
    }
#pragma unroll
    for (unsigned s = 0; s < SLOTS; s++) {
      const unsigned tok = warp + s * 8;
      if (tok < cur_batch) {
        partial_a[tok][lane] = sum_a[s];
        partial_b[tok][lane] = sum_b[s];
      }
    }
    __syncthreads();

    if (lane == 0) {
#pragma unroll
      for (unsigned s = 0; s < SLOTS; s++) {
        const unsigned tok = warp + s * 8;
        if (tok < cur_batch) {
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
        sum_a = __fmaf_rn(ds4_half_to_float(s_a[lane][k]), xk[k], sum_a);
        sum_b = __fmaf_rn(ds4_half_to_float(s_b[lane][k]), xk[k], sum_b);
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
        sum_a = __fmaf_rn(ds4_half_to_float(a[offset + i]), xk[i], sum_a);
        sum_b = __fmaf_rn(ds4_half_to_float(b[offset + i]), xk[i], sum_b);
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
  // Single precision, matching the reference, which accumulates these dot
  // products in float throughout (ds4.c:3877 and some twenty sites like it).
  // The double this replaced was precision the reference never had, and sm_121
  // runs FP64 at a fraction of the FP32 rate: it cost about 1.0s of the 12.7s
  // prefill spends on the device. Contraction is pinned for the same reason as
  // the pair kernel above -- so that later changes to this loop cannot move the
  // results by moving nvcc's contraction choice.
  float sum = 0.0f;
  for (unsigned i = tid; i < cols; i += blockDim.x)
    sum = __fmaf_rn(ds4_half_to_float(weights[offset + i]), ds4_half_round(token_x[i]), sum);
  __shared__ float partial[256];
  partial[tid] = sum;
  __syncthreads();
  for (unsigned s = 128; s; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[(unsigned long long)t * rows + row] = float(partial[0]);
}
