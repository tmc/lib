
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
#define DS4_F16_PAIR_MAX_BATCH 8

// ds4_f16_pair_ordered_chunks_batch projects batch token-major activation rows
// against the same weight pair with one launch, reading each weight element
// once for all tokens. Every token keeps the chunk partition, the ascending
// within-chunk order, and the ascending chunk-partial sum of the single-token
// kernel above, so each output is bit-identical to a single-token launch.
//
// One warp owns one token: lane l of the warp owns chunk l, exactly the
// partition the single-token kernel gives thread l, walked in the same
// ascending order and reduced by the same ascending sum of 32 partials. The
// batch therefore spreads across warps instead of serializing inside one, and
// the weights each warp reads are staged once for the whole block.
//
// The launcher passes Block(32*batch), so warp t owns token t. Grid is one
// block per row, so gridDim.x is the row count; out_a and out_b are
// token-major, row-minor.
//
// Measured on GB10 over the 2048-token prefill, nsys kernel summary, matched
// ABBA census of this exact body against the single-warp baseline, 15872
// launches in every arm: 11.07 s -> 4.68 s, 697.4 -> 294.9 us/launch. One warp
// per token without the float tiles gets only to ~6.5 s, because sharing the
// weight LOAD across warps makes each warp repeat the software half-to-float
// conversion over the same halves. No other kernel clears the 0.15 s
// threshold; the non-target deltas sum to -0.15 s and -0.02 s across two clean
// captures. Whole program: 30.5 t/s prefill and 59.8 s of device time at the
// baseline, 33.9 t/s and 53.2 s here. Bit-exactness is not a digest comparison
// but a byte-for-byte one: diff -r over the retained prefill capture trees for
// n=2,64,65,66,128,129,2048 differs only in identity.json.
extern "C" __global__ void ds4_f16_pair_ordered_chunks_batch(
    const unsigned short *a, const unsigned short *b, const float *x,
    float *out_a, float *out_b, unsigned cols, unsigned batch) {
  const unsigned row = blockIdx.x, tid = threadIdx.x;
  const unsigned lane = tid & 31, warp = tid >> 5;
  const unsigned token = warp;
  const unsigned long long offset = (unsigned long long)row * cols;
  const unsigned long long rows = gridDim.x;
  const unsigned long long token_off = (unsigned long long)token * cols;

  // One shared buffer per matrix, aliased two ways. The cols <= 4096 branches
  // stage raw halves through it; the cols == 4096 branch stages a tile of
  // already-converted floats through the same bytes. Sharing the storage is
  // what keeps the float tiles free: ptxas reports 18688 bytes of smem and 40
  // registers for this kernel either way, so residency per SM is unchanged and
  // the surrounding kernels are untouched.
  __shared__ float s_raw_a[32 * 130 / 2], s_raw_b[32 * 130 / 2];
  unsigned short (*s_a)[130] = (unsigned short (*)[130])s_raw_a;
  unsigned short (*s_b)[130] = (unsigned short (*)[130])s_raw_b;
  float (*s_af)[33] = (float (*)[33])s_raw_a;
  float (*s_bf)[33] = (float (*)[33])s_raw_b;
  __shared__ float partial_a[DS4_F16_PAIR_MAX_BATCH][32];
  __shared__ float partial_b[DS4_F16_PAIR_MAX_BATCH][32];

  if (cols == 4096) {
    // Hot path. Each weight element is converted to float ONCE per block and
    // staged as float, so the warps share the software half-to-float
    // conversion rather than each of them repeating it. Staging walks a tile
    // of 32 elements per chunk at a time so the float tile fits inside the
    // buffer the half staging already used.
    // Each lane still walks its own 128-element chunk in ascending order: the
    // tiles split that walk, they do not reorder it.
    float sum_a = 0.0f, sum_b = 0.0f;
    const float *xk = x + token_off + (lane << 7);
    for (unsigned t0 = 0; t0 < 128; t0 += 32) {
      __syncthreads();
      for (unsigned idx = tid; idx < 32 * 32; idx += blockDim.x) {
        const unsigned c = idx >> 5, j = idx & 31;
        const unsigned long long e = offset + ((unsigned long long)c << 7) + t0 + j;
        s_af[c][j] = ds4_half_to_float(a[e]);
        s_bf[c][j] = ds4_half_to_float(b[e]);
      }
      __syncthreads();
      if (token < batch) {
        for (unsigned j = 0; j < 32; j++) {
          sum_a += s_af[lane][j] * xk[t0 + j];
          sum_b += s_bf[lane][j] * xk[t0 + j];
        }
      }
    }
    if (token < batch) {
      partial_a[warp][lane] = sum_a;
      partial_b[warp][lane] = sum_b;
    }
    __syncthreads();

    if (lane == 0 && token < batch) {
      float total_a = 0.0f, total_b = 0.0f;
      for (unsigned i = 0; i < 32; i++) {
        total_a += partial_a[warp][i];
        total_b += partial_b[warp][i];
      }
      out_a[(unsigned long long)token * rows + row] = total_a;
      out_b[(unsigned long long)token * rows + row] = total_b;
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

    if (token < batch) {
      unsigned start = lane * chunk;
      unsigned count = (cols > start) ? (cols - start) : 0;
      if (count > chunk) count = chunk;
      const float *xk = x + token_off + start;
      float sum_a = 0.0f, sum_b = 0.0f;
      for (unsigned k = 0; k < count; k++) {
        sum_a += ds4_half_to_float(s_a[lane][k]) * xk[k];
        sum_b += ds4_half_to_float(s_b[lane][k]) * xk[k];
      }
      partial_a[warp][lane] = sum_a;
      partial_b[warp][lane] = sum_b;
    }
    __syncthreads();

    if (lane == 0 && token < batch) {
      float total_a = 0.0f, total_b = 0.0f;
      for (unsigned i = 0; i < 32; i++) {
        total_a += partial_a[warp][i];
        total_b += partial_b[warp][i];
      }
      out_a[(unsigned long long)token * rows + row] = total_a;
      out_b[(unsigned long long)token * rows + row] = total_b;
    }
  } else {
    if (token < batch) {
      unsigned long long chunk = ((unsigned long long)cols + 31) / 32;
      unsigned long long start = lane * chunk, end = start + chunk;
      if (end > cols) end = cols;
      float sum_a = 0.0f, sum_b = 0.0f;
      for (unsigned long long i = start; i < end; i++) {
        sum_a += ds4_half_to_float(a[offset + i]) * x[token_off + i];
        sum_b += ds4_half_to_float(b[offset + i]) * x[token_off + i];
      }
      partial_a[warp][lane] = sum_a;
      partial_b[warp][lane] = sum_b;
    }
    __syncthreads();

    if (lane == 0 && token < batch) {
      float total_a = 0.0f, total_b = 0.0f;
      for (unsigned i = 0; i < 32; i++) {
        total_a += partial_a[warp][i];
        total_b += partial_b[warp][i];
      }
      out_a[(unsigned long long)token * rows + row] = total_a;
      out_b[(unsigned long long)token * rows + row] = total_b;
    }
  }
}
