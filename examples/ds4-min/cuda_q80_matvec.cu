
// DS4_Q80_MAX_BATCH is the compile-time maximum batch width of
// ds4_q8_0_matvec_quantized_batch. cudaQ80MaxBatch in cuda_q80_linux.go must
// hold the same value; TestCUDAQualificationQ80MaxBatchConstantSync asserts it.
#define DS4_Q80_MAX_BATCH 32

// Small speculative verification batches use a narrower accumulator tile.  In
// particular, keeping this separate from the 32-token prefill kernel avoids
// reserving registers for tokens that are not present in a decode batch.
#define DS4_Q80_SMALL_MAX_BATCH 8

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

// ds4_q8_0_matvec_quantized_aligned is ds4_q8_0_matvec_quantized reading the
// same weights out of the dense artifact's split dq/qs planes in device memory
// instead of the model's 34-byte GGUF blocks in the host mapping. It is the
// float-activation twin of ds4_q8_0_matvec_q8_1_aligned in cuda_q81.cu, and the
// bit-exactness argument there applies here unchanged: b = lane and the b += 32
// stride are kept, so each lane sums the same blocks in the same order; the
// eight __dp4a steps stay in order into an int dot; the scale comes back
// through ds4_half_to_float from the same half bits ds4_repack_q8_0_aligned
// copied out of the raw block. Only the fetch moves.
extern "C" __global__ void ds4_q8_0_matvec_quantized_aligned(
    const unsigned short *dq, const unsigned char *qs,
    const ds4_q8_activation *x, float *out,
    unsigned blocks, unsigned rows) {
  const unsigned warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned row = blockIdx.x * 8 + warp_id;
  if (row >= rows) return;
  float sum = 0.0f;
  for (unsigned b = lane; b < blocks; b += 32) {
    const unsigned long long blk = (unsigned long long)row * blocks + b;
    const int4 *qv = reinterpret_cast<const int4 *>(qs + blk * 32ull);
    const int4 q0 = qv[0], q1 = qv[1];
    const int w16[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    const int *x4 = reinterpret_cast<const int *>(x[b].qs);
    int dot = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int w4 = w16[i];
      dot = __dp4a(w4, x4[i], dot);
    }
    const float d = ds4_half_to_float(dq[blk]);
    sum += d * x[b].d * float(dot);
  }
  for (unsigned stride = 16; stride; stride >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  if (lane == 0) out[row] = sum;
}

// ds4_q8_0_matvec_quantized_blockdiag_aligned is
// ds4_q8_0_matvec_quantized_blockdiag with the same fetch change. The block
// diagonal is eight separate tensors to buildDenseArtifact, which repacks each
// into its own aligned plane, so a diagonal here is a byte stride rather than a
// row offset: plane_bytes is one plane's total size and dq_bytes its dq half.
extern "C" __global__ void ds4_q8_0_matvec_quantized_blockdiag_aligned(
    const unsigned char *planes, const ds4_q8_activation *x, float *out,
    unsigned blocks, unsigned rows,
    unsigned long long plane_bytes, unsigned long long dq_bytes) {
  const unsigned warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned row = blockIdx.x * 8 + warp_id, diag = blockIdx.y;
  if (row >= rows) return;
  const unsigned char *plane = planes + (unsigned long long)diag * plane_bytes;
  const unsigned short *dq = reinterpret_cast<const unsigned short *>(plane);
  const unsigned char *qs = plane + dq_bytes;
  const ds4_q8_activation *xb = x + (unsigned long long)diag * blocks;
  float sum = 0.0f;
  for (unsigned b = lane; b < blocks; b += 32) {
    const unsigned long long blk = (unsigned long long)row * blocks + b;
    const int4 *qv = reinterpret_cast<const int4 *>(qs + blk * 32ull);
    const int4 q0 = qv[0], q1 = qv[1];
    const int w16[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    const int *x4 = reinterpret_cast<const int *>(xb[b].qs);
    int dot = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int w4 = w16[i];
      dot = __dp4a(w4, x4[i], dot);
    }
    const float d = ds4_half_to_float(dq[blk]);
    sum += d * xb[b].d * float(dot);
  }
  for (unsigned stride = 16; stride; stride >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  if (lane == 0) out[(unsigned long long)diag * rows + row] = sum;
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
  const unsigned t_base = blockIdx.y * DS4_Q80_MAX_BATCH;

  if (row >= rows || t_base >= batch) return;
  const unsigned cur_batch = (batch - t_base < DS4_Q80_MAX_BATCH) ? (batch - t_base) : DS4_Q80_MAX_BATCH;

  const unsigned char *weight_row = weights + (unsigned long long)row * blocks * 34;
  const ds4_q8_activation *batch_x = x + (unsigned long long)t_base * blocks;
  float *batch_out = out + (unsigned long long)t_base * rows;

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
      if (t < cur_batch) {
        const ds4_q8_activation *tx = batch_x + (unsigned long long)t * blocks;
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
    if (t < cur_batch) {
      float s = sum[t];
      #pragma unroll
      for (unsigned stride = 16; stride; stride >>= 1)
        s += __shfl_down_sync(0xffffffff, s, stride);
      if (lane == 0) batch_out[(unsigned long long)t * rows + row] = s;
    }
  }
}

// ds4_q8_0_matvec_quantized_small is the <=8-token form used by speculative
// verification.  It preserves the unbatched accumulation order for each
// token, but does not carry the 32-token accumulator array used by prefill.
extern "C" __global__ void ds4_q8_0_matvec_quantized_small(
    const unsigned char *weights, const ds4_q8_activation *x, float *out,
    unsigned blocks, unsigned rows, unsigned batch) {
  const unsigned warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned row = blockIdx.x * 8 + warp_id;
  if (row >= rows || batch == 0 || batch > DS4_Q80_SMALL_MAX_BATCH) return;

  const unsigned char *weight_row = weights + (unsigned long long)row * blocks * 34;
  float sum[DS4_Q80_SMALL_MAX_BATCH] = {0.0f};
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
    for (unsigned t = 0; t < DS4_Q80_SMALL_MAX_BATCH; t++) {
      if (t < batch) {
        const int *x4 = reinterpret_cast<const int *>(x[(unsigned long long)t * blocks + b].qs);
        int dot = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) dot = __dp4a(w4[i], x4[i], dot);
        sum[t] += d * x[(unsigned long long)t * blocks + b].d * float(dot);
      }
    }
  }
  #pragma unroll
  for (unsigned t = 0; t < DS4_Q80_SMALL_MAX_BATCH; t++) {
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

// DS4_Q80_BLOCKDIAG_MAX_BATCH is the compile-time batch tile width of
// ds4_q8_0_matvec_quantized_blockdiag_batch. cudaQ80BlockDiagMaxBatch in
// cuda_q80_linux.go must hold the same value;
// TestCUDAQualificationQ80BlockDiagMaxBatchConstantSync asserts it.
#define DS4_Q80_BLOCKDIAG_MAX_BATCH 8

// ds4_q8_0_matvec_quantized_blockdiag_batch is
// ds4_q8_0_matvec_quantized_blockdiag with the weight load hoisted out of a
// token loop. The weight row a warp reads is addressed by (diag, row) alone, so
// every token in a batch reads the same weight bytes; the batch streams the
// block-diagonal weight tensor once per tile instead of once per token.
// Tiles of up to DS4_Q80_BLOCKDIAG_MAX_BATCH tokens run along gridDim.z.
//
// Each token keeps its own accumulator, its own lane-strided walk over b, and
// its own shfl_down reduction, so no term moves within any sum and the float32
// results are bit-identical to one blockdiag launch per token. Activations and
// outputs are token major: token t's activations start at t * diags * blocks
// blocks, and its outputs at t * diags * rows.
extern "C" __global__ void ds4_q8_0_matvec_quantized_blockdiag_batch(
    const unsigned char *weights, const ds4_q8_activation *x, float *out,
    unsigned blocks, unsigned rows, unsigned diags, unsigned batch) {
  const unsigned warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned row = blockIdx.x * 8 + warp_id, diag = blockIdx.y;
  const unsigned t_base = blockIdx.z * DS4_Q80_BLOCKDIAG_MAX_BATCH;
  if (row >= rows || t_base >= batch) return;
  const unsigned cur_batch = (batch - t_base < DS4_Q80_BLOCKDIAG_MAX_BATCH)
      ? batch - t_base : DS4_Q80_BLOCKDIAG_MAX_BATCH;
  const unsigned char *weight_row = weights + ((unsigned long long)diag * rows + row) * blocks * 34;
  const ds4_q8_activation *xb = x + ((unsigned long long)t_base * diags + diag) * blocks;
  const unsigned long long token_blocks = (unsigned long long)diags * blocks;
  float sum[DS4_Q80_BLOCKDIAG_MAX_BATCH] = {0.0f};
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
    for (unsigned t = 0; t < DS4_Q80_BLOCKDIAG_MAX_BATCH; t++) {
      if (t < cur_batch) {
        const ds4_q8_activation *tx = xb + (unsigned long long)t * token_blocks;
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
  for (unsigned t = 0; t < DS4_Q80_BLOCKDIAG_MAX_BATCH; t++) {
    if (t < cur_batch) {
      float s = sum[t];
      #pragma unroll
      for (unsigned stride = 16; stride; stride >>= 1)
        s += __shfl_down_sync(0xffffffff, s, stride);
      if (lane == 0)
        out[((unsigned long long)(t_base + t) * diags + diag) * rows + row] = s;
    }
  }
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
