
struct ds4_q8_1_activation { unsigned short d, sum; signed char qs[32]; };
static_assert(sizeof(ds4_q8_1_activation) == 36, "Q8_1 block size");

// ds4_half_round already implements the binary16 nearest-even boundary. Encode
// its rounded float result without another rounding step or toolkit headers.
__device__ unsigned short ds4_q81_half_bits(float value) {
  unsigned bits = __float_as_uint(ds4_half_round(value));
  unsigned sign = (bits >> 16) & 0x8000, mag = bits & 0x7fffffff;
  if (mag >= 0x7f800000) return (unsigned short)(sign | 0x7c00 | ((mag & 0x7fffff) ? 0x200 : 0));
  if (mag == 0) return (unsigned short)sign;
  int exponent = int(mag >> 23) - 127;
  unsigned mantissa = mag & 0x7fffff;
  if (exponent >= -14) return (unsigned short)(sign | unsigned(exponent + 15) << 10 | mantissa >> 13);
  return (unsigned short)(sign | ((mantissa | 0x800000) >> (-exponent - 1)));
}

extern "C" __global__ void ds4_quant_q8_1(const float *in, ds4_q8_1_activation *out) {
  const unsigned tid = threadIdx.x;
  float value = in[(unsigned long long)blockIdx.x * 32 + tid];
  float magnitude = fabsf(value), sum = value;
  for (unsigned stride = 16; stride; stride >>= 1) {
    magnitude = fmaxf(magnitude, __shfl_xor_sync(0xffffffff, magnitude, stride));
    sum += __shfl_xor_sync(0xffffffff, sum, stride);
  }
  float d = magnitude / 127.0f;
  out[blockIdx.x].qs[tid] = magnitude == 0.0f ? 0 : (signed char)roundf(value / d);
  if (tid == 0) { out[blockIdx.x].d = ds4_q81_half_bits(d); out[blockIdx.x].sum = ds4_q81_half_bits(sum); }
}

extern "C" __global__ void ds4_quant_q8_1_batch(
    const float *in, ds4_q8_1_activation *out,
    unsigned blocks, unsigned stride) {
  const unsigned flat = blockIdx.x, token = flat / blocks, block = flat % blocks;
  const unsigned tid = threadIdx.x;
  float value = in[(unsigned long long)token * stride + (unsigned long long)block * 32 + tid];
  float magnitude = fabsf(value), sum = value;
  for (unsigned offset = 16; offset; offset >>= 1) {
    magnitude = fmaxf(magnitude, __shfl_xor_sync(0xffffffff, magnitude, offset));
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  float d = magnitude / 127.0f;
  out[flat].qs[tid] = magnitude == 0.0f ? 0 : (signed char)roundf(value / d);
  if (tid == 0) { out[flat].d = ds4_q81_half_bits(d); out[flat].sum = ds4_q81_half_bits(sum); }
}

extern "C" __global__ void ds4_q8_0_matvec_q8_1(
    const unsigned char *weights, const ds4_q8_1_activation *x, float *out,
    unsigned blocks) {
  const unsigned row = blockIdx.x, lane = threadIdx.x;
  const unsigned char *weight_row = weights + (unsigned long long)row * blocks * 34;
  float sum = 0.0f;
  for (unsigned b = lane; b < blocks; b += 32) {
    const unsigned char *p = weight_row + (unsigned long long)b * 34;
    const signed char *w = reinterpret_cast<const signed char *>(p + 2);
    int dot = 0;
    for (unsigned i = 0; i < 32; i++) dot += int(w[i]) * int(x[b].qs[i]);
    const float d = ds4_half_to_float(unsigned(p[0]) | unsigned(p[1]) << 8);
    sum += d * ds4_half_to_float(x[b].d) * float(dot);
  }
  for (unsigned stride = 16; stride; stride >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  if (lane == 0) out[row] = sum;
}

extern "C" __global__ void ds4_q8_0_matvec_q8_1_batch(
    const unsigned char *weights, const ds4_q8_1_activation *x, float *out,
    unsigned blocks, unsigned rows, unsigned batch) {
  const unsigned row = blockIdx.x, token = blockIdx.y, lane = threadIdx.x;
  if (row >= rows || token >= batch) return;
  const unsigned char *weight_row = weights + (unsigned long long)row * blocks * 34;
  const ds4_q8_1_activation *token_x = x + (unsigned long long)token * blocks;
  float sum = 0.0f;
  for (unsigned b = lane; b < blocks; b += 32) {
    const unsigned char *p = weight_row + (unsigned long long)b * 34;
    const signed char *w = reinterpret_cast<const signed char *>(p + 2);
    int dot = 0;
    for (unsigned i = 0; i < 32; i++) dot += int(w[i]) * int(token_x[b].qs[i]);
    const float d = ds4_half_to_float(unsigned(p[0]) | unsigned(p[1]) << 8);
    sum += d * ds4_half_to_float(token_x[b].d) * float(dot);
  }
  for (unsigned stride = 16; stride; stride >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  if (lane == 0) out[(unsigned long long)token * rows + row] = sum;
}
