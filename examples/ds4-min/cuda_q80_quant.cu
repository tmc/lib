
__device__ __forceinline__ void ds4_quant_q8_0_block(const float *x, ds4_q8_activation *q) {
  const unsigned tid = threadIdx.x;
  __shared__ float magnitude[32];
  magnitude[tid] = fabsf(x[tid]);
  __syncthreads();
  for (unsigned stride = 16; stride; stride >>= 1) {
    if (tid < stride) magnitude[tid] = fmaxf(magnitude[tid], magnitude[tid + stride]);
    __syncthreads();
  }
  const float d = magnitude[0] / 127.0f;
  const float inverse = d != 0.0f ? 1.0f / d : 0.0f;
  int value = (int)lrintf(x[tid] * inverse);
  value = value > 127 ? 127 : value < -128 ? -128 : value;
  q->qs[tid] = (signed char)value;
  if (tid == 0) q->d = d;
}

extern "C" __global__ void ds4_quant_q8_0(const float *in, ds4_q8_activation *out) {
  ds4_quant_q8_0_block(in + (unsigned long long)blockIdx.x * 32, out + blockIdx.x);
}

extern "C" __global__ void ds4_quant_q8_0_strided(const float *in, ds4_q8_activation *out,
                                                 unsigned blocks, unsigned stride) {
  const unsigned row = blockIdx.x / blocks;
  const unsigned block = blockIdx.x % blocks;
  ds4_quant_q8_0_block(in + (unsigned long long)row * stride + block * 32, out + blockIdx.x);
}
