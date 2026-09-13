
extern "C" __global__ void ds4_quant_q8_0(const float *in, ds4_q8_activation *out) {
  const unsigned tid = threadIdx.x;
  const float *x = in + (unsigned long long)blockIdx.x * 32;
  ds4_q8_activation *q = out + blockIdx.x;
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

