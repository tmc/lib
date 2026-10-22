// Q4_K weights use the GGUF 144-byte block; activations use Q8_K.
// The integer partials and packed scale/min halves follow ds4's Q4_K dot.
__device__ float ds4_q4_block_q8_k(const unsigned char *p, const ds4_q8_k *x) {
  int total = 0, minimum = 0;
  const unsigned char *scales = p + 4;
  for (unsigned g = 0; g < 8; g++) {
    unsigned scale, mn;
    if (g < 4) { scale = scales[g] & 63; mn = scales[g + 4] & 63; }
    else {
      scale = (scales[g + 4] & 15) | ((scales[g - 4] >> 6) << 4);
      mn = (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4);
    }
    int dot = 0;
    for (unsigned i = 0; i < 32; i++) {
      unsigned q = (p[16 + (g / 2) * 32 + i] >> ((g & 1) * 4)) & 15;
      dot += int(q) * int(x->qs[g * 32 + i]);
    }
    total += int(scale) * dot;
    minimum += int(mn) * (int(x->bsums[2 * g]) + int(x->bsums[2 * g + 1]));
  }
  float d = ds4_half_to_float(unsigned(p[0]) | unsigned(p[1]) << 8);
  float dm = ds4_half_to_float(unsigned(p[2]) | unsigned(p[3]) << 8);
  return x->d * (d * float(total) - dm * float(minimum));
}

extern "C" __global__ void ds4_q4_k_selected_q8_k(
    const unsigned char *weights, const int *ids, const ds4_q8_k *x, float *out,
    unsigned cols, unsigned rows, unsigned experts, unsigned slotted) {
  unsigned row = blockIdx.x % rows, slot = blockIdx.x / rows;
  int expert = ids[slot];
  if (expert < 0 || unsigned(expert) >= experts) {
    if (threadIdx.x == 0) out[blockIdx.x] = __int_as_float(0x7fffffff);
    return;
  }
  unsigned blocks = cols / 256;
  unsigned long long offset = (static_cast<unsigned long long>(expert) * rows + row) * blocks * 144;
  const ds4_q8_k *input = x + (slotted ? slot * blocks : 0);
  float sum = 0;
  for (unsigned b = threadIdx.x; b < blocks; b += blockDim.x)
    sum += ds4_q4_block_q8_k(weights + offset + (unsigned long long)b * 144, input + b);
  __shared__ float partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = partial[0];
}
