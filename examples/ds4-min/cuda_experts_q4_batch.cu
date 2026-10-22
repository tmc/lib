// Expert-major CSR projections. Pair numbers remain token-major in outputs;
// the gate/up input is per token, while the down input is per selected slot.
// Each pair keeps the decode kernel's 256-thread block reduction order.
template<bool paired>
__device__ void ds4_q4_k_gather_body(
    const unsigned char *weights, const unsigned char *up,
    const int *offsets, const int *pairs, const ds4_q8_k *input,
    float *out, float *up_out, unsigned cols, unsigned rows,
    unsigned experts, unsigned slots, unsigned npairs) {
  unsigned row = blockIdx.x, expert = blockIdx.y;
  unsigned blocks = cols / 256;
  __shared__ float partial[256], up_partial[256];
  for (int at = offsets[expert]; at < offsets[expert + 1]; at++) {
    int pair = pairs[at];
    if (pair < 0 || unsigned(pair) >= npairs) continue;
    unsigned long long dest = static_cast<unsigned long long>(pair) * rows + row;
    if (expert == experts) {
      if (threadIdx.x == 0) {
        out[dest] = __int_as_float(0x7fffffff);
        if (paired) up_out[dest] = __int_as_float(0x7fffffff);
      }
      continue;
    }
    unsigned long long offset = (static_cast<unsigned long long>(expert) * rows + row) * blocks * 144;
    const ds4_q8_k *x = input + static_cast<unsigned long long>(unsigned(pair) / slots) * blocks;
    float sum = 0, up_sum = 0;
    for (unsigned b = threadIdx.x; b < blocks; b += blockDim.x) {
      sum += ds4_q4_block_q8_k(weights + offset + static_cast<unsigned long long>(b) * 144, x + b);
      if (paired) up_sum += ds4_q4_block_q8_k(up + offset + static_cast<unsigned long long>(b) * 144, x + b);
    }
    partial[threadIdx.x] = sum;
    if (paired) up_partial[threadIdx.x] = up_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
        if (paired) up_partial[threadIdx.x] += up_partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      out[dest] = partial[0];
      if (paired) up_out[dest] = up_partial[0];
    }
    // Protect shared reductions before the next selected pair writes them.
    __syncthreads();
  }
}

extern "C" __global__ void ds4_q4_k_pair_selected_q8_k_gather(
    const unsigned char *gate, const unsigned char *up, const int *offsets,
    const int *pairs, const ds4_q8_k *input, float *gate_out, float *up_out,
    unsigned cols, unsigned rows, unsigned experts, unsigned slots, unsigned npairs) {
  ds4_q4_k_gather_body<true>(gate, up, offsets, pairs, input, gate_out, up_out, cols, rows, experts, slots, npairs);
}

extern "C" __global__ void ds4_q4_k_selected_q8_k_gather(
    const unsigned char *weights, const int *offsets, const int *pairs,
    const ds4_q8_k *input, float *out, unsigned cols, unsigned rows,
    unsigned experts, unsigned npairs) {
  ds4_q4_k_gather_body<false>(weights, nullptr, offsets, pairs, input, out, nullptr, cols, rows, experts, 1, npairs);
}
