#include <cstdint>
__constant__ float table[64];
__device__ __noinline__ float helper(float x, int n) {
  float a[32];
  for (int i = 0; i < 32; i++) a[i] = x * i + table[i];
  return a[n & 31];
}
template <int N>
__global__ void scale(float* y, const float* x, int n) {
  __shared__ float tile[N];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  tile[threadIdx.x % N] = x[i % n];
  __syncthreads();
  y[i % n] = tile[(threadIdx.x + 1) % N] * 2 + helper(x[0], i);
}
__global__ void __launch_bounds__(1024, 2) spill(float* y, const float* x) {
  float acc[96];
  #pragma unroll
  for (int k = 0; k < 96; k++) acc[k] = x[threadIdx.x * 96 + k];
  #pragma unroll
  for (int r = 0; r < 8; r++)
    #pragma unroll
    for (int k = 0; k < 96; k++) acc[k] = acc[k] * acc[(k * 7 + r) % 96] + 1.0f;
  float s = 0;
  #pragma unroll
  for (int k = 0; k < 96; k++) s += acc[k];
  y[threadIdx.x] = s;
}
inline void launch_scale(float* y, const float* x, int n) { scale<128><<<1, 128>>>(y, x, n); }
