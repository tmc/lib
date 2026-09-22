#include <cuda_fp16.h>

__global__ void geglu(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = x[i];
  y[i] = g * 0.5f * (1.f + tanhf(g));
}

__global__ void rmsnorm(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  y[i] = v * rsqrtf(v * v + 1e-6f);
}
