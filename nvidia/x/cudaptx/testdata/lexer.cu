// Fixture source for x/cudaptx lexer tests. Each construct below is chosen to
// exercise one PTX syntax rule; see x/cudaptx/testdata/README.
#include <cuda_runtime.h>

__device__ __forceinline__ float scale(float x, float k) {
  return x * k + 1.0f;  // inlined -> .loc ... inlined_at
}

extern "C" __device__ float externalHelper(float x);

// .visible .entry with .param, predicated branch, .loc chain.
extern "C" __global__ void guarded(const float *in, float *out, int n, float k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = scale(in[i], k);
  }
}

// __launch_bounds__ -> .maxntid / .minnctapersm
extern "C" __global__ __launch_bounds__(256, 4) void bounded(float *p) {
  p[threadIdx.x] += 1.0f;
}

// nvcc normalizes __launch_bounds__ to all three .maxntid dimensions, so the
// emitted directive is always "nx, ny, nz" even for a 1D bound.
extern "C" __global__ __launch_bounds__(64) void shaped3d(float *p) {
  int i = threadIdx.z * 16 * 16 + threadIdx.y * 16 + threadIdx.x;
  p[i] = (float)i;
}

// Struct-by-value param -> .param .align N .b8 name[size]
struct Big { float a[8]; int b; };
extern "C" __global__ void byvalue(Big b, float *out) {
  out[threadIdx.x] = b.a[threadIdx.x & 7] + (float)b.b;
}

// Non-inlined device function -> .visible .func with return param,
// plus a call to an extern declaration -> bodiless .extern .func.
extern "C" __device__ __noinline__ float notInlined(float x) {
  return externalHelper(x) * 2.0f;
}
extern "C" __global__ void calls(float *p) {
  p[threadIdx.x] = notInlined(p[threadIdx.x]);
}
