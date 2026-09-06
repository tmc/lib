// cudashim.h declares just enough of the CUDA device environment for a host
// C++ compiler to parse forward.cu.
//
// It exists so that template, macro and signature errors in the kernel source
// are caught on any machine, without a GPU and without NVRTC. It is a syntax
// harness only: nothing here computes anything, and passing it says nothing
// about numerical behavior.
#pragma once

#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__
#define __shared__ static

struct dim3_stub { int x, y, z; };
static dim3_stub blockIdx, threadIdx, blockDim, gridDim;

struct float4 { float x, y, z, w; };
struct ushort4 { unsigned short x, y, z, w; };
static inline float4 make_float4(float x, float y, float z, float w) {
    float4 v; v.x = x; v.y = y; v.z = z; v.w = w; return v;
}
static inline float __shfl_down_sync(unsigned, float v, int) { return v; }
static inline float __uint_as_float(unsigned int) { return 0.0f; }
static inline float rsqrtf(float) { return 0.0f; }
static inline float expf(float) { return 0.0f; }
static inline float powf(float, float) { return 0.0f; }
static inline float cosf(float) { return 0.0f; }
static inline float sinf(float) { return 0.0f; }
static inline void __syncthreads() {}
