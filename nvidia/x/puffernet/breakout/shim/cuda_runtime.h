// NVRTC stand-in for <cuda_runtime.h>. NVRTC compiles breakout.cu with
// -default-device, so its host code becomes device code that is never
// called; these macros make the runtime calls in it no-ops.
#pragma once
#define cudaMalloc(p, n) 0
#define cudaFree(p) 0
#define cudaMemcpy(dst, src, n, kind) 0
#define cudaMemcpyToSymbol(sym, src, n) 0
#define cudaStreamSynchronize(s) 0

// puf_reset and puf_step launch kernels, which in device code needs the
// device runtime. These stand-ins let them compile.
extern "C" __device__ void* cudaGetParameterBuffer(size_t alignment, size_t size) {
    return 0;
}
extern "C" __device__ void* cudaGetParameterBufferV2(void* func, dim3 grid, dim3 block, unsigned int shared) {
    return 0;
}
extern "C" __device__ cudaError_t cudaLaunchDevice(void* func, void* params, dim3 grid, dim3 block, unsigned int shared, cudaStream_t stream) {
    return cudaErrorNotSupported;
}
extern "C" __device__ cudaError_t cudaLaunchDeviceV2(void* params, cudaStream_t stream) {
    return cudaErrorNotSupported;
}
