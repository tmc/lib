/*
 * Dual-API Symbol Interposition Shim for libcuda.so.1 / libcudart.so
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>

typedef int (*cuLaunchKernel_t)(void* f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, void* hStream, void** kernelParams, void** extra);

static cuLaunchKernel_t real_cuLaunchKernel = NULL;

int cuLaunchKernel(void* f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                   unsigned int sharedMemBytes, void* hStream, void** kernelParams, void** extra) {
    if (!real_cuLaunchKernel) {
        real_cuLaunchKernel = (cuLaunchKernel_t)dlsym(RTLD_NEXT, "cuLaunchKernel");
    }
    // Pass-through execution
    return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}
