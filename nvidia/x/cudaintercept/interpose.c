/*
 * Dual-API Symbol Interposition Shim for libcuda.so.1 / libcudart.so
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <pthread.h>

typedef int (*cuLaunchKernel_t)(void* f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, void* hStream, void** kernelParams, void** extra);

static cuLaunchKernel_t real_cuLaunchKernel = NULL;
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static _Thread_local int resolving;

static void resolve_symbol(void) {
    if (resolving) return;
    resolving = 1;
    real_cuLaunchKernel = (cuLaunchKernel_t)dlsym(RTLD_NEXT, "cuLaunchKernel");
    resolving = 0;
}

int cuLaunchKernel(void* f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                   unsigned int sharedMemBytes, void* hStream, void** kernelParams, void** extra) {
    pthread_once(&resolve_once, resolve_symbol);
    if (!real_cuLaunchKernel) return 3; /* CUDA_ERROR_NOT_INITIALIZED: fail closed. */
    return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}
