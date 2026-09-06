// Deliberately invalid cases for Compute Sanitizer; run only under the checker.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

__global__ void read_input(const int *input, int *output) {
    output[threadIdx.x] = input[threadIdx.x];
}

__global__ void warp_barrier(int *output, unsigned mask) {
    // The bad case excludes lanes 16..31 from the mask while they still call
    // the barrier. All named lanes participate; no named lane is left waiting.
    __syncwarp(mask);
    output[threadIdx.x] = threadIdx.x;
}

static bool check(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return false;
}

int main(int argc, char **argv) {
    if (argc != 3 || (std::strcmp(argv[1], "initcheck") && std::strcmp(argv[1], "synccheck")) ||
        (std::strcmp(argv[2], "clean") && std::strcmp(argv[2], "bad"))) {
        std::fprintf(stderr, "usage: checkers initcheck|synccheck clean|bad\n");
        return 2;
    }
    bool init = !std::strcmp(argv[1], "initcheck");
    bool bad = !std::strcmp(argv[2], "bad");
    int *input = nullptr, *output = nullptr;
    int host[32];
    for (int i = 0; i < 32; ++i) host[i] = i;
    bool ok = check(cudaMalloc(&input, sizeof(host)), "allocate input");
    if (ok) ok = check(cudaMalloc(&output, sizeof(host)), "allocate output");
    if (ok && (!init || !bad))
        ok = check(cudaMemcpy(input, host, sizeof(host), cudaMemcpyHostToDevice), "initialize input");
    if (ok) {
        if (init) read_input<<<1, 32>>>(input, output);
        else warp_barrier<<<1, 32>>>(output, bad ? 0x0000ffffu : 0xffffffffu);
        ok = check(cudaGetLastError(), "launch");
        if (ok) {
            std::printf("%s %s launch submitted\n", argv[1], argv[2]);
            std::fflush(stdout);
            ok = check(cudaDeviceSynchronize(), "synchronize");
        }
    }
    if (ok) ok = check(cudaMemcpy(host, output, sizeof(host), cudaMemcpyDeviceToHost), "download");
    if (ok && !bad) {
        for (int i = 0; i < 32; ++i) {
            if (host[i] != i) {
                std::fprintf(stderr, "output[%d]=%d, want %d\n", i, host[i], i);
                ok = false;
            }
        }
    }
    if (output && !check(cudaFree(output), "free output")) ok = false;
    if (input && !check(cudaFree(input), "free input")) ok = false;
    if (!ok) return 1;
    std::printf("%s %s completed\n", argv[1], argv[2]);
    return 0;
}
