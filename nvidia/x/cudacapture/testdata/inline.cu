// A unit whose kernel inlines a device function from a header. A capture
// bundles the .cu files of its unit and not the headers, so the disassembly
// names a file the capture has no line table for: those lines are unknowable
// rather than unclaimed, and an attribution to them must survive.
#include "helpers.cuh"

__global__ void blend(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = clamp01(in[i]);
    }
}
