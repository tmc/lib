// A header of the inline.cu translation unit, included rather than compiled.
// Its body is force-inlined into the kernel, so the disassembler attributes
// part of that kernel to this file and part to inline.cu.

__device__ __forceinline__ float clamp01(float v) {
    if (v < 0.0f) {
        return 0.0f;
    }
    if (v > 1.0f) {
        return 1.0f;
    }
    return v;
}
