// Fixture kernels for the cudacapture tests. gen.sh compiles this file; the
// .ptx, .sass and .cubin.hex beside it are that compiler's output.
//
// Two entry points with different linkage, so the artifacts carry both
// spellings a CUBIN can hold: scale is a C++ symbol and reaches the CUBIN
// Itanium-mangled as _Z5scalePfPKff, while fill is extern "C" and reaches it
// verbatim. LoadCode joins the CUBIN section, the PTX entry and the SASS
// function on that symbol, so a unit spelling every kernel the same way on
// both sides never exercises the join.

__global__ void scale(float *out, const float *in, float k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = in[i] * k;
}

extern "C" __global__ void fill(int *out, int v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = v;
}
