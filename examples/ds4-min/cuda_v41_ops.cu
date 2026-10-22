#include <stdint.h>

// Arithmetic contract: antirez/ds4 11e8cad, ds4_deepseek41_cuda.cuh.
// These deliberately serial scale groups preserve the reference rounding.
__device__ float v41_bf16(float x) {
    uint32_t b = __float_as_uint(x);
    if ((b & 0x7f800000u) != 0x7f800000u) b += 0x7fffu + ((b >> 16) & 1u);
    return __uint_as_float(b & 0xffff0000u);
}
__device__ float v41_mini(unsigned i, bool fp4) {
    const float code[8] = {0,.5f,1,1.5f,2,3,4,6};
    if (fp4) return code[i];
    unsigned e = i >> 3, m = i & 7;
    return e ? ldexpf(float(8+m), int(e)-10) : ldexpf(float(m), -9);
}
__device__ float v41_round(float x, bool fp4) {
    unsigned best = 0, n = fp4 ? 8 : 127;
    float distance = fabsf(x);
    for (unsigned i = 1; i < n; i++) {
        float d = fabsf(__fsub_rn(x, v41_mini(i, fp4)));
        if (d < distance || (d == distance && !(i&1) && (best&1))) { best=i; distance=d; }
    }
    return v41_mini(best, fp4);
}
__device__ float v41_pow2(float x) {
    unsigned b = __float_as_uint(x);
    return __uint_as_float((b & 0x7f800000u) + ((b & 0x7fffffu) ? 0x800000u : 0u));
}
extern "C" __global__ void ds4_v41_quantize(float *x, uint64_t count, unsigned mode) {
    uint64_t group = uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    unsigned block = mode == 0 ? 1 : mode == 3 ? 16 : 32;
    uint64_t start = group*block;
    if (start >= count) return;
    if (!mode) { x[start] = v41_bf16(x[start]); return; }
    float peak = 0;
    for (unsigned j=0; j<block; j++) peak=fmaxf(peak,fabsf(v41_bf16(x[start+j])));
    float scale = mode == 1 ? v41_pow2(__fmul_rn(fmaxf(peak,1e-4f),1.f/448)) :
        mode == 2 ? v41_pow2(__fmul_rn(fmaxf(peak,0x1.8p-124f),1.f/6)) :
        v41_round(__fdiv_rn(fmaxf(peak,.01171875f),6),false);
    for (unsigned j=0; j<block; j++) {
        float value=v41_bf16(x[start+j]);
        float q=v41_round(__fdiv_rn(fabsf(value),scale),mode != 1);
        x[start+j]=v41_bf16(__fmul_rn(copysignf(q,value),scale));
    }
}
extern "C" __global__ void ds4_v41_rope(float *x, const float *frequency,
    unsigned width, unsigned heads, unsigned rows, unsigned start, unsigned stride, unsigned inverse) {
    uint64_t pair=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (pair >= uint64_t(heads)*rows*32) return;
    unsigned lane=pair%32;
    uint64_t head=pair/32;
    unsigned row=head/heads;
    float theta=__fmul_rn(float(start+row*stride),frequency[lane]);
    float c=float(cos(double(theta))), s=float(sin(double(theta)));
    if (inverse) s=-s;
    uint64_t i=head*width+width-64+lane*2;
    float re=x[i], im=x[i+1];
    x[i]=v41_bf16(__fsub_rn(__fmul_rn(re,c),__fmul_rn(im,s)));
    x[i+1]=v41_bf16(__fadd_rn(__fmul_rn(re,s),__fmul_rn(im,c)));
}
