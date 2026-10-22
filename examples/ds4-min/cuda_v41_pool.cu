#include <stdint.h>

// Pair arithmetic: antirez/ds4 11e8cad ds4_deepseek41_cuda.cuh:219-257.
__device__ float v41_pool_bf16(float x) {
    uint32_t b=__float_as_uint(x);
    if((b&0x7f800000u)!=0x7f800000u)b+=0x7fffu+((b>>16)&1u);
    return __uint_as_float(b&0xffff0000u);
}
extern "C" __global__ void ds4_v41_pool(float *out,const float *kv,const float *scores,
    const float *prev_kv,const float *prev_scores,unsigned width,unsigned pairs,unsigned tail) {
    uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=uint64_t(width)*pairs)return;
    unsigned col=i%width;
    int64_t row=int64_t(i/width)*2-tail;
    uint64_t b=uint64_t(row+1)*width+col;
    float ka=row<0?prev_kv[col]:kv[uint64_t(row)*width+col];
    float sa=row<0?prev_scores[col]:scores[uint64_t(row)*width+col];
    float sb=scores[b],peak=fmaxf(sa,sb);
    float ea=expf(__fsub_rn(sa,peak)),eb=expf(__fsub_rn(sb,peak));
    out[i]=v41_pool_bf16(__fdiv_rn(__fadd_rn(__fmul_rn(ka,ea),__fmul_rn(kv[b],eb)),__fadd_rn(ea,eb)));
}
// Separate launch: carry writes must happen after every pooling read completes.
extern "C" __global__ void ds4_v41_pool_carry(float *prev_kv,float *prev_scores,
    const float *kv,const float *scores,unsigned width,unsigned row,unsigned clear) {
    unsigned col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=width)return;
    prev_kv[col]=clear?0:kv[uint64_t(row)*width+col];
    prev_scores[col]=clear?0:scores[uint64_t(row)*width+col];
}
