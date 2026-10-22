// antirez/ds4 11e8cad: ds4_deepseek41_cuda.cuh:179-216.
// Device integer widths come only from compileCUDAV41Device's owned header.
static_assert(sizeof(unsigned short)==2,"Engram F16 word width");
__device__ float v41_engram_bf16(float x) {
    unsigned bits=__float_as_uint(x);
    if ((bits & 0x7f800000u)!=0x7f800000u) bits+=0x7fffu+((bits>>16)&1u);
    return __uint_as_float(bits&0xffff0000u);
}
extern "C" __global__ void ds4_v41_engram_half(unsigned short *out,const float *x,unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count){unsigned short h;asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x[i]));out[i]=h;}
}
extern "C" __global__ void ds4_v41_engram_round(float *x,unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)x[i]=v41_engram_bf16(x[i]);
}
extern "C" __global__ void ds4_v41_engram_add(float *residual,const float *kv,const float *qw,const float *kw,unsigned width) {
    const unsigned head=threadIdx.x/32u,lane=threadIdx.x&31u;
    const unsigned offset=head*width,key=head*width,value=4u*width;
    float h2=0,k2=0,dot=0;
    for(unsigned i=lane;i<width;i+=32u){
        float h=residual[offset+i],k=v41_engram_bf16(kv[key+i]);
        unsigned wi=head*width+i;
        h2+=h*h;k2+=k*k;dot+=h*(qw[wi]*kw[wi])*k;
    }
    for(unsigned delta=16;delta;delta>>=1){
        h2+=__shfl_down_sync(0xffffffffu,h2,delta);
        k2+=__shfl_down_sync(0xffffffffu,k2,delta);
        dot+=__shfl_down_sync(0xffffffffu,dot,delta);
    }
    h2=__shfl_sync(0xffffffffu,h2,0);k2=__shfl_sync(0xffffffffu,k2,0);
    dot=__shfl_sync(0xffffffffu,dot,0)*rsqrtf(h2/width+1e-20f)*rsqrtf(k2/width+1e-20f)*rsqrtf(float(width));
    const float signed_root=copysignf(sqrtf(fmaxf(fabsf(dot),1e-6f)),dot);
    const float gate=1.f/(1.f+expf(-signed_root));
    for(unsigned i=lane;i<width;i+=32u)residual[offset+i]=v41_engram_bf16(residual[offset+i]+gate*v41_engram_bf16(kv[value+i]));
}
