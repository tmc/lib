// CUDA-owned half conversion is required: canonical Q8_1 stores both d/s as
// IEEE binary16, not the float-scale fallback ABI. No host libc headers.
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef short int16_t;
typedef int int32_t;
#include <stdint.h>
#include <cuda_fp16.h>
static_assert(sizeof(uint8_t)==1&&sizeof(int8_t)==1,"byte widths");
static_assert(sizeof(uint16_t)==2&&sizeof(int16_t)==2&&sizeof(int32_t)==4,"integer widths");
struct v41_q81 { __half2 ds; signed char qs[32]; };
static_assert(sizeof(v41_q81)==36&&alignof(v41_q81)==4,"canonical Q8_1 ABI");

// Pinned ds4_repack.cu442-459: scale stream, pad to64, then32 codes/block.
extern "C" __global__ void ds4_v41_aligned_repack(unsigned char *aligned,const unsigned char *raw,
    uint64_t blocks,uint64_t scale_bytes){
    uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=blocks*34)return;
    uint64_t block=i/34;unsigned j=i%34;
    if(j<2)aligned[block*2+j]=raw[block*34+j];
    else aligned[scale_bytes+block*32+j-2]=raw[block*34+j];
}
extern "C" __global__ void ds4_v41_aligned_clear(unsigned char *p,uint64_t n){uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)p[i]=0;}

// Literal arithmetic of quantize.cu5-49, single contiguous column, no padding
// because eligible K is divisible1024. XOR reductions are not down reductions.
extern "C" __global__ void ds4_v41_aligned_quantize(v41_q81 *y,const float *x,unsigned cols){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cols)return;
    unsigned lane=threadIdx.x&31,b=i/32;
    float value=x[i],peak=fabsf(value),sum=value;
    for(unsigned step=16;step;step>>=1){peak=fmaxf(peak,__shfl_xor_sync(0xffffffffu,peak,step));sum+=__shfl_xor_sync(0xffffffffu,sum,step);}
    float d=peak/127.f;
    signed char q=peak==0?0:static_cast<signed char>(roundf(value/d));
    y[b].qs[lane]=q;
    if(!lane)y[b].ds=__floats2half2_rn(d,sum);
}

// Same two aligned16-byte code loads, eight DP4As/block, per-lane block walk,
// and shuffle-down accumulation as ds4_mmq.cu3951-3986. WARPS varies by device.
extern "C" __global__ void ds4_v41_aligned_matvec(float *out,const unsigned char *aligned,
    const v41_q81 *x,unsigned rows,unsigned blocks,uint64_t scale_bytes){
    unsigned warp=threadIdx.x>>5,lane=threadIdx.x&31;
    unsigned row=blockIdx.x*(blockDim.x/32)+warp;if(row>=rows)return;
    const __half *dq=reinterpret_cast<const __half *>(aligned);
    const int4 *qs=reinterpret_cast<const int4 *>(aligned+scale_bytes);
    uint64_t base=uint64_t(row)*blocks;
    float acc=0;
    for(unsigned b0=0;b0<blocks;b0+=32){
        unsigned b=b0+lane;int4 a=qs[(base+b)*2],z=qs[(base+b)*2+1];
        const int *u=reinterpret_cast<const int *>(x[b].qs);
        int dot=0;
        dot=__dp4a(a.x,u[0],dot);dot=__dp4a(a.y,u[1],dot);dot=__dp4a(a.z,u[2],dot);dot=__dp4a(a.w,u[3],dot);
        dot=__dp4a(z.x,u[4],dot);dot=__dp4a(z.y,u[5],dot);dot=__dp4a(z.z,u[6],dot);dot=__dp4a(z.w,u[7],dot);
        acc+=__half2float(dq[base+b])*__low2float(x[b].ds)*float(dot);
    }
    for(unsigned step=16;step;step>>=1)acc+=__shfl_down_sync(0xffffffffu,acc,step);
    if(!lane)out[row]=acc;
}
// Diagnostic device oracle boundary; not called by production Eval.
extern "C" __global__ void ds4_v41_aligned_dots(int *out,const unsigned char *aligned,
    const v41_q81 *x,unsigned rows,unsigned blocks,uint64_t scale_bytes){
    uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=uint64_t(rows)*blocks)return;
    unsigned b=i%blocks;const signed char *w=reinterpret_cast<const signed char *>(aligned+scale_bytes+i*32);
    int dot=0;for(unsigned j=0;j<32;j++)dot+=int(w[j])*int(x[b].qs[j]);out[i]=dot;
}
