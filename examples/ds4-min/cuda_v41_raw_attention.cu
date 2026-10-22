#include <stdint.h>

extern "C" __global__ void ds4_v41_raw_copy(float *dst,const float *src,unsigned n,unsigned clear) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)dst[i]=clear?0:src[i];
}
extern "C" __global__ void ds4_v41_raw_commit(float *ring,const float *kv,unsigned position) {
    unsigned d=blockIdx.x*blockDim.x+threadIdx.x;
    if(d<512)ring[(position%128)*512+d]=kv[d];
}
// Deliberately scalar-order device reference evaluator. This is not the
// optimized C batched-attention reduction/half-probability route.
extern "C" __global__ void ds4_v41_raw_attend(float *out,const float *q,
    const float *ring,const float *sinks,float *workspace,unsigned position) {
    unsigned head=blockIdx.x*blockDim.x+threadIdx.x;
    if(head>=64)return;
    unsigned n=position+1;if(n>128)n=128;
    unsigned first=position+1-n;
    float *scores=workspace+head*128;
    // Each request owns all 64 rows. Define unused cells too, so an old
    // request's scratch contents cannot influence this invocation.
    for(unsigned row=0;row<128;row++)scores[row]=0;
    float peak=sinks[head];
    const float scale=__fdiv_rn(1.f,float(sqrt(512.0)));
    for(unsigned row=0;row<n;row++) {
        unsigned slot=(first+row)%128;
        float dot=0;
        for(unsigned d=0;d<512;d++)dot=__fadd_rn(dot,__fmul_rn(q[head*512+d],ring[slot*512+d]));
        scores[row]=__fmul_rn(dot,scale);
        peak=fmaxf(peak,scores[row]);
    }
    float denominator=float(exp(double(__fsub_rn(sinks[head],peak))));
    for(unsigned row=0;row<n;row++) {
        scores[row]=float(exp(double(__fsub_rn(scores[row],peak))));
        denominator=__fadd_rn(denominator,scores[row]);
    }
    float inv=__fdiv_rn(1.f,denominator);
    for(unsigned d=0;d<512;d++) {
        float sum=0;
        for(unsigned row=0;row<n;row++)sum=__fadd_rn(sum,__fmul_rn(ring[((first+row)%128)*512+d],scores[row]));
        out[head*512+d]=__fmul_rn(sum,inv);
    }
}
