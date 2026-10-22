#include <cuda_fp16.h>
#include <math.h>

// Serialized MXFP4: one E8M0 scale, then 16 bytes carrying low/high halves.
// Q8_1: half d, half sum (unused), then 32 signed codes. Each warp owns row/slot.
extern "C" __global__ void ds4_mxfp4_selected_q8_1(
 const unsigned char *weights, const int *ids, const unsigned char *input,
 float *output, unsigned cols, unsigned rows, unsigned experts,
 unsigned slots, unsigned per_slot) {
 unsigned row=blockIdx.x, slot=blockIdx.y, lane=threadIdx.x;
 if(row>=rows || slot>=slots) return;
 int expert=ids[slot];
 if(expert<0 || unsigned(expert)>=experts) {
  if(lane==0) output[slot*rows+row]=nanf("");
  return;
 }
 const float code[16]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
 unsigned blocks=cols/32;
 const unsigned char *w=weights+(static_cast<unsigned long long>(expert)*rows+row)*blocks*17;
 const unsigned char *x=input+(per_slot ? static_cast<unsigned long long>(slot)*blocks*36 : 0);
 float sum=0;
 for(unsigned b=0;b<blocks;b++) {
  const unsigned char *p=w+b*17;
  unsigned e=p[0];
  float scale=e==0 ? ldexpf(1.f,-127) : e==255 ? nanf("") : ldexpf(1.f,int(e)-127);
  unsigned nibble=(p[1+(lane&15)] >> (lane>=16 ? 4 : 0)) & 15;
  unsigned short dbits=unsigned(x[b*36]) | unsigned(x[b*36+1])<<8;
  float d=__half2float(__ushort_as_half(dbits));
  sum += scale*code[nibble]*d*float(static_cast<signed char>(x[b*36+4+lane]));
 }
 for(unsigned offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
 if(lane==0)output[slot*rows+row]=sum;
}
