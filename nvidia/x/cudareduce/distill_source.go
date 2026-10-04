package cudareduce

import "github.com/tmc/lib/nvidia/x/cudakernel"

// SparseTeacherSource is the ordinary three-dispatch FP32 CUDA baseline.
// It uses cudablock.cuh's block reductions with exactly 32 threads per block.
const SparseTeacherSource = `#include "cudablock.cuh"
__device__ bool distill_finite(float x) { return (__float_as_uint(x)&0x7f800000u)!=0x7f800000u; }
extern "C" __global__ void sparse_teacher(
 const float* z, const int* ids, const float* weights, const int* masks,
 int* status, float* stats, float* losses, float* gradients, float* total,
 int rows, int vocab, int retained, int objective, float temperature, int stage) {
 int row=blockIdx.x, lane=threadIdx.x;
 if(stage==2) {
   if(row || lane) return;
   double sum=0;
   for(int r=0;r<rows;r++) { status[r]=0; sum+=(double)losses[r]; }
   float value=(float)sum;
   if(!distill_finite(value)) { status[0]=2; return; }
   total[0]=value; return;
 }
 if(row>=rows) return;
 if(stage==0) {
   if(lane) return;
   int bad=(masks[row]!=0 && masks[row]!=1); double mass=0;
   for(int j=0;j<vocab;j++) {
     float x=z[row*vocab+j];
     if(!distill_finite(x) || !distill_finite(x/temperature)) bad=1;
   }
   for(int i=0;i<retained;i++) {
     int id=ids[row*retained+i]; float w=weights[row*retained+i];
     if(id<0 || id>=vocab || !distill_finite(w) || w<0) bad=1;
     for(int a=0;a<i;a++) if(ids[row*retained+a]==id) bad=1;
     mass+=(double)w;
   }
   if(objective==0 ? !(mass>0) : (mass>1 || (retained==vocab && mass!=1))) bad=1;
   status[row]=bad?1:0; return;
 }
 // All threads take the same masked-row branch before any collective.
 if(!masks[row]) {
   for(int j=lane;j<vocab;j+=32) gradients[row*vocab+j]=0;
   if(!lane) { losses[row]=0; status[row]=0; for(int i=0;i<4;i++) stats[row*4+i]=0; }
   return;
 }
 __shared__ float scratch[1];
 __shared__ float maximum, omitted_maximum, denominator, omitted_denominator;
 float m=-__int_as_float(0x7f800000), om=m;
 for(int j=lane;j<vocab;j+=32) {
   float x=z[row*vocab+j]/temperature; m=fmaxf(m,x);
   bool found=false; for(int i=0;i<retained;i++) if(ids[row*retained+i]==j) found=true;
   if(!found) om=fmaxf(om,x);
 }
 float reduced=block_reduce_max<float,32>(m,scratch);
 if(!lane) maximum=reduced;
 __syncthreads();
 reduced=block_reduce_max<float,32>(om,scratch);
 if(!lane) omitted_maximum=reduced;
 __syncthreads();
 float s=0, os=0;
 for(int j=lane;j<vocab;j+=32) {
   float x=z[row*vocab+j]/temperature; s+=expf(x-maximum);
   bool found=false; for(int i=0;i<retained;i++) if(ids[row*retained+i]==j) found=true;
   if(!found) os+=expf(x-omitted_maximum);
 }
 reduced=block_reduce_sum<float,32>(s,scratch);
 if(!lane) denominator=reduced;
 __syncthreads();
 reduced=block_reduce_sum<float,32>(os,scratch);
 if(!lane) omitted_denominator=reduced;
 __syncthreads();
 double mass=0; for(int i=0;i<retained;i++) mass+=(double)weights[row*retained+i];
 double other=objective==1 ? 1-mass : 0;
 float error=0;
 for(int j=lane;j<vocab;j+=32) {
   float q=expf(z[row*vocab+j]/temperature-maximum)/denominator;
   double p=0; bool found=false;
   for(int i=0;i<retained;i++) if(ids[row*retained+i]==j) { found=true; p=weights[row*retained+i]; }
   if(objective==0) p/=mass;
   float grad=(float)((q-p)/temperature);
   if(objective==1 && !found) grad=(float)(q-other*(expf(z[row*vocab+j]-omitted_maximum)/omitted_denominator));
   gradients[row*vocab+j]=grad;
   if(!distill_finite(grad)) error=1;
 }
 reduced=block_reduce_sum<float,32>(error,scratch);
 if(!lane) {
   double loss=0;
   for(int i=0;i<retained;i++) {
     double p=weights[row*retained+i]; if(objective==0) p/=mass;
     if(p>0) loss+=p*(log(p)-((double)z[row*vocab+ids[row*retained+i]]/temperature-maximum-log((double)denominator)));
   }
   // Empty omitted sets and zero teacher terms never evaluate log(0).
   if(other>0) loss+=other*(log(other)-((double)omitted_maximum-maximum+log((double)omitted_denominator)-log((double)denominator)));
   losses[row]=(float)loss;
   stats[row*4]=maximum; stats[row*4+1]=denominator;
   stats[row*4+2]=retained==vocab?0:omitted_maximum;
   stats[row*4+3]=omitted_denominator;
   status[row]=(reduced!=0 || !distill_finite((float)loss))?2:0;
 }
}
`

// SparseTeacherSchema describes the pointer-array launch ABI.
func SparseTeacherSchema() (cudakernel.Schema, error) {
	var p []cudakernel.Param
	for _, n := range []string{"z", "ids", "weights", "masks", "status", "stats", "losses", "gradients", "total"} {
		p = append(p, cudakernel.Param{Name: n, Type: "ptr", IsPointer: true})
	}
	for _, n := range []string{"rows", "vocab", "retained", "objective"} {
		p = append(p, cudakernel.Param{Name: n, Type: "s32"})
	}
	p = append(p, cudakernel.Param{Name: "temperature", Type: "f32"}, cudakernel.Param{Name: "stage", Type: "s32"})
	return cudakernel.NewSchema(p...)
}
