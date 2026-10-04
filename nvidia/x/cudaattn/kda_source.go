package cudaattn

import "github.com/tmc/lib/nvidia/x/cudakernel"

// KDASource is the ordinary D=128 FP32 recurrence. One CUDA thread executes
// the bounded batch in order. State is [sequence,head,value,key]; token arrays
// are [token,head,channel]. Every multiply and add rounds separately to nearest
// FP32, with increasing-key reductions, no FMA and no flush-to-zero.
// Scratch receives all results; status is zero only when every result is finite.
// Public output/state publication is a separate operation after completion.
const KDASource = `
__device__ __forceinline__ bool kda_isfinite(float x) {
 return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u;
}
extern "C" __global__ void kda_fp32(
 const float* q, const float* key, const float* value, const float* decay,
 const float* beta, const float* initial, const int* offsets,
 float* state, float* output, int* status, int sequences, int heads, float scale) {
 if (blockIdx.x || threadIdx.x) return;
 int bad = 0;
 for (int s=0; s<sequences; ++s) for (int h=0; h<heads; ++h) {
  const int base = (s*heads+h)*128*128;
  for (int i=0; i<128*128; ++i) state[base+i]=initial[base+i];
  for (int t=offsets[s]; t<offsets[s+1]; ++t) {
   const int token=(t*heads+h)*128;
   for (int v=0; v<128; ++v) {
    const int row=base+v*128;
    float dot=0.0f;
    for (int k=0; k<128; ++k) {
     state[row+k]=__fmul_rn(state[row+k],decay[token+k]);
     dot=__fadd_rn(dot,__fmul_rn(state[row+k],key[token+k]));
    }
    const float correction=__fmul_rn(beta[t*heads+h],__fsub_rn(value[token+v],dot));
    float readout=0.0f;
    for (int k=0; k<128; ++k) {
     state[row+k]=__fadd_rn(state[row+k],__fmul_rn(correction,key[token+k]));
     bad |= !kda_isfinite(state[row+k]);
     readout=__fadd_rn(readout,__fmul_rn(state[row+k],q[token+k]));
    }
    output[token+v]=__fmul_rn(scale,readout);
    bad |= !kda_isfinite(output[token+v]);
   }
  }
 }
 status[0]=bad;
}
`

// KDASchema describes the exact kda_fp32 kernel parameter ABI.
func KDASchema() (cudakernel.Schema, error) {
	var params []cudakernel.Param
	for _, name := range []string{"q", "key", "value", "decay", "beta", "initial", "offsets", "state", "output", "status"} {
		params = append(params, cudakernel.Param{Name: name, Type: "ptr", IsPointer: true})
	}
	params = append(params, cudakernel.Param{Name: "sequences", Type: "s32"}, cudakernel.Param{Name: "heads", Type: "s32"}, cudakernel.Param{Name: "scale", Type: "f32"})
	return cudakernel.NewSchema(params...)
}
