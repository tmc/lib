package cudaquant

import (
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// RMSNormQ8_1UnfusedSource emits a two-kernel alternative to RMSNormQ8_1Source.
// rms_f32 writes rows*cols float32 values to a distinct intermediate buffer;
// quant_q8_1 reads that buffer and writes rows*(cols/32) Q8_1 blocks. Launch
// them in order on the same stream, with one one-dimensional block per row.
// Only thread zero computes. Inputs obey the fused kernel's numerical contract.
// The intermediate remains available to secondary consumers after both launches.
// Both entries use the fused seven-parameter ABI. The quantizer ignores weight,
// eps and weighted; its in argument is the normalization output.
// Header and builtin requirements match RMSNormQ8_1Source.
func RMSNormQ8_1UnfusedSource() string {
	return `#include <cuda_fp16.h>
struct Q8_1 { __half d, s; signed char q[32]; };
static_assert(sizeof(Q8_1) == 36, "q8_1 layout");
extern "C" __global__ void rms_f32(const float* in, const float* weight,
    float* out, unsigned rows, unsigned cols, float eps, unsigned weighted) {
  unsigned r = blockIdx.x;
  if (r >= rows || threadIdx.x != 0) return;
  double ss = 0;
  for (unsigned c=0; c<cols; ++c) { double v=in[(size_t)r*cols+c]; ss += v*v; }
  float inv = (float)(1.0 / sqrt(ss/cols + (double)eps));
  for (unsigned c=0; c<cols; ++c) {
    float v=in[(size_t)r*cols+c]*inv;
    if (weighted) v*=weight[c];
    out[(size_t)r*cols+c]=v;
  }
}
extern "C" __global__ void quant_q8_1(const float* in, const float* weight,
    Q8_1* out, unsigned rows, unsigned cols, float eps, unsigned weighted) {
  unsigned r = blockIdx.x;
  if (r >= rows || threadIdx.x != 0) return;
  for (unsigned b=0; b<cols/32; ++b) {
    float maximum=0;
    for (unsigned i=0; i<32; ++i)
      maximum=fmaxf(maximum,fabsf(in[(size_t)r*cols+b*32+i]));
    float d=maximum/127.0f;
    Q8_1* dst=out+(size_t)r*(cols/32)+b;
    dst->d=__float2half_rn(d);
    int sum=0;
    for (unsigned i=0; i<32; ++i) {
      float v=in[(size_t)r*cols+b*32+i];
      int q=d == 0 ? 0 : (int)roundf(v/d);
      q=q>127 ? 127 : (q < -127 ? -127 : q);
      dst->q[i]=(signed char)q; sum+=q;
    }
    dst->s=__float2half_rn((float)sum*d);
  }
}`
}

// RMSNormQ8_1UnfusedArtifacts compiles the normalization and quantization
// entries, in launch order, for a physical target. Compilation preserves each
// entry's resource requirements. This does not allocate, load or launch kernels.
func RMSNormQ8_1UnfusedArtifacts(target string) ([2]*cudakernel.Artifact, error) {
	a, err := normQuantArtifacts(target, RMSNormQ8_1UnfusedSource(), []string{"rms_f32", "quant_q8_1"},
		func(src string, opts ...nvrtcx.Option) (normQuantModule, error) { return nvrtcx.Compile(src, opts...) })
	if err != nil {
		return [2]*cudakernel.Artifact{}, err
	}
	return [2]*cudakernel.Artifact{a[0], a[1]}, nil
}
