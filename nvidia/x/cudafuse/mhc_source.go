package cudafuse

import "fmt"

// MHCSource returns the two ordinary CUDA kernels for c. Their ABI is the
// ordered input/output FP32 pointers followed by a signed 32-bit count. The
// normalization count is Batch; the post count is Batch*4*Width. Width, Repeat
// and the exact FP32 epsilon are specialized into source identity. Launches
// use one-dimensional blocks and grids. No architecture-specific features,
// shared headers, BF16 storage, or PDL are required.
func MHCSource(c MHCConfig) ([2]FusedKernel, error) {
	var out [2]FusedKernel
	if err := c.validate(); err != nil {
		return out, err
	}
	prefix := fmt.Sprintf("#define MHC_WIDTH %d\n#define MHC_REPEAT %d\n#define MHC_EPS %sf\n", c.Width, c.Repeat, fmt.Sprintf("%.9e", c.Epsilon))
	out[0] = FusedKernel{Name: "mhc_sinkhorn_fp32", Source: prefix + mhcNormalizeSource, Inputs: []string{"logits"}, Outputs: []string{"mix"}}
	out[1] = FusedKernel{Name: "mhc_post_fp32", Source: prefix + mhcPostSource, Inputs: []string{"x", "post", "residual", "mix"}, Outputs: []string{"out"}}
	return out, nil
}

// Scaling the positive denominator keeps finite positive epsilon, including
// very large epsilon, from overflowing a four-element sum in FP32.
const mhcNormalizeSource = `
__device__ void mhc_normalize(float *a, int axis) {
    for (int q=0; q<4; ++q) {
        float scale = MHC_EPS;
        for (int k=0; k<4; ++k) {
            int i = axis == 0 ? q*4+k : k*4+q;
            scale = fmaxf(scale,a[i]);
        }
        float sum = MHC_EPS / scale;
        for (int k=0; k<4; ++k) {
            int i = axis == 0 ? q*4+k : k*4+q;
            sum += a[i]/scale;
        }
        for (int k=0; k<4; ++k) {
            int i = axis == 0 ? q*4+k : k*4+q;
            a[i] = (a[i]/scale)/sum;
        }
    }
}
extern "C" __global__ void mhc_sinkhorn_fp32(const float *logits, float *mix, int count) {
    if (blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) return;
    unsigned int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (count <= 0 || b >= (unsigned int)count) return;
    float a[16];
    for (int r=0;r<4;++r) {
        float mx=logits[b*16+r*4];
        for (int k=1;k<4;++k) mx=fmaxf(mx,logits[b*16+r*4+k]);
        float sum=0.0f;
        for (int k=0;k<4;++k) { a[r*4+k]=expf(logits[b*16+r*4+k]-mx);sum+=a[r*4+k]; }
        for (int k=0;k<4;++k) a[r*4+k]=a[r*4+k]/sum+MHC_EPS;
    }
    mhc_normalize(a,1);
    for (int round=1;round<MHC_REPEAT;++round) {
        mhc_normalize(a,0);
        mhc_normalize(a,1);
    }
    for (int k=0;k<16;++k) mix[b*16+k]=a[k];
}
`

const mhcPostSource = `
extern "C" __global__ void mhc_post_fp32(const float *x, const float *post, const float *residual, const float *mix, float *out, int count) {
    if (blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) return;
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (count <= 0 || i >= (unsigned int)count) return;
    unsigned int c=i%MHC_WIDTH;
    unsigned int n=(i/MHC_WIDTH)%4;
    unsigned int b=i/(4*MHC_WIDTH);
    float v=x[b*MHC_WIDTH+c]*post[b*4+n];
    for (int m=0;m<4;++m) v+=mix[b*16+m*4+n]*residual[b*4*MHC_WIDTH+m*MHC_WIDTH+c];
    out[i]=v;
}
`
