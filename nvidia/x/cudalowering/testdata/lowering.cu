// Fixture kernels for the cudalowering PTX-versus-SASS comparison tests.
//
// Each kernel exists to make one lowering decision visible in the SASS that
// the PTX alone does not predict. Compiled by gen.sh; the generated .ptx and
// .sass files are byte-exact toolchain output and must not be hand-edited.

// fmaChain asks for double-precision fused multiply-add. The PTX says
// fma.rn.f64; whether the SASS says DFMA is up to ptxas.
__global__ void fmaChain(double *out, const double *a, const double *b, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	double acc = a[i];
	double m = b[i];
#pragma unroll
	for (int k = 0; k < 8; k++) {
		acc = acc * m + a[i];
	}
	out[i] = acc;
}

// spiller keeps two dozen values live at once. Every value lives in a virtual
// register in the PTX, so the PTX contains no local memory traffic at all; any
// LDL or STL in the SASS was introduced by ptxas.
__global__ void spiller(float *out, const float *in, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	float v[24];
#pragma unroll
	for (int k = 0; k < 24; k++) {
		v[k] = in[i + k] * (float)(k + 1);
	}
	float s = 0.0f;
#pragma unroll
	for (int k = 0; k < 24; k++) {
		s += v[k] * v[(k * 7 + 3) % 24];
	}
	out[i] = s;
}

// shuffleReduce asks for a warp shuffle, which lowers to SHFL and to nothing
// else: a SHFL with no shfl.sync above it would be the assembler inventing one.
__global__ void shuffleReduce(float *out, const float *in, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float v = i < n ? in[i] : 0.0f;
#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		v += __shfl_down_sync(0xffffffff, v, off);
	}
	if ((threadIdx.x & 31) == 0) {
		out[i >> 5] = v;
	}
}

// atomicKeep uses the value the atomic returns, so the fetch-and-add has to
// survive into the SASS as ATOMG.
__global__ void atomicKeep(float *out, const float *in, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		out[i] = atomicAdd(&out[0], in[i]);
	}
}

// atomicDrop throws the returned value away. The PTX still says
// atom.global.add.f32, exactly as for atomicKeep; ptxas is free to substitute
// the cheaper reduction, and does.
__global__ void atomicDrop(float *out, const float *in, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		atomicAdd(&out[0], in[i]);
	}
}
