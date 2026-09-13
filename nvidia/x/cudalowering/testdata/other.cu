// A module unrelated to lowering.cu. Comparing the PTX of one against the SASS
// of the other is how the tests prove that unmatched kernels are reported
// rather than silently skipped.

__global__ void unrelated(int *out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		out[i] = i;
	}
}
