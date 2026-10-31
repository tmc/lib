extern "C" __global__ void aot_fill(int *out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = i * 3 + 1;
}
