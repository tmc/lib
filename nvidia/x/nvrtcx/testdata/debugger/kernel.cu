extern "C" __global__ void debug_probe(int *out, int value) {
 int index = threadIdx.x;
 out[index] = value + index;
}
