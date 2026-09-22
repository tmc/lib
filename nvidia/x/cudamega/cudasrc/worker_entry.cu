
extern "C" __global__ void dag_worker(const unsigned *input, unsigned *scratch,
    unsigned *completed, volatile unsigned *abort_code, unsigned n, unsigned long long epoch) {
  if (gridDim.x != grid_ctas || gridDim.y != 1 || gridDim.z != 1 || blockDim.y != 1 || blockDim.z != 1) {
    if (threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, %[1]du);
    return;
  }
  if (!n || !epoch) {
    if (threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, %[2]du);
    return;
  }
