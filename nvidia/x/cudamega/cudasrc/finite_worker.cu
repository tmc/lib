extern "C" __global__ void finite_worker(const unsigned* in, unsigned* out,
    volatile unsigned* abort, unsigned n, unsigned long long epoch) {
  if (!epoch) {
    if (threadIdx.x == 0) atomicCAS((unsigned *)abort, 0u, %[1]du);
    return;
  }
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n && *abort==0) { out[i]=in[i]+1; __threadfence(); }
  __syncthreads();
}
