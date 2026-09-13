// Smallest real-compiler fixture: one kernel, no debug info.
extern "C" __global__ void addone(float *p) {
  p[threadIdx.x] += 1.0f;
}
