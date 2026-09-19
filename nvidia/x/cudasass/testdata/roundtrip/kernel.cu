extern "C" __global__ void addk(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
extern "C" __global__ void intmix(int* o, int n) {
    int t = threadIdx.x;
    int u = t + n;
    int v = t * 3;
    int w = u ^ v;
    o[t] = w + (u & v) - (t | n);
}
