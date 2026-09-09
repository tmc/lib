extern "C" __global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void compute(float *x, int n, int rounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    for (int j = 0; j < rounds; j++) v = fmaf(v, 1.000001f, 0.0001f);
    x[i] = v;
}
