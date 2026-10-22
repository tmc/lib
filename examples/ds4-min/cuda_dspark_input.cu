// Replicate the first lane of each row into the remaining lane-major HC.
// Lane0 is never written, so in-place expansion has no cross-block race.
extern "C" __global__ void ds4_dspark_repeat_hc(float *rows) {
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    if (d >= 4096) return;
    float *x = rows + (unsigned long long)row * 16384;
    float value = x[d];
    for (unsigned int lane = 1; lane < 4; lane++) x[lane * 4096 + d] = value;
}
