// The file of the unit that defines the kernel.
//
// The padding below puts the definition at a line number main.cu does not
// have, so a reader that reports the unit's file with the kernel's line names
// a location that does not exist. That is the defect this unit was built for.
//
// pad
// pad
// pad
// pad
__global__ void accumulate(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] += in[i];
    }
}
