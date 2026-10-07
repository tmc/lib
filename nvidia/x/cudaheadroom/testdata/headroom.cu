extern "C" __global__ void headroom(unsigned int *out, unsigned int bytes) {
    __shared__ volatile unsigned char fixed[256];
    extern __shared__ volatile unsigned char dynamic[];
    fixed[0] = 17;
    fixed[255] = 19;
    dynamic[0] = 23;
    dynamic[bytes - 1] = 29;
    out[0] = fixed[0];
    out[1] = fixed[255];
    out[2] = dynamic[0];
    out[3] = dynamic[bytes - 1];
}
