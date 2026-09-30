// Each output value is independently specified by the controlled host source.
extern "C" __global__ void dependency_write(long long *out, int index, long long value) {
 out[index]=value;
}
extern "C" __global__ void dependency_double(long long *out) {
 out[1]=2*out[0];
}
extern "C" __global__ void dependency_combine(long long *out) {
 out[3]=out[0]+out[1]+out[2];
}
