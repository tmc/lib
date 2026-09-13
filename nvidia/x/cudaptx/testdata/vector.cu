// Vector loads produce brace-delimited operand lists
// (ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [...]), which must not be mistaken
// for block delimiters.
extern "C" __global__ void vec4(const float4 *in, float4 *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float4 v = in[i];
    out[i] = make_float4(v.x + 1.f, v.y + 2.f, v.z + 3.f, v.w + 4.f);
  }
}
