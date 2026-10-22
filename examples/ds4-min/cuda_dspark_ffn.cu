#include <cuda_fp16.h>

struct dspark_q8_mmq { float d[4]; signed char qs[128]; };
static_assert(sizeof(dspark_q8_mmq) == 144, "MMQ activation ABI");
struct dspark_q8_d2s6 { __half ds[8]; signed char qs[128]; };
static_assert(sizeof(dspark_q8_d2s6) == 144, "D2S6 activation ABI");

extern "C" __global__ void ds4_dspark_ffn_quant_d2s6(
    const float *in, dspark_q8_d2s6 *out, unsigned cols, unsigned tokens) {
  unsigned lane = threadIdx.x, chunk = blockIdx.x / tokens, token = blockIdx.x % tokens;
  unsigned at = chunk * 128 + lane * 4;
  float4 v = reinterpret_cast<const float4 *>(in + token * cols)[at / 4];
  float maximum = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
  for (unsigned shift = 8; shift; shift >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, shift));
  float sum = v.x + v.y + v.z + v.w;
  for (unsigned shift = 2; shift; shift >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, shift);
  float inverse = maximum == 0.0f ? 0.0f : 127.0f / maximum;
  dspark_q8_d2s6 &q = out[blockIdx.x];
  q.qs[lane*4] = roundf(v.x * inverse);
  q.qs[lane*4+1] = roundf(v.y * inverse);
  q.qs[lane*4+2] = roundf(v.z * inverse);
  q.qs[lane*4+3] = roundf(v.w * inverse);
  if (lane % 4 == 0 && lane < 24) q.ds[2+lane/4] = __float2half_rn(sum);
  if (lane % 16 == 0) q.ds[lane/16] = __float2half_rn(maximum == 0.0f ? 0.0f : 1.0f / inverse);
}

// D4 uses the source's float scale, and IQ2's scale multiplication precedes
// the dot. Unlike the selected Q8_1 alternate there is no integer /8 truncation.
extern "C" __global__ void ds4_dspark_ffn_iq2_mmq(
    const unsigned char *gate, const unsigned char *up, const int *ids,
    const dspark_q8_mmq *in, float *gate_out, float *up_out,
    unsigned cols, unsigned rows, unsigned tokens, unsigned slots, unsigned experts) {
  unsigned row = blockIdx.x % rows, pair = blockIdx.x / rows, token = pair / slots;
  unsigned lane = threadIdx.x;
  int expert = ids[pair];
  if (token >= tokens || expert < 0 || unsigned(expert) >= experts) return;
  unsigned long long offset = ((unsigned long long)expert * rows + row) * (cols/256) * 66;
  float g = 0.0f, u = 0.0f;
  for (unsigned part = lane; part < cols/32; part += 32) {
    const dspark_q8_mmq &a = in[(part/4)*tokens+token];
    const unsigned char *p[2] = {gate + offset + (part/8)*66, up + offset + (part/8)*66};
    float values[2];
    for (unsigned which = 0; which < 2; which++) {
      const unsigned char *q = p[which] + 2 + (part%8)*8;
      unsigned aux = unsigned(q[4]) | unsigned(q[5])<<8 | unsigned(q[6])<<16 | unsigned(q[7])<<24;
      int dot = 0;
      for (unsigned group = 0; group < 4; group++) {
        unsigned long long grid = ds4_cuda_iq2xxs_grid[q[group]];
        unsigned signs = ds4_cuda_ksigns_iq2xs[(aux>>(group*7))&127];
        for (unsigned i = 0; i < 8; i++) {
          int weight = int((grid>>(8*i))&255);
          if (signs & (1u<<i)) weight = -weight;
          dot += weight * int(a.qs[(part%4)*32+group*8+i]);
        }
      }
      float d = __half2float(__ushort_as_half(unsigned(p[which][0]) | unsigned(p[which][1])<<8));
      float weight_scale = d * float((aux>>27)|1) / 8.0f;
      values[which] = float(dot) * weight_scale * a.d[part%4];
    }
    g += values[0]; u += values[1];
  }
  for (unsigned shift = 16; shift; shift >>= 1) { g += __shfl_down_sync(0xffffffff,g,shift); u += __shfl_down_sync(0xffffffff,u,shift); }
  if (lane == 0) { gate_out[blockIdx.x] = g; up_out[blockIdx.x] = u; }
}

extern "C" __global__ void ds4_dspark_ffn_weighted_mid(
    const float *gate, const float *up, const float *weights, float *mid, unsigned width, unsigned pairs) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width*pairs) return;
  float g = isfinite(gate[i]) ? gate[i] : 0.0f;
  float u = isfinite(up[i]) ? up[i] : 0.0f;
  g = fminf(g,10.0f); u = fminf(fmaxf(u,-10.0f),10.0f);
  mid[i] = (g / (1.0f + expf(-g))) * u * weights[i/width];
}

extern "C" __global__ void ds4_dspark_ffn_q2_mmq(
    const unsigned char *weights, const int *ids, const dspark_q8_d2s6 *in,
    float *out, unsigned cols, unsigned rows, unsigned pairs, unsigned experts) {
  unsigned row = blockIdx.x % rows, pair = blockIdx.x / rows, lane = threadIdx.x;
  int expert = ids[pair];
  if (pair >= pairs || expert < 0 || unsigned(expert) >= experts) return;
  unsigned long long offset = ((unsigned long long)expert*rows+row)*(cols/256)*84;
  float value = 0.0f;
  for (unsigned group = lane; group < cols/16; group += 32) {
    unsigned block = group/16, sub = group%16;
    const unsigned char *w = weights+offset+block*84;
    const dspark_q8_d2s6 &a = in[(group/8)*pairs+pair];
    unsigned qbase = 32*(sub/8)+16*(sub&1), shift = ((sub/2)&3)*2;
    int dot = 0, codesum = 0;
    for (unsigned i = 0; i < 16; i++) {
      int code = int(a.qs[(group%8)*16+i]);
      dot += int((w[16+qbase+i]>>shift)&3)*code; codesum += code;
    }
    float d = __half2float(__ushort_as_half(unsigned(w[80])|unsigned(w[81])<<8));
    float m = __half2float(__ushort_as_half(unsigned(w[82])|unsigned(w[83])<<8));
    float wd = __half2float(__float2half_rn(d*float(w[sub]&15)));
    float wm = __half2float(__float2half_rn(m*float(w[sub]>>4)));
    float activation_scale = __half2float(a.ds[(group%8)/4]);
    float minimum_sum = group%8 < 6 ? __half2float(a.ds[2+group%8]) : activation_scale*float(codesum);
    value += wd*float(dot)*activation_scale - wm*minimum_sum;
  }
  for (unsigned shift = 16; shift; shift >>= 1) value += __shfl_down_sync(0xffffffff,value,shift);
  if (lane == 0) out[blockIdx.x] = value;
}

extern "C" __global__ void ds4_dspark_ffn_copy(
    const float *in, float *out, unsigned count) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = in[i];
}

// Raw GGUF Q8_0 weights, MMQ block-major activations. Integer dots cover one
// 32-value scale block, retaining the float32 activation scale (not Q8_1 half).
extern "C" __global__ void ds4_dspark_ffn_q8_mmq(
    const unsigned char *weights, const dspark_q8_mmq *x, float *out,
    unsigned cols, unsigned rows, unsigned tokens) {
  unsigned row = blockIdx.x % rows, token = blockIdx.x / rows;
  if (token >= tokens) return;
  float value = 0.0f;
  for (unsigned b = threadIdx.x; b < cols / 32; b += 32) {
    const unsigned char *w = weights + ((unsigned long long)row * (cols / 32) + b) * 34;
    const dspark_q8_mmq &a = x[(b / 4) * tokens + token];
    int dot = 0;
    for (unsigned i = 0; i < 32; i++) dot += int((signed char)w[2+i]) * int(a.qs[(b % 4)*32+i]);
    unsigned short bits = unsigned(w[0]) | (unsigned(w[1]) << 8);
    value += __half2float(__ushort_as_half(bits)) * a.d[b % 4] * float(dot);
  }
  for (unsigned shift = 16; shift; shift >>= 1) value += __shfl_down_sync(0xffffffff, value, shift);
  if (threadIdx.x == 0) out[token * rows + row] = value;
}

// One output coordinate per thread. Expert slot order is significant: it
// matches the support reference rather than using atomic expert accumulation.
extern "C" __global__ void ds4_dspark_ffn_sum(
    const float *routed, const float *shared, float *out,
    unsigned width, unsigned slots) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width) return;
  float v = 0.0f;
  for (unsigned slot = 0; slot < slots; slot++) v += routed[slot * width + i];
  out[i] = v + shared[i];
}
