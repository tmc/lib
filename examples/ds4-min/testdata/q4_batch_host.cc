// Host-only CUDA source control; no device execution or admission.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <functional>
#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__ static
#define QK_K 256
struct Dim { unsigned x, y; };
Dim blockIdx, blockDim{256, 1};
thread_local Dim threadIdx;
class Barrier {
  std::mutex mu;
  std::condition_variable cv;
  unsigned count = 0, phase = 0;
public:
  void wait() {
    std::unique_lock<std::mutex> lock(mu);
    unsigned p = phase;
    if (++count == 256) { count = 0; phase++; cv.notify_all(); }
    else cv.wait(lock, [&]{ return phase != p; });
  }
} barrier;
void __syncthreads() { barrier.wait(); }
float __int_as_float(unsigned x) { float v; memcpy(&v, &x, 4); return v; }
float __fdiv_rn(float a, float b) { return a / b; }
float __fmul_rn(float a, float b) { return a * b; }
float ds4_half_to_float(uint16_t h) { _Float16 v; memcpy(&v, &h, 2); return float(v); }
float dev_f16_to_f32(uint16_t h) { return ds4_half_to_float(h); }
int __dp4a(int a, int b, int sum) {
  for (unsigned i = 0; i < 4; i++) sum += int(int8_t(unsigned(a) >> (8*i))) * int(int8_t(unsigned(b) >> (8*i)));
  return sum;
}
void ds4_die(const char *s) { fprintf(stderr, "%s\n", s); exit(1); }
struct ds4_q8_k { float d; int8_t qs[256]; int16_t bsums[16]; };
using block_q8_K = ds4_q8_k;
using cuda_block_q8_K = ds4_q8_k;
struct cuda_block_q4_K { uint16_t d, dmin; uint8_t scales[12], qs[128]; };
static_assert(sizeof(ds4_q8_k) == 292 && sizeof(cuda_block_q4_K) == 144, "layout");
thread_local int host_total, host_minimum, oracle_total, oracle_minimum;
// ORACLE
// PRODUCTION
void launch(const std::function<void()> &fn) {
  std::vector<std::thread> workers;
  for (unsigned i = 0; i < 256; i++) workers.emplace_back([&,i]{ threadIdx = {i,0}; fn(); });
  for (auto &worker : workers) worker.join();
}
void fail(const char *s) { fprintf(stderr, "%s\n", s); exit(1); }
double max_error = 0, max_bound = 0;
double sum_abs(const cuda_block_q4_K &w, const ds4_q8_k &x) {
  double sum = 0;
  for (unsigned g = 0; g < 8; g++) {
    uint8_t sc, mn;
    dev_q4_K_get_scale_min(g, w.scales, &sc, &mn);
    for (unsigned i = 0; i < 32; i++) {
      unsigned q = (w.qs[(g/2)*32+i] >> ((g&1)*4)) & 15;
      double value = double(dev_f16_to_f32(w.d))*sc*q - double(dev_f16_to_f32(w.dmin))*mn;
      sum += std::fabs(value*double(x.d)*x.qs[g*32+i]);
    }
  }
  return sum;
}
void compare(float got, float want, double absolute_sum, const char *stage) {
  double error = std::fabs(double(got)-want), bound = 8*FLT_EPSILON*absolute_sum;
  max_error = std::fmax(max_error, error); max_bound = std::fmax(max_bound, bound);
  if (!std::isfinite(got) || error > bound) {
    fprintf(stderr, "got=%.9g want=%.9g abs=%.9g bound=%.9g\n", got, want, error, bound);
    fail(stage);
  }
}
int main(int argc, char **argv) {
  if (argc != 2) fail("missing columns");
  const unsigned cols = unsigned(atoi(argv[1])), blocks = cols/256;
  if (cols != 512 && cols != 2048 && cols != 4096) fail("unregistered columns");
  constexpr unsigned rows = 3, experts = 256, npairs = 12;
  const int ids[npairs] = {255,2,0,2,17,255,17,255,2,0,17,-1};
  std::vector<cuda_block_q4_K> gate(experts*rows*blocks), up(gate.size());
  for (unsigned b = 0; b < gate.size(); b++) {
    gate[b].d = 0x3800; gate[b].dmin = 0xb400;
    up[b].d = 0x3400; up[b].dmin = 0xb800;
    for (unsigned i = 0; i < 12; i++) { gate[b].scales[i] = uint8_t(b*19+i*37+43); up[b].scales[i] = uint8_t(b*31+i*17+91); }
    for (unsigned i = 0; i < 128; i++) { gate[b].qs[i] = uint8_t(b*7+i*13); up[b].qs[i] = uint8_t(b*11+i*29); }
  }
  std::vector<float> input(npairs*cols);
  for (unsigned i = 0; i < input.size(); i++) input[i] = float(int((i*13+i/cols*31)%251)-125)/73;
  // Unique maxima avoid substituting CPU sequential tie selection for the
  // device tree's signed-maximum tie policy.
  for (unsigned b = 0; b < input.size()/256; b++) input[b*256+7] = (b%2 ? -3.125f : 3.125f);
  for (unsigned i = 0; i < 256; i++) input[i] = 0;
  std::vector<ds4_q8_k> q(input.size()/256), reference(q.size());
  ds4_quantize_row_q8_K(input.data(), reference.data(), input.size());
  for (unsigned b = 0; b < q.size(); b++) { blockIdx = {b,0}; launch([&]{ ds4_quant_q8_k(input.data(), q.data()); }); }
  if (memcmp(q.data(), reference.data(), q.size()*292)) fail("quantizer bytes");
  unsigned partials = 0;
  for (unsigned b = 0; b < q.size(); b++) {
    float got = ds4_q4_block_q8_k(reinterpret_cast<const unsigned char *>(&gate[b]), &q[b]);
    float want = dev_dot_q4_K_q8_K_block(&gate[b], &q[b]);
    if (host_total != oracle_total || host_minimum != oracle_minimum) fail("integer partial");
    compare(got, want, sum_abs(gate[b], q[b]), "dot output"); partials++;
  }
  int offsets[experts+2] = {}, pairs[npairs];
  unsigned at = 0;
  for (unsigned e = 0; e <= experts; e++) {
    offsets[e] = at;
    for (unsigned p = 0; p < npairs; p++) if ((ids[p] < 0 ? experts : unsigned(ids[p])) == e) pairs[at++] = p;
  }
  offsets[experts+1] = at;
  std::vector<float> gout(npairs*rows, NAN), uout(gout), down(gout);
  for (unsigned e = 0; e <= experts; e++) if (offsets[e] != offsets[e+1]) {
    for (unsigned r = 0; r < rows; r++) {
      blockIdx = {r,e};
      launch([&]{ ds4_q4_k_pair_selected_q8_k_gather(reinterpret_cast<unsigned char *>(gate.data()), reinterpret_cast<unsigned char *>(up.data()), offsets, pairs, q.data(), gout.data(), uout.data(), cols, rows, experts, 6, npairs); });
      launch([&]{ ds4_q4_k_selected_q8_k_gather(reinterpret_cast<unsigned char *>(gate.data()), offsets, pairs, q.data(), down.data(), cols, rows, experts, npairs); });
    }
  }
  unsigned values = 0;
  for (unsigned p = 0; p < npairs; p++) for (unsigned r = 0; r < rows; r++) {
    unsigned dst = p*rows+r;
    if (ids[p] < 0) { if (std::isfinite(gout[dst]) || std::isfinite(uout[dst]) || std::isfinite(down[dst])) fail("invalid bin output"); continue; }
    float g = 0, u = 0, d = 0;
    double ga = 0, ua = 0, da = 0;
    for (unsigned b = 0; b < blocks; b++) {
      unsigned off = (ids[p]*rows+r)*blocks+b;
      g += dev_dot_q4_K_q8_K_block(&gate[off], &q[p/6*blocks+b]);
      u += dev_dot_q4_K_q8_K_block(&up[off], &q[p/6*blocks+b]);
      d += dev_dot_q4_K_q8_K_block(&gate[off], &q[p*blocks+b]);
      ga += sum_abs(gate[off], q[p/6*blocks+b]); ua += sum_abs(up[off], q[p/6*blocks+b]); da += sum_abs(gate[off], q[p*blocks+b]);
    }
    compare(gout[dst], g, ga, "gate/up output"); compare(uout[dst], u, ua, "gate/up output"); compare(down[dst], d, da, "down output"); values += 3;
  }
  printf("quantized_blocks=%zu exact_bytes=%zu exact_integer_partials=%u finite_outputs=%u invalid_outputs=9 max_abs=%.9g max_bound=%.9g\n", q.size(), q.size()*292, partials*2, values, max_error, max_bound);
}
