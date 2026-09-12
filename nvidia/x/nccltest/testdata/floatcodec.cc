#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// CUDA's host conversion routines require no CUDA device or runtime calls.
int main() {
  static_assert(sizeof(__half) == 2 && sizeof(__nv_bfloat16) == 2);
  static_assert(sizeof(__nv_fp8_e4m3) == 1 && sizeof(__nv_fp8_e5m2) == 1);
  unsigned long long bits;
  for (;;) {
    int result = std::scanf("%llx", &bits);
    if (result == EOF) return std::ferror(stdin) ? 1 : 0;
    if (result != 1) return 2;
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    __half h = __double2half(value);
    __nv_bfloat16 b = __double2bfloat16(value);
    __nv_fp8_e4m3 e4(value);
    __nv_fp8_e5m2 e5(value);
    unsigned short hb, bb;
    unsigned char e4b, e5b;
    std::memcpy(&hb, &h, 2); std::memcpy(&bb, &b, 2);
    std::memcpy(&e4b, &e4, 1); std::memcpy(&e5b, &e5, 1);
    double decoded[] = {double(__half2float(h)), double(__bfloat162float(b)),
                        double(static_cast<float>(e4)), double(static_cast<float>(e5))};
    unsigned long long words[4];
    std::memcpy(words, decoded, sizeof(words));
    if (std::printf("%04x %04x %02x %02x %016llx %016llx %016llx %016llx\n",
                    hb, bb, e4b, e5b, words[0], words[1], words[2], words[3]) < 0) return 3;
  }
}
