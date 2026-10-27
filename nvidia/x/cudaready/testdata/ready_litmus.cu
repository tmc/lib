// Message-passing litmus test for ready.cuh on a GPU. Block pairs share a
// payload: for each epoch the producer waits for the consumer's ack of the
// previous one, writes the payload with every thread, and signals; the
// consumer waits, reads every word with plain loads, counts the words
// that do not hold the epoch, and acks. A word read stale means the
// signal became visible before the writes it publishes.
//
// Built with -DNO_WAIT the consumer skips the wait and must see stale
// words, or the test could not see broken ordering either. Built with
// -DNO_FENCE the counters work without either fence; whether that shows
// stale words depends on the hardware, so ready_test.go only reports it.
//
// Exit status 77 means no device.
#include <cstdio>
#include <cuda_runtime.h>

#ifdef NO_FENCE
#define READY_HOST
namespace ready {
__device__ __forceinline__ unsigned long long load_relaxed(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fence_acquire() {}
__device__ __forceinline__ void fetch_max(unsigned long long* p, unsigned long long v) { atomicMax(p, v); }
__device__ __forceinline__ void fetch_add(unsigned long long* p, unsigned long long v) { atomicAdd(p, v); }
__device__ __forceinline__ void fence() {}
__device__ __forceinline__ void pause(unsigned ns) { __nanosleep(ns); }
}  // namespace ready
#endif
#include "../ready.cuh"

constexpr int WORDS = 4096, EPOCHS = 2000, THREADS = 256;

__global__ void litmus(unsigned long long* c, unsigned* payload, unsigned long long* stale) {
  const int pair = blockIdx.x / 2, data = 2 * pair, ack = 2 * pair + 1;
  const bool producer = blockIdx.x % 2 == 0;
  unsigned* p = payload + pair * WORDS;
  unsigned long long bad = 0;
  for (unsigned e = 1; e <= EPOCHS; e++) {
    if (producer) {
      if (threadIdx.x == 0) ready::wait(c, ack, e > 1, e - 1);
      __syncthreads();
      for (int i = threadIdx.x; i < WORDS; i += blockDim.x) p[i] = e;
      __syncthreads();
      if (threadIdx.x == 0) ready::signal(c, data, e, 1);
    } else {
#ifndef NO_WAIT
      if (threadIdx.x == 0) ready::wait(c, data, 1, e);
#endif
      __syncthreads();
      for (int i = threadIdx.x; i < WORDS; i += blockDim.x) bad += p[i] != e;
      __syncthreads();
      if (threadIdx.x == 0) ready::signal(c, ack, e, 1);
    }
  }
  if (bad) atomicAdd(stale, bad);
}

#define check(call)                                                             \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err));              \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

int main() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    fprintf(stderr, "no CUDA device\n");
    return 77;
  }
  cudaDeviceProp prop;
  check(cudaGetDeviceProperties(&prop, 0));
  // One block per SM, launched cooperatively, so every pair is resident.
  const int pairs = prop.multiProcessorCount / 2 < 32 ? prop.multiProcessorCount / 2 : 32;
  unsigned long long *c, *stale;
  unsigned* payload;
  check(cudaMalloc(&c, 2 * pairs * sizeof *c));
  check(cudaMalloc(&stale, sizeof *stale));
  check(cudaMalloc(&payload, pairs * WORDS * sizeof *payload));
  check(cudaMemset(c, 0, 2 * pairs * sizeof *c));
  check(cudaMemset(stale, 0, sizeof *stale));
  check(cudaMemset(payload, 0, pairs * WORDS * sizeof *payload));
  void* args[] = {&c, &payload, &stale};
  check(cudaLaunchCooperativeKernel((void*)litmus, 2 * pairs, THREADS, args, 0, 0));
  check(cudaDeviceSynchronize());
  unsigned long long bad = 0;
  check(cudaMemcpy(&bad, stale, sizeof bad, cudaMemcpyDeviceToHost));
  printf("%s: %d pairs, %lld words checked, %llu stale\n", prop.name, pairs,
         (long long)pairs * EPOCHS * WORDS, bad);
  return bad != 0;
}
