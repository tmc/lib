#include <cuda_runtime.h>
constexpr unsigned task_count = 2;
constexpr unsigned grid_ctas = 1;
constexpr unsigned long long spin_limit = 16777216ull;
static_assert(task_count >= 1, "the generated worker has at least one task");
__device__ unsigned op_0(unsigned x) { return x + 3u; }
__device__ unsigned op_1(unsigned x) { return x * 2u; }
__device__ int dependency(unsigned task) { switch(task) {
case 0: return -1;
case 1: return 0;
default: return -2; }}
__device__ unsigned dispatch(unsigned task, unsigned x) { switch(task) {
case 0: return op_0(x);
case 1: return op_1(x);
default: return 0; }}

extern "C" __global__ void dag_worker(const unsigned *input, unsigned *scratch,
    unsigned *completed, volatile unsigned *abort_code, unsigned n, unsigned long long epoch) {
  if (gridDim.x != grid_ctas || gridDim.y != 1 || gridDim.z != 1 || blockDim.y != 1 || blockDim.z != 1) {
    if (threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, 2u);
    return;
  }
  if (!n || !epoch) {
    if (threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, 3u);
    return;
  }
  __shared__ __align__(8) unsigned long long cta_barrier;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
        "r"((unsigned)__cvta_generic_to_shared(&cta_barrier)), "r"(blockDim.x) : "memory");
  }
  __syncthreads();
  const unsigned cta_barrier_addr = (unsigned)__cvta_generic_to_shared(&cta_barrier);
  unsigned cta_phase = 0u;

  // task 0 "add3" scope=cta rank=0 deps=[]
  if (!*abort_code) {
    if (!*abort_code) {
      const unsigned *src = input;
      for (size_t e = (size_t)blockIdx.x*blockDim.x + threadIdx.x; e < n; e += (size_t)grid_ctas*blockDim.x)
        scratch[(size_t)0*n + e] = op_0(src[e]);
    }
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(cta_barrier_addr) : "memory"); // release for task 0
    if (threadIdx.x == 0) atomicAdd(completed + 0, 1u); // publish task 0
  }

  // task 1 "multiply2" scope=cta rank=1 deps=[0]
  if (!*abort_code) {
    if (threadIdx.x == 0) {
      unsigned long long spins = 0ull;
      while (((volatile unsigned*)completed)[0] < grid_ctas) {
        if (*abort_code) break;
        if (++spins > spin_limit) { atomicCAS((unsigned *)abort_code, 0u, 4u); break; }
      }
    }
    __syncthreads();
    { asm volatile("{ .reg .pred p; L_wait_%=: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1; @!p bra L_wait_%=; }" :: "r"(cta_barrier_addr), "r"(cta_phase) : "memory"); cta_phase ^= 1u; } // acquire for task 1
    if (!*abort_code) {
      const unsigned *src = scratch + (size_t)0*n;
      for (size_t e = (size_t)blockIdx.x*blockDim.x + threadIdx.x; e < n; e += (size_t)grid_ctas*blockDim.x)
        scratch[(size_t)1*n + e] = op_1(src[e]);
    }
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(cta_barrier_addr) : "memory"); // release for task 1
    if (threadIdx.x == 0) atomicAdd(completed + 1, 1u); // publish task 1
  }

}

// ordinary_task is the reference implementation: the same tasks, one ordinary
// launch each, in rank order, with the host's stream ordering standing in for
// every fence above. It exists so the persistent result has something to be
// equal to.
extern "C" __global__ void ordinary_task(const unsigned *input, unsigned *scratch,
    unsigned n, unsigned task) {
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  int dep = dependency(task);
  if (i < n) {
    const unsigned *src = dep < 0 ? input : scratch + (size_t)dep*n;
    scratch[(size_t)task*n+i] = dispatch(task, src[i]);
  }
}
