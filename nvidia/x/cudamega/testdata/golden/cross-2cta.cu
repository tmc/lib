#include <cuda_runtime.h>
constexpr unsigned task_count = 4;
constexpr unsigned grid_ctas = 2;
constexpr unsigned grid_threads = 128;
constexpr unsigned long long spin_limit = 16777216ull;
constexpr unsigned producers_0 = 1u; // task "lower" is performed by CTAs [0,1)
constexpr unsigned producers_1 = 1u; // task "upper" is performed by CTAs [1,2)
static_assert(task_count >= 1, "the generated worker has at least one task");
__device__ unsigned op_0(unsigned x) { return x + 3u; }
__device__ unsigned op_1(unsigned x) { return x + 5u; }
__device__ unsigned op_2(unsigned x) { return x * 2u; }
__device__ unsigned op_3(unsigned x) { return x + 7u; }
__device__ int dependency(unsigned task) { switch(task) {
case 0: return -1;
case 1: return -1;
case 2: return 1;
case 3: return 0;
default: return -2; }}
__device__ unsigned dispatch(unsigned task, unsigned x) { switch(task) {
case 0: return op_0(x);
case 1: return op_1(x);
case 2: return op_2(x);
case 3: return op_3(x);
default: return 0; }}

// start_order begins with the instrument: start_order[0] is a
// ticket counter and start_order[1 + cta*task_count + task] is the ticket the
// CTA held when it took that task, or zero for a task that CTA never took.
// Tickets are dealt by one device-scope atomic add, so they are a total order
// over task starts across the whole grid and can be compared between CTAs.
// The final two words hold fresh zeroed entry arrivals and release state.
extern "C" __global__ void dag_worker(const unsigned *input, unsigned *scratch,
    unsigned *completed, unsigned long long *start_order, volatile unsigned *abort_code,
    unsigned n, unsigned long long epoch) {
  if (gridDim.x != grid_ctas || gridDim.y != 1 || gridDim.z != 1 || blockDim.x != grid_threads || blockDim.y != 1 || blockDim.z != 1) {
    if (threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, 2u);
    return;
  }
  if (!n || !epoch) {
    if (threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, 3u);
    return;
  }
  if (n > 1048576u) {
    if (threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, 3u);
    return;
  }

  // entry protocol begin
  __shared__ unsigned entry_abort;
  if (threadIdx.x==0) {
    unsigned long long *arrivals=start_order+1+(unsigned long long)task_count*grid_ctas;
    unsigned long long *released=arrivals+1;
    for (unsigned long long i=blockIdx.x; i<task_count; i+=grid_ctas)
      if (atomicAdd(completed+i,0u)!=0) atomicCAS((unsigned*)abort_code,0u,6u);
    asm volatile("fence.sc.gpu;" ::: "memory");
    unsigned long long arrival=atomicAdd(arrivals,1ull);
    if (arrival>=grid_ctas) atomicCAS((unsigned*)abort_code,0u,6u);
    if (arrival==grid_ctas-1) {
      asm volatile("fence.sc.gpu;" ::: "memory");
      atomicExch(released,1ull);
    }
    unsigned long long spins=0;
    while (atomicAdd(released,0ull)==0) {
      if (++spins>=16777216ull) { atomicCAS((unsigned*)abort_code,0u,4u); break; }
    }
    asm volatile("fence.sc.gpu;" ::: "memory");
    entry_abort=atomicAdd((unsigned*)abort_code,0u);
  }
  __syncthreads();
  if (entry_abort) return;
  // entry protocol end

  __shared__ unsigned selection_abort;
  __shared__ unsigned task_ready;
  const bool owns_0 = (blockIdx.x < 1u); // task "lower"
  const bool owns_1 = (blockIdx.x >= 1u && blockIdx.x < 2u); // task "upper"
  const bool owns_2 = (blockIdx.x < 1u); // task "cross"
  const bool owns_3 = (blockIdx.x >= 1u && blockIdx.x < 2u); // task "swap"
  unsigned done_0 = 0u;
  unsigned done_1 = 0u;
  unsigned done_2 = 0u;
  unsigned done_3 = 0u;
  unsigned remaining = 0u + (unsigned)owns_0 + (unsigned)owns_1 + (unsigned)owns_2 + (unsigned)owns_3;
  unsigned long long spins = 0ull;
  while (remaining) {
    if (threadIdx.x == 0) selection_abort = *abort_code;
    __syncthreads();
    const unsigned stop = selection_abort;
    __syncthreads();
    if (stop) break;

    // task 0 "lower" scope=device rank=0 deps=[] owner=[0,1)
    if (owns_0 && !done_0) {
      { // no dependency: ready on sight
        if (threadIdx.x == 0) start_order[1 + (size_t)blockIdx.x*task_count + 0] = atomicAdd(start_order, 1ull) + 1ull;
        const unsigned *src_0 = input;
        for (size_t e = (size_t)blockIdx.x*blockDim.x + threadIdx.x; e < n; e += (size_t)1u*blockDim.x)
          scratch[(size_t)0*n + e] = op_0(src_0[e]);
        asm volatile("fence.sc.sys;" ::: "memory"); // task 0: release at system scope
      __syncthreads();
        if (threadIdx.x == 0) atomicAdd(completed + 0, 1u); // publish task 0
        done_0 = 1u; remaining--;
      }
    }

    // task 1 "upper" scope=device rank=0 deps=[] owner=[1,2)
    if (owns_1 && !done_1) {
      { // no dependency: ready on sight
        if (threadIdx.x == 0) start_order[1 + (size_t)blockIdx.x*task_count + 1] = atomicAdd(start_order, 1ull) + 1ull;
        const unsigned *src_1 = input;
        for (size_t e = (size_t)(blockIdx.x - 1u)*blockDim.x + threadIdx.x; e < n; e += (size_t)1u*blockDim.x)
          scratch[(size_t)1*n + e] = op_1(src_1[e]);
        asm volatile("fence.sc.gpu;" ::: "memory"); // task 1: release at device scope
      __syncthreads();
        if (threadIdx.x == 0) atomicAdd(completed + 1, 1u); // publish task 1
        done_1 = 1u; remaining--;
      }
    }

    // task 2 "cross" scope=device rank=1 deps=[1] owner=[0,1)
    if (owns_2 && !done_2) {
      if (threadIdx.x == 0) {
        unsigned published = 0u;
        task_ready = 1u;
        asm volatile("ld.acquire.gpu.u32 %0, [%1+4];" : "=r"(published) : "l"(completed) : "memory"); // task 1
        if (published < producers_1) task_ready = 0u;
        if (published > producers_1) { task_ready = 0u; atomicCAS((unsigned *)abort_code, 0u, 5u); }
      }
      __syncthreads();
      const unsigned take_2 = task_ready;
      __syncthreads();
      if (take_2) {
        if (threadIdx.x == 0) start_order[1 + (size_t)blockIdx.x*task_count + 2] = atomicAdd(start_order, 1ull) + 1ull;
        asm volatile("fence.sc.gpu;" ::: "memory"); // task 2: acquire of task 1 at device scope
        const unsigned *src_2 = scratch + (size_t)1*n;
        for (size_t e = (size_t)blockIdx.x*blockDim.x + threadIdx.x; e < n; e += (size_t)1u*blockDim.x)
          scratch[(size_t)2*n + e] = op_2(src_2[e]);
        asm volatile("fence.sc.gpu;" ::: "memory"); // task 2: release at device scope
      __syncthreads();
        if (threadIdx.x == 0) atomicAdd(completed + 2, 1u); // publish task 2
        done_2 = 1u; remaining--;
      }
    }

    // task 3 "swap" scope=system rank=1 deps=[0] owner=[1,2)
    if (owns_3 && !done_3) {
      if (threadIdx.x == 0) {
        unsigned published = 0u;
        task_ready = 1u;
        asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(published) : "l"(completed) : "memory"); // task 0
        if (published < producers_0) task_ready = 0u;
        if (published > producers_0) { task_ready = 0u; atomicCAS((unsigned *)abort_code, 0u, 5u); }
      }
      __syncthreads();
      const unsigned take_3 = task_ready;
      __syncthreads();
      if (take_3) {
        if (threadIdx.x == 0) start_order[1 + (size_t)blockIdx.x*task_count + 3] = atomicAdd(start_order, 1ull) + 1ull;
        asm volatile("fence.sc.sys;" ::: "memory"); // task 3: acquire of task 0 at system scope
        const unsigned *src_3 = scratch + (size_t)0*n;
        for (size_t e = (size_t)(blockIdx.x - 1u)*blockDim.x + threadIdx.x; e < n; e += (size_t)1u*blockDim.x)
          scratch[(size_t)3*n + e] = op_3(src_3[e]);
        asm volatile("fence.sc.sys;" ::: "memory"); // task 3: release at system scope
      __syncthreads();
        if (threadIdx.x == 0) atomicAdd(completed + 3, 1u); // publish task 3
        done_3 = 1u; remaining--;
      }
    }

    ++spins;
    if (spins > spin_limit && threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, 4u);
  }

}

// ordinary_task is the reference implementation: the same tasks, one ordinary
// launch each, in rank order, with the host's stream ordering standing in for
// every fence above. It exists so the persistent result has something to be
// equal to.
extern "C" __global__ void ordinary_task(const unsigned *input, unsigned *scratch,
    unsigned n, unsigned task) {
  if (n > 1048576u) return;
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  int dep = dependency(task);
  if (i < n) {
    const unsigned *src = dep < 0 ? input : scratch + (size_t)dep*n;
    scratch[(size_t)task*n+i] = dispatch(task, src[i]);
  }
}
