// Epoch-tagged ready counters: dataflow dependencies between the stages of
// a persistent kernel, in place of grid-wide barriers.
//
// A counter set is an array of uint64 words in global memory, zeroed once
// when it is allocated and never reset. Each use of the set (one call of
// the stage pair that shares it) has an epoch, at least 1 and larger than
// every earlier one. A word holds epoch << 32 | count; a word left by an
// earlier epoch counts as zero, so a new epoch starts every counter at
// zero without a reset.
//
// A producer publishes its writes for dependency dep with
//
//	writes by any threads of the block
//	barrier over those threads (__syncthreads or a named barrier)
//	ready::signal(c, dep, epoch, n);  // one thread
//
// and a consumer that needs count units of dep calls
//
//	ready::wait(c, dep, count, epoch);  // one thread
//	barrier over the threads that read (__syncwarp, __syncthreads)
//
// before reading them. Bulk copies (TMA) read through the async proxy:
// each issuing thread follows the barrier with fence.proxy.async.global.
// wait polls with relaxed loads and backs off up to about 1 µs, so many
// waiters do not slow the producers' memory traffic.
//
// It is safe only among co-resident blocks: a cooperative launch, or a
// grid no larger than the device holds at once. A consumer spins, so a
// producer that is not yet resident never runs and the wait never
// returns. For the same reason a block must finish the producer work it
// owes before it waits.

namespace ready {

#ifdef READY_HOST
// Host test build: ready_host.h defines load_relaxed, fetch_max, fetch_add,
// fence, fence_acquire and pause.
#else
__device__ __forceinline__ unsigned long long load_relaxed(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void fence_acquire() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ __forceinline__ void fetch_max(unsigned long long* p, unsigned long long v) {
  atomicMax(p, v);
}

__device__ __forceinline__ void fetch_add(unsigned long long* p, unsigned long long v) {
  atomicAdd(p, v);
}

__device__ __forceinline__ void fence() { __threadfence(); }

__device__ __forceinline__ void pause(unsigned ns) { __nanosleep(ns); }
#endif

// signal adds n to counter dep for epoch. The max sets a word left by an
// earlier epoch to epoch << 32 and leaves a current one alone; each
// signaler's add follows its own max, so every add lands in this epoch.
// Two atomics and no retry loop: a CAS loop serialized 128 signalers of
// one MoE expert into tens of µs on H200.
__device__ __forceinline__ void signal(unsigned long long* c, int dep, unsigned epoch,
                                       unsigned n) {
  fence();
  fetch_max(c + dep, (unsigned long long)epoch << 32);
  fetch_add(c + dep, n);
}

// wait returns once counter dep holds at least count for epoch. A count
// of 0 needs no signal.
__device__ __forceinline__ void wait(const unsigned long long* c, int dep, unsigned count,
                                     unsigned epoch) {
  if (count == 0) return;
  const unsigned long long want = (unsigned long long)epoch << 32 | count;
  for (unsigned ns = 64; load_relaxed(c + dep) < want; ns = ns < 1024 ? 2 * ns : ns) pause(ns);
  fence_acquire();
}

}  // namespace ready
