// Host stand-ins for the device primitives ready.cuh uses, for
// ready_test.cc: std::atomic_ref on the counter words.
#include <atomic>
#include <thread>

#define __device__
#define __forceinline__ inline
#define READY_HOST

namespace ready {
inline unsigned long long load_relaxed(const unsigned long long* p) {
  return std::atomic_ref<unsigned long long>(*const_cast<unsigned long long*>(p))
      .load(std::memory_order_relaxed);
}
inline void fence_acquire() { std::atomic_thread_fence(std::memory_order_acquire); }
inline void fetch_max(unsigned long long* p, unsigned long long v) {
  std::atomic_ref<unsigned long long> a(*p);
  unsigned long long old = a.load(std::memory_order_relaxed);
  while (old < v && !a.compare_exchange_weak(old, v, std::memory_order_relaxed)) {
  }
}
inline void fetch_add(unsigned long long* p, unsigned long long v) {
  std::atomic_ref<unsigned long long>(*p).fetch_add(v, std::memory_order_relaxed);
}
inline void fence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
inline void pause(unsigned) { std::this_thread::yield(); }
}  // namespace ready
