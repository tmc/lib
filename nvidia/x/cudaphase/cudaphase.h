// cudaphase.h: per-block phase timestamps written from inside a kernel.
//
// A traced kernel takes a uint64 buffer of CUDAPHASE_SLOTS values per
// block and marks the start of each phase it wants timed:
//
//	CUDAPHASE_MARK(trace, 0, cudaphase_now());   // block start
//	...
//	CUDAPHASE_MARK(trace, 1, cudaphase_now());   // main loop done
//	CUDAPHASE_MARK(trace, 7, cudaphase_smid());  // a slot may hold any value
//
// Zero means unset in every slot, so cudaphase_smid returns the SM number
// plus one.
// Thread 0 of each block writes the block's row; a null buffer makes every
// mark a no-op, so one kernel serves traced and untraced launches. The row
// of block (x, y, z) starts at ((z*gridDim.y + y)*gridDim.x + x) *
// CUDAPHASE_SLOTS. cudaphase.py (or the Go package) summarizes the buffer.
//
// cudaphase_now reads %globaltimer, in nanoseconds; its resolution is
// coarse (stamps on an H200 move in 256 ns steps), and the summary reports
// the step it observed. Under cudahostemu a host clock stands in.
#pragma once
#include <stdint.h>

#ifndef CUDAPHASE_SLOTS
#define CUDAPHASE_SLOTS 16
#endif

#if defined(CUDAHOSTEMU)
#include <chrono>
inline unsigned long long cudaphase_now() {
    return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline unsigned cudaphase_smid() { return 1; }
#else
__device__ __forceinline__ unsigned long long cudaphase_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned cudaphase_smid() {
    unsigned s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s + 1;
}
#endif

#define CUDAPHASE_MARK(buf, slot, value)                                                              \
    do {                                                                                              \
        if ((buf) != nullptr && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)             \
            (buf)[(((size_t)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *          \
                      CUDAPHASE_SLOTS + (slot)] = (unsigned long long)(value);                        \
    } while (0)
