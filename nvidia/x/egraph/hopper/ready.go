package hopper

// ReadyPrelude holds the device and host-emulated primitives for epoch-tagged
// ready counters inside persistent multi-stage megakernels, following
// nvidia/x/cudaready/ready.cuh.
//
// Each counter slot in global memory is a 64-bit word holding
// (uint64_t(epoch) << 32) | count, allocated and zeroed once at initialization
// and never reset between launches or CUDA graph replays. A producer stage
// publishes its global memory writes with __syncthreads(), then calls
// eg_ready_signal(&ready[slot], epoch, 1) from thread 0 (which issues
// __threadfence(), advances the slot to epoch << 32 via atomicMax, and adds
// units via atomicAdd). A consumer stage calls
// eg_ready_wait(&ready[slot], epoch, need) from thread 0 (which polls with
// relaxed GPU-scope loads and exponential nanosleep backoff, then issues
// fence.acq_rel.gpu) before __syncthreads() and reading the intermediate
// buffer.
const ReadyPrelude = `#ifndef EG_READY_PRELUDE
#define EG_READY_PRELUDE

#if defined(EG_HOST_EMU)
#define EG_READY_STAGE(slot, epoch, expected, ptr, bytes) \
    eg_emu_ready_stage(slot, epoch, expected, ptr, bytes)
inline void eg_emu_asm_bar_sync(const char* s) {
    int id = 0, count = 0;
    if (std::sscanf(s, "bar.sync %d, %d;", &id, &count) == 2) {
        eg_emu_bar_sync(id, count);
    }
}
#define asm
#define volatile(...) eg_emu_asm_bar_sync(__VA_ARGS__)
#else
#define EG_READY_STAGE(slot, epoch, expected, ptr, bytes) ((void)0)
#endif

// eg_ready_resolve_epoch resolves the 1-based launch epoch for a persistent
// megakernel of total_blocks CTAs. When epoch > 0, it returns epoch directly.
// When epoch == 0 (e.g. inside an instantiated CUDA graph replayed without
// host parameter updates), thread 0 of each CTA atomically claims one ticket
// from *epoch_slot; since each launch dispatches total_blocks CTAs and
// completes before the next launch begins, 1 + (ticket / total_blocks)
// yields the identical 1-based epoch across all CTAs of the launch.
EG_DEV unsigned int eg_ready_resolve_epoch(unsigned long long* epoch_slot, unsigned int epoch, unsigned int total_blocks) {
    if (epoch != 0) return epoch;
    __shared__ unsigned int eg_s_epoch;
    if (threadIdx.x == 0) {
        unsigned long long t = atomicAdd(epoch_slot, 1ULL);
        eg_s_epoch = 1u + (unsigned int)(t / (unsigned long long)(total_blocks > 0 ? total_blocks : 1u));
    }
    __syncthreads();
    return eg_s_epoch;
}

// eg_ready_signal publishes units of producer work for epoch on *slot.
// atomicMax advances a word left by an earlier epoch to (epoch << 32) and
// leaves a current-epoch word unchanged; atomicAdd then increments the count
// within this epoch without a CAS retry loop.
EG_DEV void eg_ready_signal(unsigned long long* slot, unsigned int epoch, unsigned int units = 1) {
#if defined(EG_HOST_EMU)
    eg_emu_ready_signal(slot, epoch, units);
#else
    __threadfence();
    atomicMax(slot, (unsigned long long)epoch << 32);
    atomicAdd(slot, (unsigned long long)units);
#endif
}

// eg_ready_wait spins until *slot holds at least need units in epoch, then
// orders subsequent reads after the producers' global stores.
EG_DEV void eg_ready_wait(unsigned long long* slot, unsigned int epoch, unsigned int need) {
#if defined(EG_NO_READY_WAIT)
    (void)slot;
    (void)epoch;
    (void)need;
#elif defined(EG_HOST_EMU)
    eg_emu_ready_wait(slot, epoch, need);
#else
    if (need == 0) return;
    const unsigned long long want = ((unsigned long long)epoch << 32) | need;
    for (unsigned int ns = 64; ; ns = ns < 1024 ? 2 * ns : ns) {
        unsigned long long v;
        asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(slot) : "memory");
        if (v >= want) break;
        __nanosleep(ns);
    }
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
}

#endif // EG_READY_PRELUDE
`

// PackReadyWord encodes an (epoch, count) pair into a 64-bit ready counter word
// as uint64(epoch)<<32 | uint64(count).
func PackReadyWord(epoch, count uint32) uint64 {
	return uint64(epoch)<<32 | uint64(count)
}

// UnpackReadyWord decodes a 64-bit ready counter word into its (epoch, count)
// components.
func UnpackReadyWord(word uint64) (epoch, count uint32) {
	return uint32(word >> 32), uint32(word)
}
