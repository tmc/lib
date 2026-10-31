package hopper

import (
	"fmt"
	"strings"
)

// SysBarrierMode selects the system-scope (.sys) ready-counter synchronization
// mechanism across peer GPUs or NVSwitch fabric domains.
type SysBarrierMode uint8

const (
	// SysBarrierPeerAtomic synchronizes across peer-mapped HBM using
	// system-scope acquire/release fences (fence.acq_rel.sys) and
	// system-scope global atomics (atomicAdd_system).
	SysBarrierPeerAtomic SysBarrierMode = iota

	// SysBarrierMultimem synchronizes across an NVSwitch NVLS multicast team
	// using inline PTX multimem.red.release.sys.global.add.u64 on sm_90+ and
	// system-scope acquire loads (ld.acquire.sys.global.u64).
	SysBarrierMultimem
)

func (m SysBarrierMode) String() string {
	switch m {
	case SysBarrierPeerAtomic:
		return "peer_atomic"
	case SysBarrierMultimem:
		return "multimem"
	default:
		return fmt.Sprintf("SysBarrierMode(%d)", uint8(m))
	}
}

// ReadySysPreludePeer holds the device and host-emulated primitives for
// system-scope (.sys) epoch-tagged ready counters over peer-mapped global
// memory using fence.acq_rel.sys, atomicAdd_system, and
// ld.acquire.sys.global.u64.
const ReadySysPreludePeer = ReadyPrelude + `
#ifndef EG_READY_SYS_PRELUDE
#define EG_READY_SYS_PRELUDE

// eg_ready_signal_sys publishes units of producer work for epoch on *slot
// at system scope (.sys) across peer GPUs using fence.acq_rel.sys and
// system-scope atomics.
EG_DEV void eg_ready_signal_sys(unsigned long long* slot, unsigned int epoch, unsigned int units = 1) {
#if defined(EG_HOST_EMU)
    eg_emu_ready_signal(slot, epoch, units);
#else
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    atomicMax(slot, (unsigned long long)epoch << 32);
    atomicAdd_system(slot, (unsigned long long)units);
#endif
}

// eg_ready_wait_sys spins with system-scope acquire loads
// (ld.acquire.sys.global.u64) and exponential nanosleep backoff until *slot
// holds at least need units in epoch, then issues fence.acq_rel.sys.
EG_DEV void eg_ready_wait_sys(unsigned long long* slot, unsigned int epoch, unsigned int need) {
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
        asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(slot) : "memory");
        if (v >= want) break;
        __nanosleep(ns);
    }
    asm volatile("fence.acq_rel.sys;" ::: "memory");
#endif
}

#endif // EG_READY_SYS_PRELUDE
`

// ReadySysPreludeMultimem holds the device and host-emulated primitives for
// system-scope (.sys) epoch-tagged ready counters over NVSwitch NVLS multicast
// addresses using multimem.red.release.sys.global.add.u64 and
// ld.acquire.sys.global.u64.
const ReadySysPreludeMultimem = ReadyPrelude + `
#ifndef EG_READY_SYS_PRELUDE
#define EG_READY_SYS_PRELUDE

// eg_ready_signal_sys publishes units of producer work for epoch on *slot
// at system scope (.sys) across an NVSwitch multicast team using
// fence.acq_rel.sys, atomicMax, and multimem.red.release.sys.global.add.u64.
EG_DEV void eg_ready_signal_sys(unsigned long long* slot, unsigned int epoch, unsigned int units = 1) {
#if defined(EG_HOST_EMU)
    eg_emu_ready_signal(slot, epoch, units);
#else
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    atomicMax(slot, (unsigned long long)epoch << 32);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !defined(EG_HOST_EMU)
    asm volatile("multimem.red.release.sys.global.add.u64 [%0], %1;" :: "l"(slot), "l"((unsigned long long)units) : "memory");
#else
    atomicAdd_system(slot, (unsigned long long)units);
#endif
#endif
}

// eg_ready_wait_sys spins with system-scope acquire loads
// (ld.acquire.sys.global.u64) and exponential nanosleep backoff until *slot
// holds at least need units in epoch, then issues fence.acq_rel.sys.
EG_DEV void eg_ready_wait_sys(unsigned long long* slot, unsigned int epoch, unsigned int need) {
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
        asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(slot) : "memory");
        if (v >= want) break;
        __nanosleep(ns);
    }
    asm volatile("fence.acq_rel.sys;" ::: "memory");
#endif
}

#endif // EG_READY_SYS_PRELUDE
`

// ReadySysPrelude returns the system-scope ready-counter prelude for mode.
func ReadySysPrelude(mode SysBarrierMode) string {
	if mode == SysBarrierMultimem {
		return ReadySysPreludeMultimem
	}
	return ReadySysPreludePeer
}

// SysEpochBarrierConfig configures an in-megakernel system-scope (.sys) epoch
// barrier across WorldSize tensor-parallel ranks.
type SysEpochBarrierConfig struct {
	// Name is the emitted barrier kernel and device helper base name.
	// When empty, it defaults to "sys_epoch_barrier".
	Name string

	// WorldSize is the number of tensor-parallel ranks participating in the
	// barrier (must be >= 1).
	WorldSize int

	// BlocksPerRank is the number of CTAs per rank signaling the barrier slot.
	// When 0, it defaults to 1.
	BlocksPerRank int

	// Slot is the index into the 64-bit ready counter array used by the barrier.
	Slot int

	// Mode selects peer-mapped system atomics or NVSwitch multimem reductions.
	Mode SysBarrierMode
}

func (c SysEpochBarrierConfig) norm() (SysEpochBarrierConfig, error) {
	c.Name = strings.TrimSpace(c.Name)
	if c.Name == "" {
		c.Name = "sys_epoch_barrier"
	}
	if !cIdent(c.Name) {
		return c, fmt.Errorf("ready sys barrier: bad name %q", c.Name)
	}
	if c.WorldSize < 1 {
		return c, fmt.Errorf("ready sys barrier %s: WorldSize %d < 1", c.Name, c.WorldSize)
	}
	if c.BlocksPerRank < 0 {
		return c, fmt.Errorf("ready sys barrier %s: BlocksPerRank %d < 0", c.Name, c.BlocksPerRank)
	}
	if c.BlocksPerRank == 0 {
		c.BlocksPerRank = 1
	}
	if c.Slot < 0 {
		return c, fmt.Errorf("ready sys barrier %s: Slot %d < 0", c.Name, c.Slot)
	}
	if c.Mode > SysBarrierMultimem {
		return c, fmt.Errorf("ready sys barrier %s: bad mode %v", c.Name, c.Mode)
	}
	return c, nil
}

// TotalNeed returns the total number of producer units across all ranks
// (WorldSize * BlocksPerRank) that each waiting CTA expects in the barrier slot.
func (c SysEpochBarrierConfig) TotalNeed() int {
	blocks := c.BlocksPerRank
	if blocks <= 0 {
		blocks = 1
	}
	if c.WorldSize <= 0 {
		return 0
	}
	return c.WorldSize * blocks
}

// EmitSysEpochBarrierSource emits a CUDA C++ translation unit containing the
// system-scope ready prelude for cfg.Mode, a device helper Name_sync, a
// standalone barrier kernel Name, and host launchers Name_launch / Name_run.
func EmitSysEpochBarrierSource(cfg SysEpochBarrierConfig) (string, error) {
	cfg, err := cfg.norm()
	if err != nil {
		return "", err
	}
	need := cfg.TotalNeed()
	totalBlocks := cfg.WorldSize * cfg.BlocksPerRank

	var b strings.Builder
	b.WriteString(ReadySysPrelude(cfg.Mode))
	fmt.Fprintf(&b, "\n// %s: system-scope (%s) epoch barrier across %d ranks (%d blocks/rank, slot %d, need %d).\n",
		cfg.Name, cfg.Mode, cfg.WorldSize, cfg.BlocksPerRank, cfg.Slot, need)
	fmt.Fprintf(&b, "EG_DEV void %s_sync(unsigned long long* ready, unsigned int epoch) {\n", cfg.Name)
	b.WriteString("    __syncthreads();\n")
	b.WriteString("    if (threadIdx.x == 0) {\n")
	fmt.Fprintf(&b, "        eg_ready_signal_sys(&ready[%d], epoch, 1);\n", cfg.Slot)
	fmt.Fprintf(&b, "        eg_ready_wait_sys(&ready[%d], epoch, %d);\n", cfg.Slot, need)
	b.WriteString("    }\n")
	b.WriteString("    __syncthreads();\n")
	b.WriteString("}\n\n")

	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    unsigned long long* __restrict__ ready,\n    unsigned int epoch) {\n", cfg.Name)
	fmt.Fprintf(&b, "    %s_sync(ready, epoch);\n", cfg.Name)
	b.WriteString("}\n\n")

	fmt.Fprintf(&b, "inline void %s_launch(unsigned long long* __restrict__ ready, unsigned int epoch, cudaStream_t stream) {\n", cfg.Name)
	fmt.Fprintf(&b, "    const dim3 grid(%d), block(256);\n", totalBlocks)
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, ready, epoch);\n", cfg.Name)
	b.WriteString("}\n\n")
	fmt.Fprintf(&b, "inline void %s_run(unsigned long long* __restrict__ ready, unsigned int epoch, cudaStream_t stream) {\n", cfg.Name)
	fmt.Fprintf(&b, "    %s_launch(ready, epoch, stream);\n", cfg.Name)
	b.WriteString("}\n")
	return b.String(), nil
}
