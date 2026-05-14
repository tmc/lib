// Package nvlshm provides cross-process NVLink shared GPU buffers built on
// the CUDA virtual memory management (VMM) API, with optional NVSwitch
// multicast mappings. It is a Go port of the memory substrate of MoonEP
// (github.com/MoonshotAI/MoonEP, csrc/nvl_shared_buffer.cuh and
// moonep/buffer.py).
//
// A distributed buffer is built cooperatively by N processes, one per GPU.
// Each rank allocates one physical chunk on its own device with cuMemCreate,
// exports it as a POSIX file descriptor, and exchanges descriptors with every
// other rank over AF_UNIX sockets (SCM_RIGHTS). Every rank then maps all N
// chunks back-to-back into one contiguous virtual address range:
//
//	VA:   [ chunk 0 ][ chunk 1 ] ... [ chunk N-1 ]
//	       resident   resident        resident
//	       on GPU 0   on GPU 1        on GPU N-1
//
// The picture is identical on every rank: chunk i is always GPU i's memory,
// and all chunks are mapped read-write, so a kernel on any rank can write
// directly into any other rank's chunk over NVLink.
//
// # Coordination
//
// The only external coordination nvlshm needs is a [Bootstrap]: a broadcast
// of one string and a barrier. Package tcpboot provides a trivial TCP
// implementation; MPI- or collective-library-backed implementations can be
// substituted. All calls that take a Bootstrap are collective: every rank in
// the world must call them in the same order with compatible arguments, or
// ranks deadlock in a barrier.
//
// # Alignment
//
// Chunk sizes must be a multiple of the device's VMM allocation granularity
// (see [Granularity] and [PadTo]); [DistAllocMulticast] additionally requires
// a multiple of the multicast granularity ([MulticastGranularity]). Aligned
// chunks are what make the mapped region a dense array: element offsets and
// byte offsets never diverge.
//
// # Multicast
//
// [DistAllocMulticast] overlays an NVSwitch multicast mapping on the same
// physical chunks. A store to the multicast address ([DistBuffer.McPtr]) is
// fanned out by the switch to the corresponding offset of every rank's local
// chunk. Registration follows a strict order enforced internally (create on
// rank 0 → import → add device → barrier → bind → barrier); violating it at
// the driver level fails with CUDA_ERROR_ILLEGAL_STATE or hangs, which is why
// the whole sequence is packaged rather than exposed piecemeal.
//
// # Ownership and lifetime
//
// Buffers own their driver resources and release them in [DistBuffer.Close]
// (multicast view first, then chunk mappings, then the address reservation).
// Close synchronizes the whole context before unmapping, so in-flight kernels
// on any stream finish first. Close is idempotent. Closing is collective in
// spirit: a rank that unmaps while a peer still writes into its chunk gets
// undefined behavior, so tear down behind a barrier (Close does not take a
// Bootstrap; the caller sequences it).
//
// File descriptors never outlive setup: each rank's exported fd is closed as
// soon as the exchange returns, and received fds are closed as soon as they
// are imported.
//
// # Requirements
//
// The process must have a CUDA context current on the calling goroutine
// (typically driver.Init, Device.PrimaryCtxRetain, Context.SetCurrent, with
// runtime.LockOSThread) before calling any allocation function. All ranks
// must run on the same host (POSIX fd exchange) with NVLink peer access
// between the GPUs.
package nvlshm
