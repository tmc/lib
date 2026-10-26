// Package cudahostemu runs CUDA kernel source on the host CPU, for testing
// kernels where no GPU is available.
//
// The emulation is a C++20 header, cudahostemu.h, that defines the CUDA
// qualifiers, built-in variables, mask-aware warp collectives (including
// disjoint sub-warp masks), named CTA barriers (bar.sync id, count),
// cp.async, ldmatrix, mma.sync and wgmma (bf16, f16 and e4m3 forms, with
// K-major or MN-major layouts and no/32B/64B/128B swizzle descriptors),
// thread-block clusters, and a host-memory subset of the CUDA runtime. Each
// CUDA thread is a host thread. Shared memory, cp.async destinations and
// cudaMalloc memory start as 0xff bytes (NaN in every float format the
// header handles), so a kernel that reads memory it never wrote fails
// visibly; define EG_EMU_ZERO_SMEM to start shared memory zeroed. cp.async
// and, in kernels that issue wgmma.fence, wgmma are modeled at their latest
// legal timing, so a result read or an operand overwritten before the
// matching wait_group shows up on the host; EG_EMU_SYNC_WGMMA turns the
// wgmma model off. Hopper mbarriers, cp.async.bulk and 2D TMA tile loads
// (with the 128-byte swizzle) are modeled adversarially: a copy's
// destination reads NaN until a thread waits on the phase that completes it,
// so a missing wait or a wait for the wrong phase parity shows up, and a
// satisfied wait returns only once no other thread of the block can run, so
// a producer that refills a stage before its consumers release it shows up
// too; EG_EMU_SYNC_MBAR turns that model off. sgl_mock.h adds the parts of SGLang's JIT headers and
// tvm-ffi that generated launcher structs use.
//
// [Run] compiles kernel source with a launch statement and named buffers,
// runs it, and returns each buffer's final bytes. [WriteHeaders] writes the
// headers for callers that drive the compiler themselves.
//
// [MMAM16N8K16A], [MMAM16N8K16B], [MMAM16N8C], [MMAM16N8K32A], [MMAM16N8K32B],
// [MMAM16N8K64NVFP4A], [MMAM16N8K64NVFP4B], [LDMatrixM8N8X4], [WGMMADesc], and
// [SwizzleOffset] report the PTX ISA register-fragment and shared-memory
// descriptor layouts that the header emulates.
//
// The emulation checks a kernel's arithmetic, indexing and synchronization
// on small inputs. It does not model timing, memory ordering beyond
// barriers, or the GPU's floating-point contraction choices.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudahostemu
