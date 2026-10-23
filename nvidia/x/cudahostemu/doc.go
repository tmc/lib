// Package cudahostemu runs CUDA kernel source on the host CPU, for testing
// kernels where no GPU is available.
//
// The emulation is a C++20 header, cudahostemu.h, that defines the CUDA
// qualifiers, built-in variables, warp collectives, cp.async, ldmatrix,
// mma.sync and wgmma (bf16, f16 and e4m3 forms), thread-block clusters, and
// a host-memory subset of the CUDA runtime. Each CUDA thread is a host
// thread. Shared memory, cp.async destinations and cudaMalloc memory start
// as 0xff bytes (NaN in every float format the header handles), so a
// kernel that reads memory it never wrote fails visibly; define
// EG_EMU_ZERO_SMEM to start shared memory zeroed. sgl_mock.h adds the parts
// of SGLang's JIT headers and tvm-ffi that generated launcher structs use.
//
// [Run] compiles kernel source with a launch statement and named buffers,
// runs it, and returns each buffer's final bytes. [WriteHeaders] writes the
// headers for callers that drive the compiler themselves.
//
// The emulation checks a kernel's arithmetic, indexing and synchronization
// on small inputs. It does not model timing, memory ordering beyond
// barriers, or the GPU's floating-point contraction choices.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudahostemu
