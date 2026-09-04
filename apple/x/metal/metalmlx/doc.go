// Package metalmlx is a minimal Go conversion of the host-side runtime of
// MLX's Metal backend (mlx/backend/metal), built on the pure-Go Metal
// bindings. It carries over the backend's architecture — a caching allocator
// over shared-storage buffers, residency sets that keep allocations wired,
// library and pipeline caches, and a command encoder that tracks what each
// dispatch reads and writes — without the MLX array, scheduler, or primitive
// layers.
//
// Mapping from the C++ sources:
//
//	allocator.cpp/.h  -> allocator.go   (Allocator)
//	buffer_cache.h    -> buffercache.go (bufferCache)
//	resident.cpp/.h   -> resident.go    (ResidencySets)
//	device.cpp/.h     -> device.go      (Device), encoder.go (CommandEncoder)
//	event.cpp/.h      -> event.go       (Event)
//	fence.cpp         -> fence.go       (Fence)
//	kernels/steel/*   -> abi.gen.go     (kernel parameter structs; generated)
//
// The conversion is deliberately minimal. Omitted relative to the C++: the
// JIT kernel source assembly (jit_kernels.cpp) and every primitive built on
// it, the default metallib search through SwiftPM bundles and frameworks
// (LibraryFromPath takes the path), linked functions in a pipeline, and
// MLX's temporaries, which only exist to hold arrays alive. MLX's error
// poisoning, which carries a failed command buffer's error into every event
// it signaled, is reduced to an error recorded on the encoder and returned by
// [CommandEncoder.Synchronize].
//
// Two Objective-C rules the C++ gets from ARC and scoped pools have to be
// followed by hand here:
//
//   - Objects that outlive a call are retained explicitly, and released when
//     dropped. The command buffer, the compute encoder, and the fence an
//     encoder signals are the ones that matter.
//   - Autoreleased objects need a pool. Go has no per-thread pool, so calls
//     that make them run inside [WithPool]. Callers that make many Metal
//     calls of their own should do the same.
//
// A [Device] and its [Allocator] are safe for concurrent use. A
// [CommandEncoder] is not, which matches MLX: one encoder belongs to one
// stream. A Device returned by [NewDevice] is caller-owned and must be closed;
// the process-wide Device returned by [GetDevice] is not caller-owned. Device
// Close refuses live allocations and command encoders. Events and fences must
// outlive every submitted command buffer that refers to them.
//
// abi.gen.go is produced by gen_abi.go, which measures MLX's live headers
// with clang so the kernel parameter structs track the C++ source rather than
// a transcription of it. The generator refuses locally modified parameter
// headers and records the exact MLX commit in both generated files. With them
// and a library loaded from MLX's own
// metallib, this package dispatches MLX's kernels unchanged; steelgemm_test.go
// runs its steel GEMM and checks the result.
//
//go:generate go run gen_abi.go
package metalmlx
