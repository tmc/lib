// Package cudamlx is a minimal Go conversion of the host-side runtime of
// MLX's CUDA backend (mlx/backend/cuda), built on the pure-Go driver and
// nvrtc bindings. It carries over the backend's architecture — a caching
// unified-memory allocator, a batching worker synchronized with the GPU,
// an nvrtc JIT module cache, and a CUDA-graph command encoder — without
// the MLX array, scheduler, or primitive layers.
//
// Mapping from the C++ sources:
//
//	allocator.cpp/.h   -> allocator.go     (Allocator, smallSizePool)
//	buffer_cache.h     -> buffercache.go   (bufferCache)
//	device.cpp/.h      -> device.go        (Device), encoder.go (CommandEncoder)
//	worker.cpp/.h      -> worker.go        (Worker)
//	event.h/event.cu   -> event.go         (Event, SharedEvent)
//	jit_module.cpp/.h  -> jitmodule.go     (JitModule, KernelArgs)
//	device/config.h    -> abi.gen.go       (MaxNDim, Shape, Strides; generated)
//
// The conversion is deliberately minimal. Omitted relative to the C++:
// the CUDA graph exec cache and graph update path (each commit
// instantiates a fresh exec), stream-capture contexts, cluster
// dimensions, and everything above the runtime (primitives, gemms,
// reductions). Where MLX hand-rolls a mechanism the driver API already
// provides, the driver API is used instead: SharedEvent uses
// cuStreamWaitValue32/cuStreamWriteValue32 in place of MLX's signal
// kernels, and Worker waits on an event from a goroutine instead of
// cudaLaunchHostFunc.
//
// abi.gen.go is produced by gen_abi.go, which measures MLX's live headers
// with clanglayout (via `appledocs clangdump layout --json`) so kernel
// ABI constants and mirrored records track the C++ source rather than a
// transcription of it.
//
//go:generate go run gen_abi.go
package cudamlx
