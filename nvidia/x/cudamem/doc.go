// Package cudamem provides typed, memory-safe abstractions for GPU device allocations,
// host-pinned memory, streaming I/O, and stream-ordered memory pools.
//
// # Generic Device Slices
//
// [Slice] represents a typed contiguous array allocated in GPU device memory.
// It wraps a [driver.DevicePtr] with element-size and bounds information, preventing
// out-of-bounds access and pointer arithmetic bugs.
//
// # Grouped Allocations
//
// [Arena] holds allocations that share a lifetime. Buffers come from [AllocIn], which
// records the first failure instead of returning it, so a structure of many buffers is
// checked once with [Arena.Err] and released once with [Arena.Free].
//
// # Garbage Collection & Memory Pinning
//
// When transferring data between host Go memory and GPU device memory ([Slice.CopyFromHost],
// [Slice.CopyToHost]), package cudamem uses [runtime.Pinner] to keep the Go
// backing array pinned for the duration required by the CUDA call. Asynchronous
// copies retain the pin until stream completion.
//
// # Standard Library Streaming I/O
//
// [Reader] implements [io.Reader] and [io.WriterTo]; [Writer] implements
// [io.Writer] and [io.ReaderFrom]. Transfers between device memory and these
// interfaces use host staging buffers. They do not provide direct file or
// network DMA into device memory.
package cudamem
