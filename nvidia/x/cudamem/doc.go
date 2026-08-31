// Package cudamem provides typed, memory-safe abstractions for GPU device allocations,
// host-pinned memory, streaming I/O, and stream-ordered memory pools.
//
// # Generic Device Slices
//
// [Slice] represents a typed contiguous array allocated in GPU device memory.
// It wraps a [driver.DevicePtr] with element-size and bounds information, preventing
// out-of-bounds access and pointer arithmetic bugs.
//
// # Garbage Collection & Memory Pinning
//
// When transferring data between host Go memory and GPU device memory ([Slice.CopyFromHost],
// [Slice.CopyToHost]), package cudamem uses [runtime.Pinner] to guarantee that the Go
// runtime's moving garbage collector does not relocate the backing array during asynchronous
// or synchronous DMA operations.
//
// # Standard Library Streaming I/O
//
// [Reader] and [Writer] wrap byte slices in device memory to implement [io.ReaderFrom]
// and [io.WriterTo], enabling direct streaming DMA between GPU device buffers and
// standard Go interfaces such as [os.File] or [net.Conn].
package cudamem
