// Package cuda provides idiomatic, high-level Go abstractions over NVIDIA CUDA
// devices, execution contexts, streams, and events.
//
// All operations build upon the purego dynamic bindings in [github.com/tmc/lib/nvidia/driver],
// requiring no cgo and no CUDA development headers during build time.
//
// # Concurrency and Thread Affinity
//
// The CUDA Driver API associates execution contexts with the calling OS thread.
// Because the Go runtime multiplexes goroutines across multiple OS threads (M:N scheduling),
// setting a thread-local context via [driver.CtxSetCurrent] is hazardous if the goroutine
// switches threads across scheduling points.
//
// Package cuda avoids context-drift by managing device primary contexts and providing
// explicit, stream-ordered operations. When thread-local context manipulation is necessary,
// use [Device.WithContext], which pins the calling goroutine to its OS thread
// via [runtime.LockOSThread] for the duration of the closure.
//
// # Streams
//
// [Device.NewStream] creates an independent stream; [Device.DefaultStream] returns the
// device's legacy default stream. Because DefaultStream never returns nil, an API that
// takes a stream can require one and stay stream-ordered throughout, rather than
// branching between synchronous and asynchronous paths. A caller that wants the host to
// wait synchronizes the stream.
//
// # Context Cancellation
//
// [Stream.Synchronize] accepts a [context.Context]. If the context is cancelled or expires
// before the stream completes all enqueued operations, [Stream.Synchronize] unblocks
// while polling and returns [context.Canceled] or [context.DeadlineExceeded]. Native
// calls and lock acquisition are not interrupted by cancellation. Hardware
// execution on the physical GPU Streaming Multiprocessors continues until the kernel finishes.
package cuda
