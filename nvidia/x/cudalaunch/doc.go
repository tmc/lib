// Package cudalaunch configures CUDA kernel launches and marshals Go values as
// kernel parameters.
//
// A Launcher borrows its function, module, stream, and device buffers. The
// caller must keep them live through stream completion. A Launcher may be
// reused after a launch completes, but its configuration methods and Launch
// must not be called concurrently. Separate launchers may submit concurrently
// to separate streams.
//
// # Kernel Execution Builder
//
// [Kernel] constructs a [Launcher] configuration supporting grid/block dimensioning,
// dynamic shared memory allocation, and stream assignment:
//
//	err := cudalaunch.Kernel(fn).
//		Grid(128, 1).
//		Block(256, 1).
//		SharedMem(1024).
//		On(stream).
//		Launch(d_a.Ptr(), d_b.Ptr(), d_c.Ptr(), int32(n))
//
// # Automatic Parameter Marshalling
//
// [Launcher.Launch] automatically packs Go arguments into the array-of-pointers layout
// expected by cuLaunchKernel. Supported argument types include Go integers, floats,
// [driver.DevicePtr], structs, pointers, and any [DevicePtrProvider] such as
// [github.com/tmc/lib/nvidia/x/cudamem.Slice]. An empty [DeviceSlice] packs as a null
// device pointer, so an optional argument such as a missing bias vector is passed as a
// zero Slice rather than a hand-rolled null sentinel.
//
// # Kernel Binding
//
// [Bind] resolves a whole set of kernels from a loaded module into the fields of a
// struct, naming each kernel with a "cuda" struct tag:
//
//	type kernels struct {
//		Embed driver.Function `cuda:"EmbeddingLookup"`
//		Norm  driver.Function `cuda:"RMSNorm"`
//	}
//	var k kernels
//	if err := cudalaunch.Bind(mod, &k); err != nil {
//		return err
//	}
package cudalaunch
