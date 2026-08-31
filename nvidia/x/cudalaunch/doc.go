// Package cudalaunch provides an ergonomic, type-safe builder for launching CUDA kernels
// and automatic parameter marshalling without unsafe pointer bookkeeping.
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
// [driver.DevicePtr], structs, pointers, and types implementing [cudamem.Slice].
package cudalaunch
