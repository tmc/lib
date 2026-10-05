// Cudabind generates Go bindings for the kernels of a CUDA C++ source file.
//
// Usage:
//
//	cudabind [flags] file.cu
//
// Cudabind parses the file with libclang in CUDA mode and writes a Go file
// that embeds the source and the headers it includes, compiles them with
// NVRTC when loaded, and has a typed method for each __global__ function
// and a getter and setter for each __device__ or __constant__ variable.
// It is meant for go:generate, run in the package directory:
//
//	//go:generate go run github.com/tmc/lib/nvidia/x/cmd/cudabind -type vecModule -I shim -std c++20 vec.cu
//
// For a kernel
//
//	__global__ void step(Env* envs, const float* actions, int n)
//
// the module type gets the method
//
//	func (m *vecModule) step(stream *cuda.Stream, grid, block [3]uint32,
//		envs cudamem.Slice[env], actions cudamem.Slice[float32], n int32) error
//
// C types map to Go types as follows. Integer and floating-point types map
// to the Go types of the same size. __half, __nv_bfloat16 and other 16-bit
// floating-point types map to uint16. A pointer parameter is a
// [cudamem.Slice] of its element type, void* a Slice[byte] and T** a
// Slice[driver.DevicePtr]; a pointer field is a driver.DevicePtr. A struct
// declared in the package directory maps to a generated Go struct of the
// same layout, named by -prefix and the C name, with fields named in
// CamelCase. A struct from a toolkit header maps by its shape: uint4 to
// [4]uint32, __nv_bfloat162 to [2]uint16, anything else to an array of its
// bytes. Unions and structs with bit-fields also map to arrays of bytes.
//
// Kernels with C++ linkage are bound by their NVRTC name expressions.
// Cudabind rejects overloaded kernels and skips template kernels with a
// warning; an extern "C" kernel that calls the template can be bound instead.
//
// The constants of an enum declared in the package directory become Go
// constants, named like structs: with -prefix p, BREAKOUT_STATE_GROUPS is
// pBreakoutStateGroups and geom::Side::LEFT is pGeomSideLeft. Parameters and
// fields of enum type have the Go integer type of its size. Object-like
// macros defined in the package directory that clang evaluates to a number
// become constants too, with the value they have at the end of the file;
// -macros selects which.
//
// Clang supplies the sizes and field offsets. Where Go would place a field
// earlier than C does, as it does after a uint4, which C aligns to 16
// bytes, the Go struct gets explicit padding. The generated file asserts
// every size and offset at compile time, and a generated test compiles the
// source with NVRTC and checks them, and the constants, against the device
// compiler.
//
// The generated names are unexported when the -type name is. Each binding
// in a package needs its own -type, and its own -prefix if two bindings
// declare C types of the same name.
//
// The flags are:
//
//	-type name       Go type of the loaded module (required)
//	-prefix p        prefix of the Go names of C structs
//	-o file          output file (default file.cu.gen.go)
//	-I dir           include directory, repeatable
//	-D name[=value]  macro definition, repeatable
//	-std std         C++ standard, such as c++20
//	-default-device  compile unannotated functions as device functions
//	-kernels a,b     kernels to bind (default all)
//	-globals a,b     variables to bind (default all)
//	-macros a,b      macros to bind (default all that are numbers)
//	-shared a,b      kernels that take dynamic shared memory
//	-arch sm_NN      architecture clang parses for (default sm_80)
//	-libclang path   libclang to parse with (default $CUDABIND_LIBCLANG)
//
// The package internal/testvec is a small example of a binding and its use.
//
// Include directories are searched after the toolkit's headers when clang
// parses, so stand-ins for toolkit headers that only NVRTC needs do not
// hide the real ones, and as in-memory headers when NVRTC compiles.
//
// Cudabind needs a libclang whose resource directory has clang's CUDA
// wrapper headers, as a full clang installation does; a libclang package on
// its own may not. CUDA versions newer than the libclang knows are parsed
// as the newest it does, which suffices for declarations. Generation needs
// no GPU and no NVRTC; the generated code needs both.
package main
