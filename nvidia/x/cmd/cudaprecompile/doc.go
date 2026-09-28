// Cudaprecompile compiles CUDA C++ sources ahead of time for a Go package.
//
// Usage:
//
//	cudaprecompile [flags] file.cu...
//
// It is meant for go:generate:
//
//	//go:generate go run github.com/tmc/lib/nvidia/x/cmd/cudaprecompile softmax.cu
//
// Cudaprecompile writes a Go file declaring a [cudaprecompiled.Module] that
// embeds the sources, their headers, and an image with machine code for
// each -arch and PTX for -ptx. At run time the driver loads the machine code
// for the device, or compiles the PTX for a newer device. If the image is
// stale or unusable, the module compiles the sources with NVRTC instead, so
// precompiling is an optimization, never a requirement. (The image is a
// fatbin, the format nvcc produces with -fatbin.)
//
// Sources and headers must be in the package directory or below it. The
// headers embedded are the .h, .cuh, .hpp and .inl files in the directories
// of the sources and, recursively, in each -I directory.
//
// Several sources are compiled as relocatable device code and linked with
// link-time optimization, which keeps inlining across files. -nolto links
// machine code instead, which is faster to build but slower to run and
// cannot produce PTX. Including the other sources from one file remains
// the best choice for small programs.
//
// The flags are:
//
//	-o file.go
//		Write the Go file here. The image goes beside it with a .bin
//		suffix. The default is the first source's name with
//		_precompiled.go in place of .cu.
//	-var name
//		Name of the Module variable (default "precompiled").
//	-pkg name
//		Package name (default $GOPACKAGE).
//	-arch list
//		Comma-separated machine code targets
//		(default "sm_80,sm_90,sm_100,sm_120").
//		Machine code for sm_XY runs on later devices sm_XZ.
//	-ptx target
//		PTX target, or "none" (default: compute_XY for the lowest -arch).
//	-I dir
//		Add an include directory. Repeatable.
//	-D name[=value]
//		Define a macro. Repeatable.
//	-std version
//		C++ standard, such as c++17.
//	-name expr
//		Record the lowered name of a name expression, such as
//		"scale<float>". Repeatable. Needs a single source.
//	-nolto
//		Link several sources without link-time optimization.
//	-tag tag
//		Embed the image only in builds with this build tag, in a second
//		file ending in _image.go. Other builds compile at run time, and
//		the image need not be committed.
//	-check
//		Report whether the generated file is out of date, without
//		writing it. Exits 1 if it is.
package main
