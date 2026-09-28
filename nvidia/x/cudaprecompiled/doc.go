// Package cudaprecompiled loads CUDA C++ device code that may have been
// compiled ahead of time by cudaprecompile.
//
// A [Module] carries the sources, their options, and optionally a
// precompiled image. [Module.Load] uses the image when it matches the
// sources and the device, and otherwise compiles the sources with NVRTC
// for the device. Precompiling is an optimization: it removes the NVRTC
// compile, and the NVRTC dependency, from startup, but a Module without an
// image works the same way.
//
// An image is stale when the embedded sources or options differ from the
// ones it was built from, such as after editing a kernel without rerunning
// go generate. Load never uses a stale image.
//
// cudaprecompile writes the Module declaration:
//
//	//go:generate go run github.com/tmc/lib/nvidia/x/cmd/cudaprecompile softmax.cu
//
//	mod, err := precompiled.Load(dev)
//	if err != nil {
//		return err
//	}
//	defer mod.Close()
//	fn, err := mod.Function("softmax")
package cudaprecompiled
