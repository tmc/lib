package main

import (
	_ "embed"
	"errors"
	"fmt"
	"io"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// ForwardKernelsCUDA is the embedded CUDA C++ source defining all transformer layer primitives.
//
//go:embed forward.cu
var ForwardKernelsCUDA string

// GPUModules holds compiled and loaded CUDA functions for model execution.
type GPUModules struct {
	mod           driver.Module
	ptx           string
	embedFn       driver.Function
	normFn        driver.Function
	gemvFn        driver.Function
	ropeFn        driver.Function
	storeKVFn     driver.Function
	swigluFn      driver.Function
	addResFn      driver.Function
	fusedQKVFn    driver.Function
	fusedSwiGLUFn driver.Function
	gemvBF16Fn    driver.Function
	convertBF16Fn driver.Function
	embedBF16Fn   driver.Function
}

// PTX returns the exact JIT-compiled PTX assembly string.
func (gm *GPUModules) PTX() string {
	return gm.ptx
}

// CompileGPUModules compiles and loads the forward kernels onto dev.
func CompileGPUModules(dev *cuda.Device) (*GPUModules, error) {
	return compileGPUModules(dev, ForwardKernelsCUDA+"\n"+forwardBF16CUDA)
}

// compileGPUModules compiles src, which must define every forward kernel,
// and loads it onto dev.
func compileGPUModules(dev *cuda.Device, src string) (*GPUModules, error) {
	opts := []nvrtcx.Option{
		nvrtcx.WithName("llama_forward.cu"),
		nvrtcx.WithDeviceArch(dev),
	}
	mod, err := nvrtcx.Compile(src, opts...)
	if err != nil {
		return nil, fmt.Errorf("compile forward kernels: %w", releaseErrorOwner(err))
	}

	loaded, err := mod.Load(dev)
	if err != nil {
		return nil, fmt.Errorf("load module: %w", releaseErrorOwner(err))
	}

	gm, err := lookupModules(loaded, loaded.Function, (*driver.Module).Close)
	if gm != nil {
		gm.ptx = mod.PTX()
	}
	return gm, err
}

// lookupModules looks up every forward kernel in loaded with lookup. If a
// lookup fails, it unloads loaded with unload and returns the lookup error;
// if the unload fails too, it returns both errors joined with a GPUModules
// that still owns loaded, for the caller to Close.
func lookupModules(loaded driver.Module, lookup func(string) (driver.Function, error), unload func(*driver.Module) error) (*GPUModules, error) {
	var err error
	fn := func(name string) driver.Function {
		if err != nil {
			return 0
		}
		f, lerr := lookup(name)
		if lerr != nil {
			err = fmt.Errorf("lookup kernel %s: %w", name, lerr)
		}
		return f
	}
	gm := &GPUModules{
		mod:           loaded,
		embedFn:       fn("EmbeddingLookup"),
		normFn:        fn("RMSNorm"),
		gemvFn:        fn("LinearGEMV"),
		ropeFn:        fn("ApplyRoPE"),
		storeKVFn:     fn("StoreKVCache"),
		swigluFn:      fn("SwiGLU"),
		addResFn:      fn("AddResidual"),
		fusedQKVFn:    fn("FusedRMSNormQKV"),
		fusedSwiGLUFn: fn("FusedSwiGLUDown"),
		gemvBF16Fn:    fn("LinearGEMVBF16"),
		convertBF16Fn: fn("WeightConvertProbeBF16"),
		embedBF16Fn:   fn("EmbeddingLookupBF16"),
	}
	if err != nil {
		if cerr := gm.close(unload); cerr != nil {
			return gm, errors.Join(err, cerr)
		}
		return nil, err
	}
	return gm, nil
}

// releaseErrorOwner releases what a failed call still holds. When nvrtcx
// cannot destroy a program after compiling or unload a module after a
// failed load, its error implements io.Closer, which retries the cleanup.
// releaseErrorOwner calls that Close and returns err, joined with the Close
// error if the retry fails, so the result still carries the Closer. A nil
// error or one without a Closer is returned unchanged, with nothing closed.
func releaseErrorOwner(err error) error {
	var c io.Closer
	if err == nil || !errors.As(err, &c) {
		return err
	}
	if cerr := c.Close(); cerr != nil {
		return errors.Join(err, fmt.Errorf("release resources held by failed call: %w", cerr))
	}
	return err
}

// Close unloads the compiled CUDA module. If unloading fails, gm keeps the
// module, so that Close can be called again. After a successful Close,
// calling it again does nothing.
func (gm *GPUModules) Close() error {
	return gm.close((*driver.Module).Close)
}

// close unloads gm's module with unload and forgets it only on success.
func (gm *GPUModules) close(unload func(*driver.Module) error) error {
	if gm.mod == 0 {
		return nil
	}
	if err := unload(&gm.mod); err != nil {
		return err
	}
	gm.mod = 0
	return nil
}
