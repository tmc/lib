//go:build !darwin

package main

import (
	_ "embed"
	"fmt"

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
}

// PTX returns the exact JIT-compiled PTX assembly string.
func (gm *GPUModules) PTX() string {
	return gm.ptx
}

// CompileGPUModules compiles and loads the forward kernels onto dev.
func CompileGPUModules(dev *cuda.Device) (*GPUModules, error) {
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("compute capability: %w", err)
	}

	opts := []nvrtcx.Option{
		nvrtcx.WithName("llama_forward.cu"),
		nvrtcx.WithArch(major, minor),
	}
	mod, err := nvrtcx.Compile(ForwardKernelsCUDA, opts...)
	if err != nil {
		return nil, fmt.Errorf("compile forward kernels: %w", err)
	}

	loaded, err := mod.Load(dev)
	if err != nil {
		return nil, fmt.Errorf("load module: %w", err)
	}

	embedFn, _ := loaded.GetFunction("EmbeddingLookup")
	normFn, _ := loaded.GetFunction("RMSNorm")
	gemvFn, _ := loaded.GetFunction("LinearGEMV")
	ropeFn, _ := loaded.GetFunction("ApplyRoPE")
	storeKVFn, _ := loaded.GetFunction("StoreKVCache")
	swigluFn, _ := loaded.GetFunction("SwiGLU")
	addResFn, _ := loaded.GetFunction("AddResidual")
	fusedQKVFn, _ := loaded.GetFunction("FusedRMSNormQKV")
	fusedSwiGLUFn, _ := loaded.GetFunction("FusedSwiGLUDown")

	return &GPUModules{
		mod:           loaded,
		ptx:           mod.PTX(),
		embedFn:       embedFn,
		normFn:        normFn,
		gemvFn:        gemvFn,
		ropeFn:        ropeFn,
		storeKVFn:     storeKVFn,
		swigluFn:      swigluFn,
		addResFn:      addResFn,
		fusedQKVFn:    fusedQKVFn,
		fusedSwiGLUFn: fusedSwiGLUFn,
	}, nil
}

// Close unloads the compiled CUDA module.
func (gm *GPUModules) Close() error {
	if gm.mod != 0 {
		return gm.mod.Unload()
	}
	return nil
}
