//go:build !darwin

package main

import (
	_ "embed"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// ForwardKernelsCUDA is the embedded CUDA C++ source defining all transformer layer primitives.
//
//go:embed forward.cu
var ForwardKernelsCUDA string

// Kernels names the transformer primitives defined by ForwardKernelsCUDA.
type Kernels struct {
	Embed       driver.Function `cuda:"EmbeddingLookup"`
	Norm        driver.Function `cuda:"RMSNorm"`
	GEMV        driver.Function `cuda:"LinearGEMV"`
	RoPE        driver.Function `cuda:"ApplyRoPE"`
	StoreKV     driver.Function `cuda:"StoreKVCache"`
	SwiGLU      driver.Function `cuda:"SwiGLU"`
	AddResidual driver.Function `cuda:"AddResidual"`
	FusedQKV    driver.Function `cuda:"FusedRMSNormQKV"`
	FusedSwiGLU driver.Function `cuda:"FusedSwiGLUDown"`
}

// GPUModules holds compiled and loaded CUDA functions for model execution.
type GPUModules struct {
	Kernels
	mod driver.Module
	ptx string
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

	gm := &GPUModules{mod: loaded, ptx: mod.PTX()}
	if err := cudalaunch.Bind(loaded, &gm.Kernels); err != nil {
		_ = loaded.Unload()
		return nil, fmt.Errorf("bind forward kernels: %w", err)
	}
	return gm, nil
}

// Close unloads the compiled CUDA module.
func (gm *GPUModules) Close() error {
	if gm.mod != 0 {
		return gm.mod.Unload()
	}
	return nil
}
