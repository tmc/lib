//go:build !darwin

package main

import (
	_ "embed"
	"fmt"

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
	Embed            *cuda.Kernel `cuda:"EmbeddingLookup"`
	Norm             *cuda.Kernel `cuda:"RMSNorm"`
	GEMV             *cuda.Kernel `cuda:"LinearGEMV"`
	RoPE             *cuda.Kernel `cuda:"ApplyRoPE"`
	StoreKV          *cuda.Kernel `cuda:"StoreKVCache"`
	SwiGLU           *cuda.Kernel `cuda:"SwiGLU"`
	AddResidual      *cuda.Kernel `cuda:"AddResidual"`
	FusedQKV         *cuda.Kernel `cuda:"FusedRMSNormQKV"`
	FusedSwiGLU      *cuda.Kernel `cuda:"FusedSwiGLUDown"`
	BatchEmbed       *cuda.Kernel `cuda:"BatchEmbeddingLookup"`
	BatchNorm        *cuda.Kernel `cuda:"BatchRMSNorm"`
	BatchGEMV        *cuda.Kernel `cuda:"BatchLinearGEMV"`
	BatchSwiGLU      *cuda.Kernel `cuda:"BatchSwiGLU"`
	BatchAddBias     *cuda.Kernel `cuda:"BatchAddBias"`
	BatchAddResidual *cuda.Kernel `cuda:"BatchAddResidual"`
	BatchFusedQKV    *cuda.Kernel `cuda:"BatchFusedRMSNormQKV"`
	BatchFusedSwiGLU *cuda.Kernel `cuda:"BatchFusedSwiGLUDown"`
	BatchArgmax      *cuda.Kernel `cuda:"BatchArgmax"`
}

// GPUModules holds compiled and loaded CUDA functions for model execution.
type GPUModules struct {
	Kernels
	mod *cuda.Module
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

	loaded, err := loadCUDAProgram(dev, mod)
	if err != nil {
		return nil, fmt.Errorf("load module: %w", err)
	}

	gm := &GPUModules{mod: loaded, ptx: mod.PTX()}
	if err := cudalaunch.BindOwned(loaded, &gm.Kernels); err != nil {
		return nil, finishCUDAInit(fmt.Errorf("bind forward kernels: %w", err), loaded.Close)
	}
	return gm, nil
}

// Close unloads the compiled CUDA module.
func (gm *GPUModules) Close() error {
	if gm != nil && gm.mod != nil {
		return gm.mod.Close()
	}
	return nil
}
