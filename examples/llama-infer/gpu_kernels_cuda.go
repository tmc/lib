//go:build !darwin

package main

import (
	_ "embed"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// ForwardKernelsCUDA is the embedded CUDA C++ source defining all transformer layer primitives.
//
//go:embed forward.cu
var ForwardKernelsCUDA string

// Kernels names the transformer primitives defined by ForwardKernelsCUDA.
type Kernels struct {
	Embed            *cuda.Kernel `cuda:"EmbeddingLookup" weights:"yes"`
	Norm             *cuda.Kernel `cuda:"RMSNorm" weights:"yes"`
	GEMV             *cuda.Kernel `cuda:"LinearGEMV" weights:"yes"`
	RoPE             *cuda.Kernel `cuda:"ApplyRoPE"`
	StoreKV          *cuda.Kernel `cuda:"StoreKVCache"`
	SwiGLU           *cuda.Kernel `cuda:"SwiGLU"`
	AddResidual      *cuda.Kernel `cuda:"AddResidual"`
	FusedQKV         *cuda.Kernel `cuda:"FusedRMSNormQKV" weights:"yes"`
	FusedSwiGLU      *cuda.Kernel `cuda:"FusedSwiGLUDown" weights:"yes"`
	BatchEmbed       *cuda.Kernel `cuda:"BatchEmbeddingLookup" weights:"yes"`
	BatchNorm        *cuda.Kernel `cuda:"BatchRMSNorm" weights:"yes"`
	BatchGEMV        *cuda.Kernel `cuda:"BatchLinearGEMV" weights:"yes"`
	BatchSwiGLU      *cuda.Kernel `cuda:"BatchSwiGLU"`
	BatchAddBias     *cuda.Kernel `cuda:"BatchAddBias" weights:"yes"`
	BatchAddResidual *cuda.Kernel `cuda:"BatchAddResidual"`
	BatchFusedQKV    *cuda.Kernel `cuda:"BatchFusedRMSNormQKV" weights:"yes"`
	BatchFusedSwiGLU *cuda.Kernel `cuda:"BatchFusedSwiGLUDown" weights:"yes"`
	BatchArgmax      *cuda.Kernel `cuda:"BatchArgmax"`
}

// GPUModules holds compiled and loaded CUDA functions for model execution.
type GPUModules struct {
	Kernels
	mod   *cuda.Module
	ptx   string
	dtype WeightDType
}

// WeightDType returns the weight dtype these kernels were bound for.
//
// This is the dtype that was actually dispatched, read back from the bound
// module rather than from the request, so a caller reporting f16 has evidence
// that f16 entry points are what will run.
func (gm *GPUModules) WeightDType() WeightDType { return gm.dtype }

// KernelReport names the bound entry points and the selected dtype.
func (gm *GPUModules) KernelReport() string { return weightKernelReport(gm.dtype) }

// PTX returns the exact JIT-compiled PTX assembly string.
func (gm *GPUModules) PTX() string {
	return gm.ptx
}

// CompileGPUModules compiles and loads the forward kernels onto dev.
func CompileGPUModules(dev *cuda.Device) (*GPUModules, error) {
	return CompileGPUModulesAs(dev, WeightF32)
}

// CompileGPUModulesAs compiles the forward kernels and binds the entry points
// that read weights in dtype d.
//
// The float32 entry points keep their original names, so d == WeightF32 binds
// exactly what CompileGPUModules always bound. A half dtype binds the suffixed
// entry points, and a module missing one is an error: there is no fallback to
// the float32 kernel, because a silent fallback would report a half run that
// was really a float32 run.
func CompileGPUModulesAs(dev *cuda.Device, d WeightDType) (*GPUModules, error) {
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

	gm := &GPUModules{mod: loaded, ptx: mod.PTX(), dtype: d}
	if err := bindWeightKernels(loaded, d, &gm.Kernels); err != nil {
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
