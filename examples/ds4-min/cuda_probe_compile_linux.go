//go:build linux

package main

import (
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

type cudaProbeCompileOperations struct {
	physical func(*cuda.Device) nvrtcx.Option
	output   func(nvrtcx.OutputKind) nvrtcx.Option
	compile  func(string, ...nvrtcx.Option) (*nvrtcx.CompiledModule, error)
	load     func(*nvrtcx.CompiledModule, *cuda.Device, nvrtcx.OutputKind) (driver.Module, error)
}

func cudaProbeCompiler() cudaProbeCompileOperations {
	return cudaProbeCompileOperations{
		physical: nvrtcx.WithDeviceArch,
		output:   nvrtcx.WithOutput,
		compile:  nvrtcx.Compile,
		load:     (*nvrtcx.CompiledModule).LoadOutput,
	}
}

func compileCUDAProbe(device *cuda.Device, name string, ops cudaProbeCompileOperations) (*nvrtcx.CompiledModule, error) {
	return ops.compile(cudaProbeSource, nvrtcx.WithName(name), ops.physical(device), ops.output(nvrtcx.OutputCUBIN))
}

func loadCUDAProbe(compiled *nvrtcx.CompiledModule, device *cuda.Device, ops cudaProbeCompileOperations) (driver.Module, error) {
	return ops.load(compiled, device, nvrtcx.OutputCUBIN)
}
