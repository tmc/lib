//go:build !darwin

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// loadCUDAProgram selects the native artifact when available, otherwise PTX.
// A load failure is reported with its JIT diagnostics, without a format retry.
func loadCUDAProgram(dev *cuda.Device, program *nvrtcx.CompiledModule) (*cuda.Module, error) {
	kind := nvrtcx.OutputPTX
	if len(program.CUBIN()) != 0 {
		kind = nvrtcx.OutputCUBIN
	}
	mod, log, err := program.LoadOwnedOutput(dev, kind)
	if err != nil {
		return nil, fmt.Errorf("load CUDA output %d (JIT info %q, error %q): %w", kind, log.Info, log.Error, err)
	}
	return mod, nil
}
