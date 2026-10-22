//go:build linux

package main

import (
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

// TestCUDAQ80BatchRegisters reports compiled resource usage for width trials.
// Local memory is reported separately; its size does not count spill traffic.
func TestCUDAQ80BatchRegisters(t *testing.T) {
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock is required")
	}
	r := cudaSynthFoundation(t)
	if err := r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		for _, k := range []struct {
			name  string
			fn    driver.Function
			width int
		}{
			{"ds4_q8_0_matvec_q8_1_batch", r.q81MatvecBatchFn, cudaQ81MaxBatch},
			{"ds4_q8_0_matvec_quantized_batch", r.q80QuantizedBatchFn, cudaQ80MaxBatch},
			{"ds4_q8_0_matvec_quantized_small", r.q80QuantizedSmallFn, cudaQ80SmallMaxBatch},
			{"ds4_q8_0_matvec_quantized_blockdiag_batch", r.q80BlockDiagBatchFn, cudaQ80BlockDiagMaxBatch},
		} {
			regs, err := k.fn.Attribute(driver.FuncAttributeNumRegs)
			if err != nil {
				return err
			}
			local, err := k.fn.Attribute(driver.FuncAttributeLocalSizeBytes)
			if err != nil {
				return err
			}
			t.Logf("kernel %s: tile=%d registers=%d local_bytes=%d", k.name, k.width, regs, local)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
