//go:build linux && !tpu

package main

import (
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"os"
	"testing"
)

func TestCUDAForwardModuleLifetime(t *testing.T) {
	if os.Getenv("LLAMA_CUDA_PREFILL_TEST") != "1" {
		t.Skip("set LLAMA_CUDA_PREFILL_TEST=1 for CUDA validation")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	modules, err := CompileGPUModules(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer modules.Close()
	kernel := modules.BatchArgmax
	if kernel == nil {
		t.Fatal("batch argmax was not bound")
	}
	if err := modules.Close(); err != nil {
		t.Fatal(err)
	}
	if err := modules.Close(); err != nil {
		t.Fatal(err)
	}
	var count cudalaunch.Counter
	if err := cudalaunch.OwnedKernel(kernel).Count(&count).Launch(); err == nil {
		t.Fatal("stale forward kernel launched")
	}
	if count.Value() != 0 {
		t.Fatal("stale launch counted")
	}
}
