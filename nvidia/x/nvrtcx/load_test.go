package nvrtcx

import (
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestLoadInvalidCUBIN(t *testing.T) {
	if os.Getenv("CUDA_COMPILER_TEST") != "1" {
		t.Skip("CUDA_COMPILER_TEST=1 required")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	// NUL-terminated non-ELF bytes make an invalid input regardless of whether
	// the driver attempts to interpret it as a binary image or PTX text.
	cm := &CompiledModule{output: OutputCUBIN, cubin: []byte("invalid cubin\x00")}
	module, err := cm.Load(dev)
	if err == nil {
		_ = dev.WithContext(module.Unload)
		t.Fatal("accepted invalid cubin")
	}
	if !strings.Contains(err.Error(), "load cubin:") {
		t.Fatalf("lost cubin load diagnostic: %v", err)
	}
}
