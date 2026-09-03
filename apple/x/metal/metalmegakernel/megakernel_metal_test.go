//go:build darwin

package metalmegakernel_test

import (
	"testing"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmegakernel"
)

func TestGeneratedFusedFFNCompiles(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}
	defer dev.Close()
	msl, err := metalmegakernel.GenerateFusedFFNMSL(metalmegakernel.FusedFFNConfig{
		HiddenDim:       64,
		IntermediateDim: 128,
		RMSNormEpsilon:  1e-5,
	})
	if err != nil {
		t.Fatal(err)
	}
	lib, err := dev.CompileLibrary(msl)
	if err != nil {
		t.Fatalf("compile generated MSL: %v", err)
	}
	if _, err := dev.NewComputePipeline(lib, "MetalFusedFFN"); err != nil {
		t.Fatalf("create MetalFusedFFN pipeline: %v", err)
	}
}
