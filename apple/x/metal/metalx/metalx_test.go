package metalx_test

import (
	"testing"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalx"
)

const sampleKernel = `
#include <metal_stdlib>
using namespace metal;

kernel void VectorAdd(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    uint id               [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}
`

func TestCompile(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}

	pso, err := metalx.Compile(dev, sampleKernel, "VectorAdd")
	if err != nil {
		t.Fatalf("metalx.Compile failed: %v", err)
	}
	if pso.GetID() == 0 {
		t.Fatal("expected valid PSO handle")
	}

	// Test cache hit
	pso2, err := metalx.Compile(dev, sampleKernel, "VectorAdd")
	if err != nil {
		t.Fatalf("cached compile failed: %v", err)
	}
	if pso2.GetID() != pso.GetID() {
		t.Errorf("expected cached PSO ID %v, got %v", pso.GetID(), pso2.GetID())
	}
}
