//go:build darwin

package metalicb

import (
	"testing"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
)

func TestGraphExecReplay(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}

	msl := `
#include <metal_stdlib>
using namespace metal;
kernel void IncrKernel(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    data[id] += 1.0f;
}
`

	lib, err := dev.CompileLibrary(msl)
	if err != nil {
		t.Fatalf("CompileLibrary failed: %v", err)
	}

	pso, err := dev.NewComputePipeline(lib, "IncrKernel")
	if err != nil {
		t.Fatalf("NewComputePipeline failed: %v", err)
	}

	buf, err := dev.AllocBuffer(256 * 4)
	if err != nil {
		t.Fatalf("AllocBuffer failed: %v", err)
	}

	graph := NewGraphExec(dev)
	graph.AddStage(
		pso,
		apple.MTLSize{Width: 256, Height: 1, Depth: 1},
		apple.MTLSize{Width: 32, Height: 1, Depth: 1},
		BufferBinding{Buffer: buf, Offset: 0, Index: 0},
	)

	// Replay 10 times with zero CPU re-encoding overhead
	for i := 0; i < 10; i++ {
		if err := graph.Launch(); err != nil {
			t.Fatalf("Launch failed at iteration %d: %v", i, err)
		}
	}

	if graph.Replays() != 10 {
		t.Errorf("expected 10 replays, got %d", graph.Replays())
	}
	t.Logf("GraphExec successfully executed %d GPU replays", graph.Replays())
}
