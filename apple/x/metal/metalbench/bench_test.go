//go:build darwin

package metalbench

import (
	"testing"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
)

func TestAutoTuneEGraphKernels(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}

	h := New(dev)

	candidates := []Candidate{
		{
			Name: "KernelTile8",
			MSL: `
#include <metal_stdlib>
using namespace metal;
kernel void KernelTile8(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    out[id] = float(id) * 2.0f;
}
`,
		},
		{
			Name: "KernelTile16",
			MSL: `
#include <metal_stdlib>
using namespace metal;
kernel void KernelTile16(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    out[id] = float(id) * 2.0f + 1.0f;
}
`,
		},
	}

	buf, err := dev.AllocBuffer(1024 * 4)
	if err != nil {
		t.Fatalf("AllocBuffer failed: %v", err)
	}

	res, err := h.AutoTune(candidates, 5, func(pso apple.MTLComputePipelineState, cb apple.MTLCommandBuffer) {
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(buf, 0, 0)
		enc.DispatchThreadsThreadsPerThreadgroup(
			apple.MTLSize{Width: 1024, Height: 1, Depth: 1},
			apple.MTLSize{Width: 64, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
	})

	if err != nil {
		t.Fatalf("AutoTune failed: %v", err)
	}

	t.Logf("Empirical E-Graph AutoTuner selected best kernel: %s (Latency: %v, Throughput: %.0f ops/s)",
		res.KernelName, res.Duration, res.OpsPerSec)
}
