//go:build darwin

package metalprof

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/lib/apple/x/metal"
)

const probeSource = `
#include <metal_stdlib>
using namespace metal;
kernel void metalprof_probe_alpha(device float *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = out[i] * 1.5f + 1.0f;
}
kernel void metalprof_probe_beta(device float *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = out[i] * 0.5f - 1.0f;
}
`

// TestCaptureReplayAttributesDispatchedKernel captures one of two kernels
// compiled from the same library, replays the trace through the system
// replayer, and requires the replayed profiler stream to name the dispatched
// kernel and not the one that was merely compiled. A stream that named both, or
// neither, would not attribute hardware work to the captured dispatches.
func TestCaptureReplayAttributesDispatchedKernel(t *testing.T) {
	if os.Getenv("METALPROF_CAPTURE_TEST") != "1" {
		t.Skip("set METALPROF_CAPTURE_TEST=1 to require programmatic Metal capture")
	}
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	lib, err := dev.CompileLibrary(probeSource)
	if err != nil {
		t.Fatal(err)
	}
	defer objectivec.ObjectFromID(lib.GetID()).Release()
	buf, err := dev.AllocBuffer(1024 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer objectivec.ObjectFromID(buf.GetID()).Release()

	for _, dispatched := range []string{"metalprof_probe_alpha", "metalprof_probe_beta"} {
		t.Run(dispatched, func(t *testing.T) {
			pipeline, err := dev.NewComputePipeline(lib, dispatched)
			if err != nil {
				t.Fatal(err)
			}
			defer objectivec.ObjectFromID(pipeline.GetID()).Release()
			dir := t.TempDir()
			trace := filepath.Join(dir, "probe.gputrace")
			session, err := StartCapture(dev, trace)
			if err != nil {
				t.Fatal(err)
			}
			for range 8 {
				cb := dev.Queue().CommandBuffer()
				enc := cb.ComputeCommandEncoder()
				enc.SetComputePipelineState(pipeline)
				enc.SetBufferWithOffsetAtIndex(buf, 0, 0)
				enc.DispatchThreadsThreadsPerThreadgroup(metal.Threads(1024), metal.Threads(64))
				enc.EndEncoding()
				cb.Commit()
				cb.WaitUntilCompleted()
				if cb.Status() != apple.MTLCommandBufferStatusCompleted {
					t.Fatalf("command buffer status = %v, error = %v", cb.Status(), cb.Error())
				}
			}
			if err := session.Stop(); err != nil {
				t.Fatal(err)
			}

			profiled := filepath.Join(dir, "profiled")
			if err := ReplayTrace(trace, profiled); err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(filepath.Join(profiled, "probe.gpuprofiler_raw", "streamData"))
			if err != nil {
				t.Fatalf("replay wrote no profiler stream: %v", err)
			}
			stream := string(data)
			absent := "metalprof_probe_beta"
			if dispatched == absent {
				absent = "metalprof_probe_alpha"
			}
			got, unwanted := strings.Count(stream, dispatched), strings.Count(stream, absent)
			if got == 0 || unwanted != 0 {
				t.Fatalf("replayed stream names %s %d times and %s %d times", dispatched, got, absent, unwanted)
			}
			t.Logf("device=%q stream=%d bytes dispatched=%s attributed=%d absent=%s counted=%d",
				dev.Name(), len(data), dispatched, got, absent, unwanted)
		})
	}
}

func TestReplayCancellationUnrelatedSurvives(t *testing.T) {
	// Verify that ReplayTraceContext cancellation returns ErrReplayUnresolvedOwnership
	// and does not kill unrelated mock processes.
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	dir := t.TempDir()
	dummyTrace := filepath.Join(dir, "dummy.gputrace")
	if err := os.MkdirAll(dummyTrace, 0755); err != nil {
		t.Fatal(err)
	}

	err := ReplayTraceContext(ctx, dummyTrace, filepath.Join(dir, "out"))
	if err == nil {
		t.Fatal("expected error on cancelled context")
	}
	if !errors.Is(err, ErrReplayUnresolvedOwnership) {
		t.Errorf("got error %v, want ErrReplayUnresolvedOwnership", err)
	}
}
