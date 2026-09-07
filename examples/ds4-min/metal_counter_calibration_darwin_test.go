//go:build darwin

package main

import (
	"os"
	"testing"
	"time"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalprof"
)

// A Metal phase profile comparable to the CUDA one needs GPU timestamps
// sampled between kernel dispatches, not host markers around encode calls.
// Three things about that path are device properties, not source facts:
// whether this device permits sampling at a dispatch boundary, how many
// samples one buffer holds, and what a resolved tick is worth in
// nanoseconds. This measures all three without loading a model, so the
// answers are in hand before any full-model instrument is built.
//
// It measures the device; it does not gate anything. Nothing here touches
// the production path.

const counterCalibrationSource = `
#include <metal_stdlib>
using namespace metal;
kernel void spin(device float *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
	float v = out[i];
	for (int k = 0; k < 64; k++) {
		v = fma(v, 1.0000001f, 1.0f);
	}
	out[i] = v;
}
`

func TestMetalCounterCalibration(t *testing.T) {
	if os.Getenv("DS4_METAL_COUNTER_CALIBRATION") == "" {
		t.Skip("DS4_METAL_COUNTER_CALIBRATION is not set")
	}
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	t.Logf("device %q", dev.Name())

	for _, point := range []struct {
		name string
		at   apple.MTLCounterSamplingPoint
	}{
		{"dispatch", apple.MTLCounterSamplingPointAtDispatchBoundary},
		{"blit", apple.MTLCounterSamplingPointAtBlitBoundary},
		{"stage", apple.MTLCounterSamplingPointAtStageBoundary},
		{"draw", apple.MTLCounterSamplingPointAtDrawBoundary},
	} {
		t.Logf("supports sampling at %s boundary: %v", point.name, dev.Raw().SupportsCounterSampling(point.at))
	}
	dispatchBoundary := dev.Raw().SupportsCounterSampling(apple.MTLCounterSamplingPointAtDispatchBoundary)
	stageBoundary := dev.Raw().SupportsCounterSampling(apple.MTLCounterSamplingPointAtStageBoundary)
	if !dispatchBoundary && !stageBoundary {
		t.Fatal("device permits no counter sampling this profile could use")
	}

	// Capacity. The layer path would need two samples per phase boundary
	// across 43 layers, so report the largest buffer this device grants
	// rather than assuming one is enough.
	largest := 0
	for _, capacity := range []int{2, 64, 256, 1024, 4096, 16384, 65536} {
		sb, err := metalprof.NewSampleBuffer(dev, capacity)
		if err != nil {
			t.Logf("capacity %d refused: %v", capacity, err)
			break
		}
		if got := sb.SampleCount(); got != capacity {
			t.Logf("capacity %d granted %d", capacity, got)
		}
		largest = capacity
	}
	t.Logf("largest accepted sample buffer capacity: %d", largest)

	// Tick unit. SampleTimestampsGpuTimestamp correlates the two clocks;
	// two correlations around a known host interval say what a GPU tick is
	// worth. Without this a resolved delta is a number with no unit.
	var cpu0, gpu0, cpu1, gpu1 apple.MTLTimestamp
	dev.Raw().SampleTimestampsGpuTimestamp(&cpu0, &gpu0)
	host := time.Now()
	time.Sleep(100 * time.Millisecond)
	elapsed := time.Since(host)
	dev.Raw().SampleTimestampsGpuTimestamp(&cpu1, &gpu1)
	if gpu1 <= gpu0 {
		t.Fatalf("gpu timestamps did not advance: %d then %d", gpu0, gpu1)
	}
	nsPerTick := float64(elapsed.Nanoseconds()) / float64(gpu1-gpu0)
	t.Logf("gpu ticks %d over host %v: %.6f ns per tick", gpu1-gpu0, elapsed, nsPerTick)

	// Resolution inside a compute encoder, which is the case the layer
	// path needs and the case the existing metalprof hardware test does
	// not cover: it samples at a blit boundary.
	lib, err := dev.CompileLibrary(counterCalibrationSource)
	if err != nil {
		t.Fatal(err)
	}
	pipeline, err := dev.NewComputePipeline(lib, "spin")
	if err != nil {
		t.Fatal(err)
	}
	const elems = 1 << 20
	buf, err := dev.AllocBuffer(elems * 4)
	if err != nil {
		t.Fatal(err)
	}
	// Stage-boundary sampling attaches two indices to a whole compute
	// pass, so it times an encoder, not a dispatch. Phase granularity by
	// this route means one encoder per phase.
	if stageBoundary {
		samples, err := metalprof.NewSampleBuffer(dev, 2)
		if err != nil {
			t.Fatal(err)
		}
		desc := apple.GetMTLComputePassDescriptorClass().Alloc().Init()
		attach := desc.SampleBufferAttachments().ObjectAtIndexedSubscript(0)
		attach.SetSampleBuffer(samples.Raw())
		attach.SetStartOfEncoderSampleIndex(0)
		attach.SetEndOfEncoderSampleIndex(1)
		cb := dev.Queue().CommandBuffer()
		enc := cb.ComputeCommandEncoderWithDescriptor(desc)
		enc.SetComputePipelineState(pipeline)
		enc.SetBufferWithOffsetAtIndex(buf, 0, 0)
		for range 8 {
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: elems / 256, Height: 1, Depth: 1},
				apple.MTLSize{Width: 256, Height: 1, Depth: 1})
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		if status := cb.Status(); status != apple.MTLCommandBufferStatusCompleted {
			t.Fatalf("stage boundary: command buffer status %v: %v", status, cb.Error())
		}
		got, err := samples.ResolveTimestamps(0, 2)
		if err != nil {
			t.Fatalf("stage boundary: %v", err)
		}
		t.Logf("stage boundary: samples %v, pass span %.0f ns, host GPUStartTime..GPUEndTime %v",
			got, float64(got[1]-got[0])*nsPerTick,
			time.Duration((cb.GPUEndTime()-cb.GPUStartTime())*1e9)*time.Nanosecond)
	}

	// Dispatch-boundary sampling is what a nine-phase profile inside one
	// encoder would need. Apple silicon may refuse it; record the answer
	// rather than assuming it.
	if !dispatchBoundary {
		t.Log("dispatch boundary unsupported: per-dispatch GPU timestamps inside one encoder are unavailable on this device")
		return
	}
	for _, barrier := range []bool{true, false} {
		const dispatches = 8
		samples, err := metalprof.NewSampleBuffer(dev, dispatches+1)
		if err != nil {
			t.Fatal(err)
		}
		cb := dev.Queue().CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pipeline)
		enc.SetBufferWithOffsetAtIndex(buf, 0, 0)
		enc.SampleCountersInBufferAtSampleIndexWithBarrier(samples.Raw(), 0, barrier)
		for i := range dispatches {
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: elems / 256, Height: 1, Depth: 1},
				apple.MTLSize{Width: 256, Height: 1, Depth: 1})
			enc.SampleCountersInBufferAtSampleIndexWithBarrier(samples.Raw(), uint(i+1), barrier)
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		if status := cb.Status(); status != apple.MTLCommandBufferStatusCompleted {
			t.Fatalf("barrier=%v: command buffer status %v: %v", barrier, status, cb.Error())
		}
		got, err := samples.ResolveTimestamps(0, dispatches+1)
		if err != nil {
			t.Fatalf("barrier=%v: %v", barrier, err)
		}
		var zero, backward int
		for i, v := range got {
			if v == 0 {
				zero++
			}
			if i > 0 && v < got[i-1] {
				backward++
			}
		}
		spans := make([]float64, 0, len(got)-1)
		for i := 1; i < len(got); i++ {
			spans = append(spans, float64(got[i]-got[i-1])*nsPerTick)
		}
		t.Logf("barrier=%v: resolved %d of %d samples, %d zero, %d backward, dispatch spans (ns) %.0f",
			barrier, len(got), dispatches+1, zero, backward, spans)
	}
}
