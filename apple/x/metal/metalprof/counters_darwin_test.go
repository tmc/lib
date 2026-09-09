//go:build darwin

package metalprof

import (
	"encoding/binary"
	"errors"
	"os"
	"testing"

	"github.com/tmc/apple/foundation"
	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
)

func TestSampleBufferResolveRejectsUnavailableAndInvalidRanges(t *testing.T) {
	var unavailable SampleBuffer
	if _, err := unavailable.ResolveTimestamps(0, 1); !errors.Is(err, ErrTimestampCountersUnavailable) {
		t.Fatalf("unavailable ResolveTimestamps error = %v, want %v", err, ErrTimestampCountersUnavailable)
	}
}

func TestDecodeTimestampsValidation(t *testing.T) {
	// Wrong length (not count * 8)
	shortData := []byte{0x01, 0x02, 0x03}
	if _, err := decodeTimestamps(shortData, 0, 1); err == nil {
		t.Errorf("expected error on short data length")
	}

	// Contains MTLCounterErrorValue
	sentinelData := make([]byte, 16)
	// Sample 0: valid 1000
	binary.LittleEndian.PutUint64(sentinelData[0:8], 1000)
	// Sample 1: MTLCounterErrorValue
	binary.LittleEndian.PutUint64(sentinelData[8:16], MTLCounterErrorValue)
	if _, err := decodeTimestamps(sentinelData, 0, 2); err == nil {
		t.Errorf("expected error on MTLCounterErrorValue sentinel")
	}

	// Valid data
	validData := make([]byte, 16)
	binary.LittleEndian.PutUint64(validData[0:8], 1000)
	binary.LittleEndian.PutUint64(validData[8:16], 5000)
	ts, err := decodeTimestamps(validData, 0, 2)
	if err != nil {
		t.Fatalf("unexpected error on valid timestamps: %v", err)
	}
	if len(ts) != 2 || ts[0] != 1000 || ts[1] != 5000 {
		t.Fatalf("unexpected timestamps: %v", ts)
	}
}

func TestSampleBufferHardwareTimestamps(t *testing.T) {
	if os.Getenv("METALPROF_HARDWARE_TEST") != "1" {
		t.Skip("set METALPROF_HARDWARE_TEST=1 to require Metal timestamp counters")
	}
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	samples, err := NewSampleBuffer(dev, 2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := samples.ResolveTimestamps(1, 2); err == nil {
		t.Fatal("ResolveTimestamps accepted an out-of-range request")
	}

	rawDev := dev.Raw()
	cb := dev.Queue().CommandBuffer()

	if rawDev.SupportsCounterSampling(apple.MTLCounterSamplingPointAtStageBoundary) {
		lib, err := dev.CompileLibrary(`
#include <metal_stdlib>
using namespace metal;
kernel void test_k(device float *buf [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    buf[id] = buf[id] * 2.0f;
}
`)
		if err != nil {
			t.Fatal(err)
		}
		pso, err := dev.NewComputePipeline(lib, "test_k")
		if err != nil {
			t.Fatal(err)
		}
		buf, err := dev.AllocBuffer(1024 * 4)
		if err != nil {
			t.Fatal(err)
		}

		passDesc := apple.GetMTLComputePassDescriptorClass().Alloc().Init()
		att := passDesc.SampleBufferAttachments().ObjectAtIndexedSubscript(0)
		att.SetSampleBuffer(samples.Raw())
		att.SetStartOfEncoderSampleIndex(0)
		att.SetEndOfEncoderSampleIndex(1)

		enc := cb.ComputeCommandEncoderWithDescriptor(passDesc)
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(buf, 0, 0)
		enc.DispatchThreadsThreadsPerThreadgroup(apple.MTLSize{Width: 1024, Height: 1, Depth: 1}, apple.MTLSize{Width: 256, Height: 1, Depth: 1})
		enc.EndEncoding()
	} else if rawDev.SupportsCounterSampling(apple.MTLCounterSamplingPointAtBlitBoundary) {
		buf, err := dev.AllocBuffer(4 << 20)
		if err != nil {
			t.Fatal(err)
		}
		enc := cb.BlitCommandEncoder()
		enc.SampleCountersInBufferAtSampleIndexWithBarrier(samples.Raw(), 0, true)
		enc.FillBufferRangeValue(buf, foundation.NSRange{Location: 0, Length: 4 << 20}, 0x5a)
		enc.SampleCountersInBufferAtSampleIndexWithBarrier(samples.Raw(), 1, true)
		enc.EndEncoding()
	} else {
		t.Skip("device supports neither stage nor blit counter sampling")
	}

	cb.Commit()
	cb.WaitUntilCompleted()
	if cb.Status() != apple.MTLCommandBufferStatusCompleted {
		t.Fatalf("command buffer status = %v, error = %v", cb.Status(), cb.Error())
	}
	got, err := samples.ResolveTimestamps(0, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("timestamps len = %d, want 2", len(got))
	}
	if got[0] == 0 && got[1] == 0 {
		t.Fatalf("expected non-zero hardware timestamps; got [0, 0]")
	}
	if got[1] <= got[0] {
		t.Fatalf("expected positive timestamp delta; start=%d, end=%d", got[0], got[1])
	}
	t.Logf("device=%q timestamps=%v delta=%d ns", dev.Name(), got, got[1]-got[0])
}
