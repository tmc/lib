//go:build darwin

package metalprof

import (
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

func TestSampleBufferHardwareTimestamps(t *testing.T) {
	if os.Getenv("METALPROF_HARDWARE_TEST") != "1" {
		t.Skip("set METALPROF_HARDWARE_TEST=1 to require Metal timestamp counters")
	}
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	samples, err := NewSampleBuffer(dev, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !dev.Raw().SupportsCounterSampling(apple.MTLCounterSamplingPointAtBlitBoundary) {
		t.Fatal("device does not support counter sampling at blit boundaries")
	}
	if _, err := samples.ResolveTimestamps(1, 2); err == nil {
		t.Fatal("ResolveTimestamps accepted an out-of-range request")
	}
	buf, err := dev.AllocBuffer(4 << 20)
	if err != nil {
		t.Fatal(err)
	}
	cb := dev.Queue().CommandBuffer()
	enc := cb.BlitCommandEncoder()
	enc.SampleCountersInBufferAtSampleIndexWithBarrier(samples.Raw(), 0, true)
	enc.FillBufferRangeValue(buf, foundation.NSRange{Location: 0, Length: 4 << 20}, 0x5a)
	enc.SampleCountersInBufferAtSampleIndexWithBarrier(samples.Raw(), 1, true)
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	if cb.Status() != apple.MTLCommandBufferStatusCompleted {
		t.Fatalf("command buffer status = %v, error = %v", cb.Status(), cb.Error())
	}
	got, err := samples.ResolveTimestamps(0, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] == 0 || got[1] <= got[0] {
		t.Fatalf("timestamps = %v, want two increasing nonzero hardware samples", got)
	}
	t.Logf("device=%q timestamps=%v duration_ns=%d", dev.Name(), got, got[1]-got[0])
}
