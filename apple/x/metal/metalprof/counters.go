//go:build darwin

package metalprof

import (
	"encoding/binary"
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/apple/foundation"
	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
)

// ErrTimestampCountersUnavailable reports that a device cannot create a
// timestamp counter sample buffer. Callers must not substitute host time for
// GPU counter values.
var ErrTimestampCountersUnavailable = errors.New("metalprof: timestamp counters unavailable")

// SampleBuffer wraps an MTLCounterSampleBuffer for hardware timestamps.
type SampleBuffer struct {
	raw   apple.MTLCounterSampleBuffer
	count int
}

// NewSampleBuffer allocates a GPU counter sample buffer on the device for recording timestamps.
func NewSampleBuffer(dev *metal.Device, capacity int) (*SampleBuffer, error) {
	if dev == nil || dev.Raw().GetID() == 0 {
		return nil, fmt.Errorf("metalprof: nil or uninitialized Metal device")
	}
	if capacity <= 0 {
		capacity = 256
	}

	// Find the timestamp counter set
	var timestampSet apple.MTLCounterSet
	for _, cs := range dev.Raw().CounterSets() {
		if cs == nil || cs.GetID() == 0 {
			continue
		}
		set := apple.MTLCounterSetObjectFromID(cs.GetID())
		if set.Name() == "MTLCommonCounterSetTimestamp" || set.Name() == "timestamp" {
			timestampSet = set
			break
		}
	}
	if timestampSet == nil || timestampSet.GetID() == 0 {
		return nil, ErrTimestampCountersUnavailable
	}

	desc := apple.GetMTLCounterSampleBufferDescriptorClass().Alloc().Init()
	desc.SetSampleCount(uint(capacity))
	desc.SetStorageMode(apple.MTLStorageModeShared)
	desc.SetLabel("metalprof_timestamps")
	desc.SetCounterSet(timestampSet)

	sb, err := dev.Raw().NewCounterSampleBufferWithDescriptorError(desc)
	if err != nil {
		return nil, fmt.Errorf("metalprof: create timestamp counter sample buffer: %w", err)
	}
	if sb.GetID() == 0 {
		return nil, ErrTimestampCountersUnavailable
	}

	return &SampleBuffer{
		raw:   sb,
		count: capacity,
	}, nil
}

// Raw returns the underlying MTLCounterSampleBuffer.
func (sb *SampleBuffer) Raw() apple.MTLCounterSampleBuffer {
	return sb.raw
}

// SampleCount returns the capacity of the sample buffer.
func (sb *SampleBuffer) SampleCount() int {
	return sb.count
}

// ResolveTimestamps extracts resolved GPU timestamp values from the sample buffer.
func (sb *SampleBuffer) ResolveTimestamps(start, count int) ([]uint64, error) {
	if sb == nil || sb.raw == nil || sb.raw.GetID() == 0 {
		return nil, ErrTimestampCountersUnavailable
	}
	if start < 0 || count <= 0 || start > sb.count-count {
		return nil, fmt.Errorf("metalprof: invalid counter sample range [%d, %d] for capacity %d", start, count, sb.count)
	}

	nsData := sb.raw.ResolveCounterRange(foundation.NSRange{
		Location: uint(start),
		Length:   uint(count),
	})
	if nsData.GetID() == 0 || nsData.Length() == 0 {
		return nil, fmt.Errorf("metalprof: failed to resolve counter sample range [%d, %d]", start, count)
	}

	dataLen := int(nsData.Length())
	numSamples := dataLen / 8
	if numSamples > count {
		numSamples = count
	}

	rawBytes := unsafe.Slice((*byte)(nsData.Bytes()), dataLen)
	out := make([]uint64, numSamples)
	for i := 0; i < numSamples; i++ {
		out[i] = binary.LittleEndian.Uint64(rawBytes[i*8:])
	}
	return out, nil
}
