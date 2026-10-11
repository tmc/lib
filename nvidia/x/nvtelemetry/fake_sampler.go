package nvtelemetry

import (
	"context"
	"sync"
)

// FakeSampler provides deterministic scripted telemetry for offline testing.
type FakeSampler struct {
	mu       sync.Mutex
	devices  []DeviceMetadata
	scripted [][]Reading
	index    int
	fallback func(ctx context.Context) ([]Reading, error)
}

// NewFakeSampler creates a fake sampler with the provided device metadata.
func NewFakeSampler(devices ...DeviceMetadata) *FakeSampler {
	return &FakeSampler{
		devices: devices,
	}
}

// Devices returns the configured fake devices.
func (f *FakeSampler) Devices() []DeviceMetadata {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]DeviceMetadata, len(f.devices))
	copy(out, f.devices)
	return out
}

// Enqueue appends batches of scripted readings to be returned by successive
// calls to Sample.
func (f *FakeSampler) Enqueue(batches ...[]Reading) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.scripted = append(f.scripted, batches...)
}

// SetFallback configures a handler invoked when scripted batches are exhausted.
func (f *FakeSampler) SetFallback(fn func(ctx context.Context) ([]Reading, error)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.fallback = fn
}

// Sample returns the next scripted batch of readings.
func (f *FakeSampler) Sample(ctx context.Context) ([]Reading, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	if f.index < len(f.scripted) {
		batch := f.scripted[f.index]
		f.index++
		return batch, nil
	}

	if f.fallback != nil {
		return f.fallback(ctx)
	}

	// Default fallback: return zero readings for configured devices
	var readings []Reading
	for _, d := range f.devices {
		readings = append(readings, Reading{
			Name:       MetricGPUUtilization,
			Unit:       UnitPercent,
			Backend:    "fake",
			DeviceUUID: d.UUID,
			Evidence:   EvidenceMeasured,
			Value:      0,
		})
	}
	return readings, nil
}
