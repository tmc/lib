//go:build darwin

package main

import (
	"context"
	"errors"

	"github.com/tmc/lib/apple/x/metal"
)

func getBenchmarkEngine(cfg Config) (*ModelEngine, error) { return GetOrCompileDarwinEngine(cfg) }
func benchmarkBackend() string                            { return "metal" }
func benchmarkObservedOutputSupported() bool              { return true }
func benchmarkPhysicalTelemetry() error {
	return nil
}
func benchmarkPeakMemorySupport(eng *ModelEngine) error {
	if eng == nil || eng.memory == nil {
		return errors.New("Metal engine allocation tracker is unavailable")
	}
	return nil
}
func benchmarkMemoryReset(ctx context.Context, eng *ModelEngine) error {
	if err := benchmarkSynchronize(ctx, eng); err != nil {
		return err
	}
	eng.memory.ResetPeak()
	return nil
}
func benchmarkPeakMemory(eng *ModelEngine) (uint64, error) {
	if eng == nil || eng.memory == nil {
		return 0, errors.New("Metal engine allocation tracker is unavailable")
	}
	_, peak := eng.memory.Snapshot()
	if peak == 0 {
		return 0, errors.New("Metal engine allocation peak is zero")
	}
	return peak, nil
}
func benchmarkDeviceID(eng *ModelEngine) (string, error) {
	if eng == nil || eng.dev == nil {
		return "", errors.New("nil Metal engine")
	}
	return eng.dev.Name(), nil
}
func resetBenchmarkCaches(eng *ModelEngine) { eng.prefixes.reset() }
func benchmarkPrefillDispatches(eng *ModelEngine) int {
	if eng.isF16 {
		active := 0
		for i := range eng.layersF16 {
			if eng.layersF16[i].InputNorm.Len() != 0 {
				active++
			}
		}
		return 4 + 12*active
	}
	active := 0
	for i := range eng.layers {
		if eng.layers[i].InputNorm.Len() != 0 {
			active++
		}
	}
	return 4 + 12*active
}
func benchmarkSynchronize(ctx context.Context, eng *ModelEngine) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if eng == nil || eng.dev == nil {
		return errors.New("nil Metal engine")
	}
	cb := eng.dev.Queue().CommandBuffer()
	cb.Commit()
	return metal.WaitCommandBuffer(ctx, cb)
}
