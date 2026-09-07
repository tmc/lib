//go:build linux && !tpu

package main

import (
	"context"
	"errors"

	"github.com/tmc/lib/nvidia/x/nvtx"
)

func getBenchmarkEngine(cfg Config) (*ModelEngine, error) { return GetOrCompileCudaEngine(cfg) }
func benchmarkBackend() string                            { return "cuda" }
func benchmarkObservedOutputSupported() bool              { return true }
func benchmarkPhysicalTelemetry() error {
	return nil
}
func benchmarkPeakMemorySupport(eng *ModelEngine) error {
	if eng == nil || eng.memory == nil {
		return errors.New("CUDA engine allocation tracker is unavailable")
	}
	return nil
}
func benchmarkMemoryReset(ctx context.Context, eng *ModelEngine) error {
	if err := benchmarkSynchronize(ctx, eng); err != nil {
		return err
	}
	eng.memory.Reset()
	return nil
}
func benchmarkPeakMemory(eng *ModelEngine) (uint64, error) {
	if eng == nil || eng.memory == nil {
		return 0, errors.New("CUDA engine allocation tracker is unavailable")
	}
	_, peak := eng.memory.Snapshot()
	if peak == 0 {
		return 0, errors.New("CUDA engine allocation peak is zero")
	}
	return peak, nil
}
func benchmarkDeviceID(eng *ModelEngine) (string, error) {
	if eng == nil || eng.dev == nil {
		return "", errors.New("nil CUDA engine")
	}
	return eng.dev.Name()
}
func resetBenchmarkCaches(*ModelEngine)           {}
func benchmarkPrefillDispatches(*ModelEngine) int { return 0 }
func benchmarkPhaseRange(name string) func() {
	if name == "" {
		return func() {}
	}
	return nvtx.ScopedRange(name)
}
func benchmarkSynchronize(ctx context.Context, eng *ModelEngine) error {
	if eng == nil || eng.stream == nil {
		return errors.New("nil CUDA engine")
	}
	return eng.stream.Synchronize(ctx)
}
