//go:build linux

package main

import (
	"context"
	"errors"
)

func getBenchmarkEngine(cfg Config) (*ModelEngine, error) { return GetOrCompileCudaEngine(cfg) }
func benchmarkBackend() string                            { return "cuda" }
func benchmarkPhysicalTelemetry() error {
	return errors.New("cuda execution does not expose exact dispatch and tier-byte counters")
}
func benchmarkDeviceID(eng *ModelEngine) (string, error) {
	if eng == nil || eng.dev == nil {
		return "", errors.New("nil CUDA engine")
	}
	return eng.dev.Name()
}
func resetBenchmarkCaches(*ModelEngine)           {}
func benchmarkPrefillDispatches(*ModelEngine) int { return 0 }
func benchmarkSynchronize(ctx context.Context, eng *ModelEngine) error {
	if eng == nil || eng.stream == nil {
		return errors.New("nil CUDA engine")
	}
	return eng.stream.Synchronize(ctx)
}
