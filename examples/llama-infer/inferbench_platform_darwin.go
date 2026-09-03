//go:build darwin

package main

import (
	"context"
	"errors"

	"github.com/tmc/lib/apple/x/metal"
)

func getBenchmarkEngine(cfg Config) (*ModelEngine, error) { return GetOrCompileDarwinEngine(cfg) }
func benchmarkBackend() string                            { return "metal" }
func benchmarkPhysicalTelemetry() error {
	return nil
}
func benchmarkDeviceID(eng *ModelEngine) (string, error) {
	if eng == nil || eng.dev == nil {
		return "", errors.New("nil Metal engine")
	}
	return eng.dev.Name(), nil
}
func resetBenchmarkCaches(eng *ModelEngine) { eng.prefixes = newPrefixCache(64<<20, 8) }
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
