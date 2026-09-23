//go:build darwin

package main

func newPlatformServingBackend(engine *ModelEngine, cfg Config, capacity int) (servingBackend, error) {
	return newScheduledEngineAdapter(engine, cfg, capacity)
}
