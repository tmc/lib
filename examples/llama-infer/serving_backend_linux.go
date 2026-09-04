//go:build linux

package main

func newPlatformServingBackend(*ModelEngine, Config, int) (servingBackend, error) {
	return nil, errServingBackendUnavailable
}
