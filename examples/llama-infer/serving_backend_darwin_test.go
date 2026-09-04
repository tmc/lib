//go:build darwin

package main

import "testing"

func scheduledBackend(t testing.TB, backend servingBackend) *scheduledEngineAdapter {
	t.Helper()
	if backend == nil {
		return nil
	}
	adapter, ok := backend.(*scheduledEngineAdapter)
	if !ok {
		t.Fatalf("serving backend = %T, want *scheduledEngineAdapter", backend)
	}
	return adapter
}
