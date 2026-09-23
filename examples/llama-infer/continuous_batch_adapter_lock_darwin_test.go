//go:build darwin

package main

import (
	"testing"
	"time"
)

func TestDecodeBatchLocksGPUBeforeRows(t *testing.T) {
	var adapter scheduledEngineAdapter
	adapter.rowsMu.Lock()

	started := make(chan struct{})
	locked := make(chan struct{})
	go func() {
		close(started)
		adapter.lockDecode()
		close(locked)
		adapter.unlockDecode()
	}()
	<-started
	// Give lockDecode time to acquire gpuMu and block on rowsMu.
	time.Sleep(10 * time.Millisecond)
	if adapter.gpuMu.TryLock() {
		adapter.gpuMu.Unlock()
		adapter.rowsMu.Unlock()
		<-locked
		t.Fatal("decode acquired rows mutex before GPU mutex")
	}

	adapter.rowsMu.Unlock()
	select {
	case <-locked:
	case <-time.After(time.Second):
		t.Fatal("decode locks were not acquired after rows mutex was released")
	}
}
