//go:build darwin

package main

import (
	"runtime"
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
	deadline := time.Now().Add(time.Second)
	for adapter.gpuMu.TryLock() {
		adapter.gpuMu.Unlock()
		if time.Now().After(deadline) {
			adapter.rowsMu.Unlock()
			t.Fatal("decode did not acquire GPU mutex before rows mutex")
		}
		runtime.Gosched()
	}

	adapter.rowsMu.Unlock()
	select {
	case <-locked:
	case <-time.After(time.Second):
		t.Fatal("decode locks were not acquired after rows mutex was released")
	}
}
