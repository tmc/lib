//go:build darwin

package main

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/examples/internal/continuousbatch"
)

func TestScheduledAdapterCloseIsSerializedAndIdempotent(t *testing.T) {
	adapter := &scheduledEngineAdapter{}
	adapter.gpuMu.Lock()
	done := make(chan struct{})
	go func() {
		adapter.Close()
		adapter.Close()
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("Close completed while device owner was active")
	case <-time.After(20 * time.Millisecond):
	}
	adapter.gpuMu.Unlock()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Close did not complete after device owner released")
	}
	results := adapter.DecodeBatch(context.Background(), []*continuousbatch.State{{}})
	if len(results) != 1 || results[0].Err == nil || !strings.Contains(results[0].Err.Error(), "backend closed") {
		t.Fatalf("DecodeBatch after Close = %+v", results)
	}
}

func TestScheduledAdapterReleaseWaitsForDeviceOwner(t *testing.T) {
	adapter := &scheduledEngineAdapter{leases: map[int]uint64{0: 1}}
	state := &continuousbatch.State{KV: scheduledSequenceState{row: 0, lease: 1}}
	adapter.gpuMu.Lock()
	done := make(chan struct{})
	go func() {
		adapter.Release(state)
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("Release completed while device owner was active")
	case <-time.After(20 * time.Millisecond):
	}
	adapter.gpuMu.Unlock()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Release did not complete after device owner released")
	}
	if state.KV != nil {
		t.Fatal("Release left state attached")
	}
}
