package main

import (
	"sync"
	"testing"
)

func TestPhysicalWaveLedgerSharedWaveAndSnapshots(t *testing.T) {
	var l physicalWaveLedger
	before := l.Snapshot()
	if err := l.Record("decode-chain", []string{"b", "a"}, 7, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := l.Record("prefix-publish", []string{"a"}, 0, map[string]uint64{"metal_device": 32}, map[string]uint64{"metal_prefix": 32}); err != nil {
		t.Fatal(err)
	}
	waves, err := l.Since(before)
	if err != nil {
		t.Fatal(err)
	}
	if len(waves) != 2 || waves[0].Dispatches != 7 || len(waves[0].Requests) != 2 || waves[0].Requests[0] != "a" {
		t.Fatalf("waves = %+v", waves)
	}
	after := l.Snapshot()
	if waves, err := l.Since(after); err != nil || len(waves) != 0 {
		t.Fatalf("later delta = %+v, %v", waves, err)
	}
}

func TestPhysicalWaveLedgerConcurrentRecords(t *testing.T) {
	var l physicalWaveLedger
	before := l.Snapshot()
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := l.Record("wave", []string{"request"}, 1, nil, nil); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	waves, err := l.Since(before)
	if err != nil || len(waves) != 32 {
		t.Fatalf("got %d waves, %v", len(waves), err)
	}
	for i, wave := range waves {
		if wave.Sequence != uint64(i+1) {
			t.Fatalf("sequence %d = %d", i, wave.Sequence)
		}
	}
}

func TestPhysicalWaveLedgerRejectsResetSnapshot(t *testing.T) {
	var old, replacement physicalWaveLedger
	if _, err := replacement.Since(old.Snapshot()); err == nil {
		t.Fatal("accepted snapshot from replaced ledger")
	}
}
