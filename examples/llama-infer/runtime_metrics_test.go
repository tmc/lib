package main

import (
	"testing"
	"time"
)

func TestRuntimeMetrics(t *testing.T) {
	metrics := NewRuntimeMetrics(2)
	metrics.SetKVAllocation(4096, 8)
	for _, position := range []int{1, 3, 9} {
		metrics.RecordForward(ForwardMeasurement{
			Position: position,
			GPUTime:  time.Millisecond,
			WallTime: 2 * time.Millisecond,
		})
	}
	forwards, bytes, allocated, tokens, maxTokens := metrics.Snapshot()
	if len(forwards) != 2 || forwards[0].Position != 3 || forwards[1].Position != 9 {
		t.Errorf("forwards = %+v, want last two measurements", forwards)
	}
	if bytes != 4096 || !allocated || tokens != 8 || maxTokens != 8 {
		t.Errorf("snapshot = (%d, %t, %d, %d), want (4096, true, 8, 8)", bytes, allocated, tokens, maxTokens)
	}
}

func TestPrefixCacheLongestMatch(t *testing.T) {
	cache := newPrefixCache(1024, 2)
	cache.store([]int32{1, 2, 3}, &prefixSnapshot{tokens: 3, bytes: 24})
	cache.store([]int32{1, 2, 3, 4}, &prefixSnapshot{tokens: 4, bytes: 32})
	got, matched, ok := cache.load([]int32{1, 2, 3, 4, 5})
	if !ok || got.tokens != 4 || matched != 4 {
		t.Fatalf("load longest prefix = (%+v, %d, %t), want 4-token hit", got, matched, ok)
	}
	metrics := cache.metrics()
	if metrics.Hits != 1 || metrics.Misses != 0 || metrics.SavedTokens != 4 {
		t.Errorf("metrics = %+v, want one 4-token hit", metrics)
	}
}

func TestDispatchTunerLocksMeasuredVariants(t *testing.T) {
	tuner := NewDispatchTuner()
	for i := 0; i < 3; i++ {
		if got := tuner.Width(); got != 64 {
			t.Fatalf("trial %d width = %d, want 64", i, got)
		}
		tuner.Record(64, 2*time.Millisecond)
	}
	for i := 0; i < 3; i++ {
		if got := tuner.Width(); got != 128 {
			t.Fatalf("trial %d width = %d, want 128", i, got)
		}
		tuner.Record(128, time.Millisecond)
	}
	data := tuner.Tracker().Telemetry()
	if !data.Convergence.Locked || data.Convergence.OptimalTG != [3]int{128, 1, 1} {
		t.Errorf("convergence = %+v, want locked 128-thread variant", data.Convergence)
	}
	if len(data.AutoTuningTrials) != 6 || len(data.HotSwaps) != 1 {
		t.Errorf("trials/hot swaps = %d/%d, want 6/1", len(data.AutoTuningTrials), len(data.HotSwaps))
	}
}
