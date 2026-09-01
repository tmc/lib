//go:build darwin

package main

import (
	"context"
	"testing"

	"github.com/tmc/lib/apple/x/metal/metalsafetensors"
)

func TestPrecisionTelemetry(t *testing.T) {
	eng := &ModelEngine{weights: &metalsafetensors.Weights{Tensors: map[string]metalsafetensors.TensorInfo{
		"layer.0.weight": {DType: "F16", OffsetStart: 0, OffsetEnd: 32},
		"layer.1.weight": {DType: "F16", OffsetStart: 32, OffsetEnd: 96},
		"layer.0.scale":  {DType: "F32", OffsetStart: 96, OffsetEnd: 112},
	}}}

	got := eng.precisionTelemetry()
	want := []TensorPrecision{
		{DType: "F16", TensorCount: 2, Bytes: 96},
		{DType: "F32", TensorCount: 1, Bytes: 16},
	}
	if len(got) != len(want) {
		t.Fatalf("precisionTelemetry() = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("precisionTelemetry()[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}

	response := NewTelemetryServer(eng, Config{MaxSeqLen: 128}).memoryTelemetry()
	if response.Precision.State != "observed" || len(response.Precision.DTypes) != len(want) {
		t.Errorf("precision telemetry = %+v, want observed dtype aggregate", response.Precision)
	}
}

func TestMemoryTelemetryUsesRuntimeMetrics(t *testing.T) {
	metrics := NewRuntimeMetrics(4)
	metrics.SetKVAllocation(8192, 32)
	metrics.RecordForward(ForwardMeasurement{Position: 7})
	prefixes := newPrefixCache(1024, 2)
	prefixes.store([]int32{1, 2}, &prefixSnapshot{tokens: 2, bytes: 16})
	prefixes.load([]int32{1, 2, 3})

	srv := NewTelemetryServer(&ModelEngine{metrics: metrics, prefixes: prefixes}, Config{MaxSeqLen: 32})
	response := srv.memoryTelemetry()
	if response.KVCache.State != "observed" || response.KVCache.AllocatedBytes != 8192 || response.KVCache.ActiveTokens != 8 {
		t.Errorf("KV cache telemetry = %+v, want observed runtime allocation", response.KVCache)
	}
	if response.PrefixCache.State != "observed" || response.PrefixCache.Entries != 1 || response.PrefixCache.Hits != 1 || response.PrefixCache.SavedTokens != 2 {
		t.Errorf("prefix cache telemetry = %+v, want observed prefix cache metrics", response.PrefixCache)
	}
}

func TestGenerateRestoresPrefixKVSnapshot(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		t.Skipf("Metal engine unavailable: %v", err)
	}
	before, _, _, _, _ := eng.metrics.Snapshot()
	if err := eng.Generate(context.Background(), cfg, "repeat prefix", 1, func(int32, string) bool { return true }); err != nil {
		t.Fatalf("first generate: %v", err)
	}
	afterFirst, _, _, _, _ := eng.metrics.Snapshot()
	if err := eng.Generate(context.Background(), cfg, "repeat prefix", 1, func(int32, string) bool { return true }); err != nil {
		t.Fatalf("second generate: %v", err)
	}
	afterSecond, _, _, _, _ := eng.metrics.Snapshot()
	firstForwards := len(afterFirst) - len(before)
	secondForwards := len(afterSecond) - len(afterFirst)
	if firstForwards <= 1 {
		t.Fatalf("first generate ran %d forwards, want prefill plus decode", firstForwards)
	}
	if secondForwards != 1 {
		t.Errorf("second generate ran %d forwards, want only decode after prefix restore", secondForwards)
	}
	metrics := eng.prefixCacheMetrics()
	if metrics.Hits < 1 || metrics.SavedTokens < 1 {
		t.Errorf("prefix metrics = %+v, want observed restored prefix", metrics)
	}
}
