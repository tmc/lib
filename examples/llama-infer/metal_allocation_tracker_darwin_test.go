//go:build darwin

package main

import (
	"testing"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func TestMetalServingConstructorRollback(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}
	defer dev.Close()
	tracker := new(metalmem.Tracker)
	eng := &ModelEngine{dev: dev, memory: tracker}
	cfg := DefaultTinyLlamaConfig()
	const maxInt = int(^uint(0) >> 1)
	if cache, err := eng.NewBatchedAttentionCache(cfg, maxInt); err == nil {
		cache.Free()
		t.Fatal("oversized cache allocation succeeded")
	}
	if current, _ := tracker.Snapshot(); current != 0 {
		t.Fatalf("after cache rollback current = %d, want 0", current)
	}
	if tail, err := eng.NewBatchedModelTail(cfg, maxInt); err == nil {
		tail.Free()
		t.Fatal("oversized tail allocation succeeded")
	}
	if current, _ := tracker.Snapshot(); current != 0 {
		t.Fatalf("after tail rollback current = %d, want 0", current)
	}
}

func TestMetalEngineAllocationTracker(t *testing.T) {
	for _, precision := range []string{"f32", "f16"} {
		t.Run(precision, func(t *testing.T) {
			cfg := DefaultTinyLlamaConfig()
			cfg.Name += "-allocation-" + precision
			cfg.Precision = precision
			eng, err := GetOrCompileEngine(cfg)
			if err != nil {
				t.Fatal(err)
			}
			baseline, initialPeak := eng.memory.Snapshot()
			t.Logf("engine baseline=%d initial_peak=%d", baseline, initialPeak)
			if baseline == 0 || initialPeak != baseline {
				t.Fatalf("engine allocation = %d, peak %d, want equal nonzero values", baseline, initialPeak)
			}

			eng.memory.ResetPeak()
			cache, err := eng.NewBatchedAttentionCache(cfg, 4)
			if err != nil {
				t.Fatal(err)
			}
			cacheBytes := uint64(2*cfg.NumLayers*4*cfg.MaxSeqLen*cfg.NumKVHeads*cfg.HeadDim*4 + 2*4*4)
			current, peak := eng.memory.Snapshot()
			t.Logf("batched cache current=%d peak=%d bytes=%d", current, peak, cacheBytes)
			if current != baseline+cacheBytes || peak != current {
				t.Fatalf("cache allocation = %d, peak %d, want %d", current, peak, baseline+cacheBytes)
			}
			cache.Free()
			cache.Free()
			if current, _ := eng.memory.Snapshot(); current != baseline {
				t.Fatalf("after cache Free current = %d, want %d", current, baseline)
			}

			eng.memory.ResetPeak()
			tail, err := eng.NewBatchedModelTail(cfg, 4)
			if err != nil {
				t.Fatal(err)
			}
			tailBytes := uint64(4*cfg.Dim*4 + 4*4 + 4*4 + 4*((cfg.VocabSize+255)/256)*2*4)
			current, peak = eng.memory.Snapshot()
			t.Logf("batched tail current=%d peak=%d bytes=%d", current, peak, tailBytes)
			if current != baseline+tailBytes || peak != current {
				t.Fatalf("tail allocation = %d, peak %d, want %d", current, peak, baseline+tailBytes)
			}
			tail.Free()
			if current, _ := eng.memory.Snapshot(); current != baseline {
				t.Fatalf("after tail Free current = %d, want %d", current, baseline)
			}

			eng.memory.ResetPeak()
			if err := eng.ensurePrefillArena(1, cfg); err != nil {
				t.Fatal(err)
			}
			small, _ := eng.memory.Snapshot()
			if small <= baseline {
				t.Fatalf("prefill allocation = %d, want greater than baseline %d", small, baseline)
			}
			if err := eng.ensurePrefillArena(256, cfg); err != nil {
				t.Fatal(err)
			}
			large, replacementPeak := eng.memory.Snapshot()
			t.Logf("prefill small=%d replacement=%d peak=%d", small, large, replacementPeak)
			if large <= small || replacementPeak != large {
				t.Fatalf("prefill replacement = %d, peak %d, previous %d", large, replacementPeak, small)
			}
			eng.synchronize()
			eng.freePrefillArena()
			if current, _ := eng.memory.Snapshot(); current != baseline {
				t.Fatalf("after prefill Free current = %d, want %d", current, baseline)
			}

			eng.Free()
			if current, _ := eng.memory.Snapshot(); current != 0 {
				t.Fatalf("after engine Free current = %d, want 0", current)
			}
		})
	}
}
