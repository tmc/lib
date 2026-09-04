//go:build darwin

package main

import (
	"os"
	"testing"
	"unsafe"

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
	if sequence, err := eng.NewSequenceKV(Config{NumLayers: 2, MaxSeqLen: maxInt, NumKVHeads: 2, HeadDim: 2}); err == nil {
		sequence.Free()
		t.Fatal("oversized sequence KV allocation succeeded")
	}
	if current, _ := tracker.Snapshot(); current != 0 {
		t.Fatalf("after sequence rollback current = %d, want 0", current)
	}
	if workspace, err := newMetalBlockWorkspace(eng, Config{NumLayers: 2, MaxSeqLen: 2}, maxInt); err == nil {
		workspace.Free()
		t.Fatal("oversized block workspace allocation succeeded")
	}
	if current, _ := tracker.Snapshot(); current != 0 {
		t.Fatalf("after workspace rollback current = %d, want 0", current)
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
			sequence, err := eng.NewSequenceKV(cfg)
			if err != nil {
				t.Fatal(err)
			}
			sequenceBytes := uint64(2 * cfg.NumLayers * cfg.MaxSeqLen * cfg.NumKVHeads * cfg.HeadDim * 4)
			current, peak = eng.memory.Snapshot()
			if current != baseline+sequenceBytes || peak != current {
				t.Fatalf("sequence allocation = %d, %d, want %d", current, peak, baseline+sequenceBytes)
			}
			sequence.Free()
			sequence.Free()
			if current, _ := eng.memory.Snapshot(); current != baseline {
				t.Fatalf("after sequence Free current = %d, want %d", current, baseline)
			}

			eng.memory.ResetPeak()
			runtime, err := newProductionPrefixRuntime(eng, cfg, 4)
			if err != nil {
				t.Fatal(err)
			}
			workspaceBytes := uint64(cfg.NumLayers * 4 * (cfg.MaxSeqLen*int(unsafe.Sizeof(metalBlockDescriptor{})) + int(unsafe.Sizeof(metalBlockRow{}))))
			current, peak = eng.memory.Snapshot()
			if current != baseline+workspaceBytes || peak != current {
				t.Fatalf("prefix workspace allocation = %d, %d, want %d", current, peak, baseline+workspaceBytes)
			}
			if err := runtime.Close(); err != nil {
				t.Fatal(err)
			}
			if current, _ := eng.memory.Snapshot(); current != baseline {
				t.Fatalf("after prefix runtime Close current = %d, want %d", current, baseline)
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

func TestRealMetalEngineMemoryTracker(t *testing.T) {
	path := os.Getenv("LLAMA_INFER_TEST_MODEL")
	if path == "" {
		t.Skip("LLAMA_INFER_TEST_MODEL is not set")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.Precision = "f16"
	cfg.MaxSeqLen = 8
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Free()
	baseline, _ := eng.memory.Snapshot()
	if baseline == 0 {
		t.Fatal("persistent Metal allocation baseline is zero")
	}
	if err := benchmarkPeakMemorySupport(eng); err != nil {
		t.Fatal(err)
	}
	if err := benchmarkMemoryReset(t.Context(), eng); err != nil {
		t.Fatal(err)
	}
	sequence, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatal(err)
	}
	wantKV := uint64(2 * cfg.NumLayers * cfg.MaxSeqLen * cfg.NumKVHeads * cfg.HeadDim * 4)
	current, _ := eng.memory.Snapshot()
	peak, err := benchmarkPeakMemory(eng)
	if err != nil {
		t.Fatal(err)
	}
	if current != baseline+wantKV || peak != current {
		t.Fatalf("sequence allocation = %d, %d, want %d", current, peak, baseline+wantKV)
	}
	sequence.Free()
	sequence.Free()
	current, peak = eng.memory.Snapshot()
	if current != baseline || peak != baseline+wantKV {
		t.Fatalf("sequence release = %d, %d, want %d, %d", current, peak, baseline, baseline+wantKV)
	}
	t.Logf("receipt persistent_bytes=%d sequence_kv_bytes=%d interval_peak_bytes=%d", baseline, wantKV, peak)
}
