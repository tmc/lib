//go:build darwin

package main

import (
	"testing"
	"time"

	"github.com/tmc/lib/examples/internal/continuousbatch"
	"github.com/tmc/lib/examples/internal/inferbench"
)

func TestObservedOutputBenchmarkCapturesFinalLogitsAndPopulatedKV(t *testing.T) {
	for _, width := range []int{1, 2, 4} {
		t.Run(string(rune('0'+width)), func(t *testing.T) {
			cfg := DefaultTinyLlamaConfig()
			eng, err := GetOrCompileEngine(cfg)
			if err != nil {
				t.Fatal(err)
			}
			adapter, err := newScheduledEngineAdapter(eng, cfg, width)
			if err != nil {
				t.Fatal(err)
			}
			defer adapter.Close()
			adapter.enablePhysicalLedger()
			adapter.observedOutputs = newObservedOutputCollector()
			baseline, _ := eng.memory.Snapshot()
			scheduler, err := continuousbatch.New(adapter, continuousbatch.Config{MaxActive: width, MaxQueue: width, MaxBatch: width, BatchWait: time.Microsecond, TokenBuffer: 3, PrefillWorkers: width})
			if err != nil {
				t.Fatal(err)
			}
			defer scheduler.Close()

			prompts := make([][]int32, width)
			for i := range prompts {
				prompts[i] = []int32{int32(3 + i), int32(7 + i)}
			}
			workload := inferbench.Workload{PromptTokenIDs: prompts, ContextTokens: 2, GeneratedTokens: 2, Concurrency: width, WarmupRuns: 1, Sampler: "greedy-argmax", CacheState: "cold", OutputMode: inferbench.ObservedOutputMode, FullLogits: cfg.VocabSize, KVLayers: cfg.NumLayers, KVPerToken: 2 * cfg.NumLayers * cfg.NumKVHeads * cfg.HeadDim}
			sample, err := benchmarkSample(t.Context(), scheduler, adapter, eng, workload, "observed")
			if err != nil {
				t.Fatal(err)
			}
			if len(sample.ObservedOutputs) != width || sample.ObservedSHA256 == "" {
				t.Fatalf("observed outputs = %+v hash=%q", sample.ObservedOutputs, sample.ObservedSHA256)
			}
			if sample.PeakMemoryBytes < baseline || sample.PeakMemoryBytes == 0 {
				t.Fatalf("peak memory = %d, want at least persistent allocation %d", sample.PeakMemoryBytes, baseline)
			}
			for i, output := range sample.ObservedOutputs {
				wantKV := 2 * cfg.NumLayers * (workload.ContextTokens + workload.GeneratedTokens - 1) * cfg.NumKVHeads * cfg.HeadDim
				if output.Request != i || output.FullLogitsElements != cfg.VocabSize || output.PopulatedKVElements != wantKV || output.PopulatedTokens != 3 || len(output.Layers) != cfg.NumLayers {
					t.Fatalf("request %d output = %+v, want logits=%d kv=%d", i, output, cfg.VocabSize, wantKV)
				}
				for layer, got := range output.Layers {
					if got.Layer != layer || len(got.KeySHA256) != 64 || len(got.ValueSHA256) != 64 {
						t.Fatalf("request %d layer %d = %+v", i, layer, got)
					}
				}
			}
			if err := adapter.resetPrefixes(); err != nil {
				t.Fatal(err)
			}
			repeated, err := benchmarkSample(t.Context(), scheduler, adapter, eng, workload, "repeated")
			if err != nil {
				t.Fatal(err)
			}
			for i := range sample.ObservedOutputs {
				got, want := repeated.ObservedOutputs[i], sample.ObservedOutputs[i]
				if got.FullLogitsSHA256 != want.FullLogitsSHA256 || got.PopulatedKVSHA256 != want.PopulatedKVSHA256 {
					t.Fatalf("request %d nondeterministic output: got=%+v want=%+v", i, got, want)
				}
			}
		})
	}
}
