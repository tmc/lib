//go:build linux

package main

import (
	"os"
	"testing"
)

func TestRealCUDAEngineMemoryTracker(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.MaxSeqLen = 8
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	baseline, _ := eng.memory.Snapshot()
	if baseline == 0 {
		t.Fatal("persistent CUDA allocation baseline is zero")
	}
	if err := benchmarkMemoryReset(t.Context(), eng); err != nil {
		t.Fatal(err)
	}
	cache, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatal(err)
	}
	wantKV := uint64(2 * eng.mCfg.NumHiddenLayers * eng.mCfg.MaxSeqLen * eng.mCfg.NumKeyValueHeads * eng.mCfg.HeadDim() * 4)
	current, _ := eng.memory.Snapshot()
	peak, err := benchmarkPeakMemory(eng)
	if err != nil {
		t.Fatal(err)
	}
	if current != baseline+wantKV || peak != current {
		t.Fatalf("sequence allocation = %d, %d, want %d, %d", current, peak, baseline+wantKV, baseline+wantKV)
	}
	cache.Free()
	cache.Free()
	current, peak = eng.memory.Snapshot()
	if current != baseline || peak != baseline+wantKV {
		t.Fatalf("sequence release = %d, %d, want %d, %d", current, peak, baseline, baseline+wantKV)
	}

	if err := benchmarkMemoryReset(t.Context(), eng); err != nil {
		t.Fatal(err)
	}
	batch, err := newGPUBatchScratchpad(eng.dev, eng.memory, &eng.mCfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	current, peak = eng.memory.Snapshot()
	if current < baseline+cudaMatrixWorkspaceBytes || peak != current {
		t.Fatalf("batch allocation = %d, %d, want at least %d", current, peak, baseline+cudaMatrixWorkspaceBytes)
	}
	batch.Free()
	batch.Free()
	current, peak = eng.memory.Snapshot()
	if current != baseline || peak < baseline+cudaMatrixWorkspaceBytes {
		t.Fatalf("batch release = %d, %d, want baseline %d and workspace peak", current, peak, baseline)
	}
	t.Logf("receipt persistent_bytes=%d sequence_kv_bytes=%d batch_peak_bytes=%d", baseline, wantKV, peak)
}
