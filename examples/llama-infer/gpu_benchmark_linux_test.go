//go:build linux

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCUDABenchmarkPrefill(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if os.Getenv("LLAMA_CUDA_PREFILL_TEST") != "1" || path == "" {
		t.Skip("set LLAMA_CUDA_PREFILL_TEST=1 and LLAMA_MODEL_PATH for CUDA validation")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.Precision = "f32"
	cfg.MaxSeqLen = 96
	cfg.Prompt = "The capital of France is"
	cfg.GenTokens = 2
	cfg.SearchDepths = []int{1}
	eng, err := GetOrCompileCudaEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	requireCUDAAttentionCheckpoint(t, path, &eng.mCfg)
	output := filepath.Join(t.TempDir(), "benchmark.txt")
	f, err := os.Create(output)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	stdout := os.Stdout
	os.Stdout = f
	defer func() { os.Stdout = stdout }()
	err = RunBenchSuite(cfg, 1)
	os.Stdout = stdout
	if err != nil {
		t.Fatal(err)
	}
	if eng.prefillScratch == nil {
		t.Fatal("benchmark did not initialize batched prefill storage")
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	if strings.Contains(text, "assumed-peak-") {
		t.Fatalf("benchmark emitted unconfigured hardware peaks: %s", text)
	}
	for _, phase := range []string{"/TTFT-1", "/TPOT-1", "/prefill-1", "/decode-1"} {
		if !strings.Contains(text, phase) {
			t.Errorf("missing %s result in %s", phase, text)
		}
	}
	t.Log(text)
}
