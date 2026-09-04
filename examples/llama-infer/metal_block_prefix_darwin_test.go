//go:build darwin

package main

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/internal/metalprefix"
)

func TestMetalBlockPrefixMatchesColdSequentialUnderPressure(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	testMetalBlockPrefixMatchesCold(t, cfg, []int32{1, 10, 20}, []int32{30, 40})
}

func TestMetalBlockPrefixRealFP16MatchesColdSequentialUnderPressure(t *testing.T) {
	model := os.Getenv("LLAMA_INFER_TEST_MODEL")
	if model == "" {
		t.Skip("LLAMA_INFER_TEST_MODEL is not set")
	}
	cfg := Llama3_2_1BConfig(model)
	cfg.Precision = "f16"
	cfg.MaxSeqLen = 64
	testMetalBlockPrefixMatchesCold(t, cfg, []int32{cfg.BOSTokenID, 22691, 11}, []int32{358, 13})
}

func testMetalBlockPrefixMatchesCold(t *testing.T, cfg Config, prefix, suffix []int32) {
	t.Helper()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	prompt := append(append([]int32(nil), prefix...), suffix...)

	cold, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer cold.Free()
	coldCurrent, err := GPUBatchedPrefillPassKV(eng, cfg, cold, prompt, 0)
	if err != nil {
		t.Fatal(err)
	}
	coldToken, err := GPUForwardPassKV(eng, cfg, cold, coldCurrent, len(prompt))
	if err != nil {
		t.Fatal(err)
	}
	coldLogits, err := eng.ReadLogits()
	if err != nil {
		t.Fatal(err)
	}

	source, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Free()
	prefixCurrent, err := GPUBatchedPrefillPassKV(eng, cfg, source, prefix, 0)
	if err != nil {
		t.Fatal(err)
	}
	bytesPerPrefix := int64(2 * cfg.NumLayers * len(prefix) * cfg.NumKVHeads * cfg.HeadDim * 4)
	cache, err := metalprefix.New(metalprefix.Config{
		Device: eng.dev, MaxBytes: 3 * bytesPerPrefix, MaxEntries: 1,
		BlockTokens: 2, Layers: cfg.NumLayers, KVHeads: cfg.NumKVHeads, HeadDim: cfg.HeadDim,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := cache.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()
	if err := cache.Put(context.Background(), prefix, prefixCurrent, sequenceSlices(source.KCache), sequenceSlices(source.VCache), metalprefix.CompletedFence()); err != nil {
		t.Fatal(err)
	}
	lease, ok, err := cache.Acquire(prompt)
	if err != nil || !ok {
		t.Fatalf("Acquire = %v, %v", ok, err)
	}
	defer lease.Release()

	// Force the prefix out of lookup while its lease remains live. Restore must
	// still see the original immutable buffers.
	pressureTokens := []int32{90, 91, 92}
	pk, pv := testKVData(cfg, len(pressureTokens), 1000)
	if err := cache.Put(context.Background(), pressureTokens, 93, pk, pv, metalprefix.CompletedFence()); err != nil {
		t.Fatal(err)
	}
	if _, hit, err := cache.Acquire(prefix); err != nil || hit {
		t.Fatalf("evicted prefix lookup = %v, %v; want miss", hit, err)
	}

	warm, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer warm.Free()
	if err := lease.Restore(sequenceSlices(warm.KCache), sequenceSlices(warm.VCache), len(prefix)); err != nil {
		t.Fatal(err)
	}
	warmCurrent, err := GPUBatchedPrefillPassKV(eng, cfg, warm, prompt[len(prefix):], len(prefix))
	if err != nil {
		t.Fatal(err)
	}
	warmToken, err := GPUForwardPassKV(eng, cfg, warm, warmCurrent, len(prompt))
	if err != nil {
		t.Fatal(err)
	}
	warmLogits, err := eng.ReadLogits()
	if err != nil {
		t.Fatal(err)
	}
	if warmToken != coldToken {
		t.Fatalf("warm token = %d, cold token = %d", warmToken, coldToken)
	}
	var maxAbs float32
	for i := range coldLogits {
		d := float32(math.Abs(float64(coldLogits[i] - warmLogits[i])))
		if d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs > 1e-4 {
		t.Fatalf("warm/cold max logit difference = %g, want <= 1e-4", maxAbs)
	}
	t.Logf("partial_prefix_tokens=%d next_token=%d max_abs_logits=%g physical=%+v", len(prefix), warmToken, maxAbs, cache.Stats())
}

func sequenceSlices(slices []metalmem.Slice[float32]) [][]float32 {
	out := make([][]float32, len(slices))
	for i := range slices {
		out[i] = slices[i].HostSlice()
	}
	return out
}

func testKVData(cfg Config, tokens int, seed float32) ([][]float32, [][]float32) {
	k := make([][]float32, cfg.NumLayers)
	v := make([][]float32, cfg.NumLayers)
	n := tokens * cfg.NumKVHeads * cfg.HeadDim
	for layer := range k {
		k[layer] = make([]float32, n)
		v[layer] = make([]float32, n)
		for i := 0; i < n; i++ {
			k[layer][i] = seed + float32(layer*n+i)
			v[layer][i] = -k[layer][i]
		}
	}
	return k, v
}
