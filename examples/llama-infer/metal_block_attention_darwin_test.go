//go:build darwin

package main

import (
	"context"
	"errors"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/examples/internal/metalprefix"
	"github.com/tmc/lib/lm/kvcache"
)

func TestBlockTableAttentionMatchesContiguous(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	prefix := []int32{1, 2, 3, 4, 5}
	k, v := blockAttentionData(cfg, len(prefix), 0.013)
	bytes := int64(2 * cfg.NumLayers * len(prefix) * cfg.NumKVHeads * cfg.HeadDim * 4)
	cache, err := metalprefix.New(metalprefix.Config{Device: eng.dev, MaxBytes: bytes * 3, MaxEntries: 1, BlockTokens: 2, Layers: cfg.NumLayers, KVHeads: cfg.NumKVHeads, HeadDim: cfg.HeadDim})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := cache.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()
	if err := cache.Put(context.Background(), prefix, 7, k, v, metalprefix.CompletedFence()); err != nil {
		t.Fatal(err)
	}
	lease, ok, err := cache.Acquire(append(append([]int32(nil), prefix...), 9))
	if err != nil || !ok {
		t.Fatalf("Acquire = %v, %v", ok, err)
	}
	defer lease.Release()

	// Eviction removes discoverability but cannot retire blocks being read.
	pk, pv := blockAttentionData(cfg, len(prefix), 2.1)
	if err := cache.Put(context.Background(), []int32{8, 8, 8, 8, 8}, 4, pk, pv, metalprefix.CompletedFence()); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := cache.Acquire(prefix); err != nil || ok {
		t.Fatalf("evicted Acquire = %v, %v", ok, err)
	}

	kd := cfg.NumKVHeads * cfg.HeadDim
	qd := cfg.NumHeads * cfg.HeadDim
	suffixK := make([]float32, 2*kd)
	suffixV := make([]float32, 2*kd)
	for i := range suffixK {
		suffixK[i] = float32(i%11-5) * 0.021
		suffixV[i] = float32(i%7-3) * 0.017
	}
	for layer := range cfg.NumLayers {
		for _, rows := range []int{1, 2, 4} {
			for row := range rows {
				q := make([]float32, qd)
				for i := range q {
					q[i] = float32((i+row)%13-6) * 0.019
				}
				got, err := RunBlockTableAttention(context.Background(), eng, cfg, lease, layer, q, suffixK, suffixV)
				if err != nil {
					t.Fatalf("layer %d B%d row %d: %v", layer, rows, row, err)
				}
				want := contiguousAttention(cfg, q, append(append([]float32(nil), k[layer]...), suffixK...), append(append([]float32(nil), v[layer]...), suffixV...))
				for i := range want {
					if d := math.Abs(float64(got[i] - want[i])); d > 2e-5 {
						t.Fatalf("layer %d B%d row %d output[%d] diff %g", layer, rows, row, i, d)
					}
				}
			}
		}
	}
	stats := cache.Stats()
	if stats.Entries != 1 || stats.Blocks <= cfg.NumLayers*3 {
		t.Fatalf("physical stats = %+v", stats)
	}
	t.Logf("physical_block_table=true prefix_blocks_per_layer=3 partial_block=true layers=%d stats=%+v", cfg.NumLayers, stats)
}

func TestBlockTableAttentionRealFP16KV(t *testing.T) {
	model := os.Getenv("LLAMA_INFER_TEST_MODEL")
	if model == "" {
		t.Skip("LLAMA_INFER_TEST_MODEL is not set")
	}
	cfg := Llama3_2_1BConfig(model)
	cfg.Precision = "f16"
	cfg.MaxSeqLen = 32
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int32{cfg.BOSTokenID, 22691, 11, 358, 13}
	seq, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer seq.Free()
	if _, err := GPUBatchedPrefillPassKV(eng, cfg, seq, tokens, 0); err != nil {
		t.Fatal(err)
	}
	prefix := tokens[:3]
	kd := cfg.NumKVHeads * cfg.HeadDim
	qd := cfg.NumHeads * cfg.HeadDim
	cache, err := metalprefix.New(metalprefix.Config{Device: eng.dev, MaxBytes: 64 << 20, MaxEntries: 2, BlockTokens: 2, Layers: cfg.NumLayers, KVHeads: cfg.NumKVHeads, HeadDim: cfg.HeadDim})
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Put(context.Background(), prefix, tokens[3], sequenceSlices(seq.KCache), sequenceSlices(seq.VCache), metalprefix.CompletedFence()); err != nil {
		t.Fatal(err)
	}
	lease, ok, err := cache.Acquire(tokens)
	if err != nil || !ok {
		t.Fatalf("Acquire = %v, %v", ok, err)
	}
	defer lease.Release()
	for layer := range cfg.NumLayers {
		q := make([]float32, qd)
		for i := range q {
			q[i] = float32(i%17-8) * 0.007
		}
		allK, allV := seq.KCache[layer].HostSlice(), seq.VCache[layer].HostSlice()
		got, err := RunBlockTableAttention(context.Background(), eng, cfg, lease, layer, q, allK[3*kd:5*kd], allV[3*kd:5*kd])
		if err != nil {
			t.Fatal(err)
		}
		want := contiguousAttention(cfg, q, allK[:5*kd], allV[:5*kd])
		for i := range want {
			if d := math.Abs(float64(got[i] - want[i])); d > 3e-5 {
				t.Fatalf("layer %d output[%d] diff %g", layer, i, d)
			}
		}
	}
	lease.Release()
	if err := cache.Close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("real_fp16_kv=true model=%s layers=%d block_table=true", model, cfg.NumLayers)
}

func TestBlockTableAttentionRejectsCancellationAndStaleLease(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	k, v := blockAttentionData(cfg, 2, 0.1)
	cache, err := metalprefix.New(metalprefix.Config{Device: eng.dev, MaxBytes: 1 << 20, MaxEntries: 2, BlockTokens: 2, Layers: cfg.NumLayers, KVHeads: cfg.NumKVHeads, HeadDim: cfg.HeadDim})
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Put(context.Background(), []int32{1, 2}, 3, k, v, metalprefix.CompletedFence()); err != nil {
		t.Fatal(err)
	}
	lease, ok, err := cache.Acquire([]int32{1, 2, 3})
	if err != nil || !ok {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	q := make([]float32, cfg.NumHeads*cfg.HeadDim)
	s := make([]float32, cfg.NumKVHeads*cfg.HeadDim)
	if _, err := RunBlockTableAttention(ctx, eng, cfg, lease, 0, q, s, s); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
	}
	lease.Release()
	if _, err := lease.PinLayer(0); !errors.Is(err, kvcache.ErrStale) {
		t.Fatalf("stale PinLayer error = %v", err)
	}
	if err := cache.Close(); err != nil {
		t.Fatal(err)
	}
}

func blockAttentionData(cfg Config, tokens int, scale float32) ([][]float32, [][]float32) {
	n := tokens * cfg.NumKVHeads * cfg.HeadDim
	k := make([][]float32, cfg.NumLayers)
	v := make([][]float32, cfg.NumLayers)
	for layer := range cfg.NumLayers {
		k[layer] = make([]float32, n)
		v[layer] = make([]float32, n)
		for i := range n {
			k[layer][i] = float32((i+layer)%17-8) * scale
			v[layer][i] = float32((i+3*layer)%19-9) * scale
		}
	}
	return k, v
}

func contiguousAttention(cfg Config, q, k, v []float32) []float32 {
	kd := cfg.NumKVHeads * cfg.HeadDim
	tokens := len(k) / kd
	out := make([]float32, len(q))
	scale := float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	for h := range cfg.NumHeads {
		kh := h / (cfg.NumHeads / cfg.NumKVHeads)
		scores := make([]float32, tokens)
		maxScore := float32(-math.MaxFloat32)
		for token := range tokens {
			for d := range cfg.HeadDim {
				scores[token] += q[h*cfg.HeadDim+d] * k[token*kd+kh*cfg.HeadDim+d]
			}
			scores[token] *= scale
			if scores[token] > maxScore {
				maxScore = scores[token]
			}
		}
		var denom float32
		for token := range tokens {
			scores[token] = float32(math.Exp(float64(scores[token] - maxScore)))
			denom += scores[token]
		}
		for d := range cfg.HeadDim {
			for token := range tokens {
				out[h*cfg.HeadDim+d] += scores[token] * v[token*kd+kh*cfg.HeadDim+d]
			}
			out[h*cfg.HeadDim+d] /= denom
		}
	}
	return out
}
