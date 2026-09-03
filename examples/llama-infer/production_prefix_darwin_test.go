//go:build darwin

package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/tmc/lib/lm/kvcache"
)

func TestProductionPrefixFullLogitsAndAllKV(t *testing.T) {
	for _, width := range []int{1, 2, 4} {
		t.Run(fmt.Sprintf("B%d", width), func(t *testing.T) {
			cfg := DefaultTinyLlamaConfig()
			eng, err := GetOrCompileEngine(cfg)
			if err != nil {
				t.Fatal(err)
			}
			runtime, err := newProductionPrefixRuntime(eng, cfg, width)
			if err != nil {
				t.Fatal(err)
			}
			defer runtime.Close()
			prefix := []int32{3, 4, 5, 6}
			source, err := eng.NewBatchedAttentionCache(cfg, 1)
			if err != nil {
				t.Fatal(err)
			}
			defer source.Free()
			packedPrefix := packTestPrompts(t, cfg, [][]int32{prefix})
			prefixChain, err := RunSegmentedLayerChain(t.Context(), eng, cfg, source, packedPrefix)
			if err != nil {
				t.Fatal(err)
			}
			prefixTail, err := eng.NewBatchedModelTail(cfg, 1)
			if err != nil {
				t.Fatal(err)
			}
			prefixResult, err := prefixTail.Run(eng, cfg, append([]float32(nil), prefixChain.Hidden...), nil)
			prefixTail.Free()
			if err != nil {
				t.Fatal(err)
			}
			if err := runtime.Put(t.Context(), prefix, prefixResult.Tokens[0], source, 0); err != nil {
				t.Fatal(err)
			}
			for layer := range source.K {
				clear(source.K[layer].HostSlice())
				clear(source.V[layer].HostSlice())
			}

			prompts := make([][]int32, width)
			leases := make([]*productionPrefixLease, width)
			tokens := make([]int32, width)
			positions := make([]int, width)
			rows := make([]int, width)
			for row := range width {
				prompts[row] = append(append([]int32(nil), prefix...), int32(20+row))
				leases[row], _, err = runtime.Acquire(prompts[row])
				if err != nil || leases[row] == nil {
					t.Fatalf("row %d Acquire: %v", row, err)
				}
				defer leases[row].Release()
				tokens[row], positions[row], rows[row] = prompts[row][len(prefix)], len(prefix), row
			}
			warmCache, err := eng.NewBatchedAttentionCache(cfg, width)
			if err != nil {
				t.Fatal(err)
			}
			defer warmCache.Free()
			warm, err := runtime.Run(t.Context(), eng, cfg, warmCache, leases, tokens, positions, rows)
			if err != nil {
				t.Fatal(err)
			}
			warmTail, err := eng.NewBatchedModelTail(cfg, width)
			if err != nil {
				t.Fatal(err)
			}
			warmResult, err := warmTail.Run(eng, cfg, append([]float32(nil), warm.Hidden...), nil)
			warmTail.Free()
			if err != nil {
				t.Fatal(err)
			}

			coldCache, err := eng.NewBatchedAttentionCache(cfg, width)
			if err != nil {
				t.Fatal(err)
			}
			defer coldCache.Free()
			coldPacked := packTestPrompts(t, cfg, prompts)
			cold, err := RunSegmentedLayerChain(t.Context(), eng, cfg, coldCache, coldPacked)
			if err != nil {
				t.Fatal(err)
			}
			coldTail, err := eng.NewBatchedModelTail(cfg, width)
			if err != nil {
				t.Fatal(err)
			}
			coldResult, err := coldTail.Run(eng, cfg, append([]float32(nil), cold.Hidden...), nil)
			coldTail.Free()
			if err != nil {
				t.Fatal(err)
			}
			var maxLogit, maxKV float32
			for i := range coldResult.Logits {
				maxLogit = max(maxLogit, float32(math.Abs(float64(warmResult.Logits[i]-coldResult.Logits[i]))))
			}
			kd := cfg.NumKVHeads * cfg.HeadDim
			for row := range width {
				actualK := make([][]float32, cfg.NumLayers)
				actualV := make([][]float32, cfg.NumLayers)
				for layer := range cfg.NumLayers {
					actualK[layer] = make([]float32, len(prompts[row])*kd)
					actualV[layer] = make([]float32, len(prompts[row])*kd)
				}
				if err := leases[row].lease.Restore(actualK, actualV, len(prefix)); err != nil {
					t.Fatal(err)
				}
				for layer := range cfg.NumLayers {
					warmBase := row*cfg.MaxSeqLen*kd + len(prefix)*kd
					copy(actualK[layer][len(prefix)*kd:], warmCache.K[layer].HostSlice()[warmBase:warmBase+kd])
					copy(actualV[layer][len(prefix)*kd:], warmCache.V[layer].HostSlice()[warmBase:warmBase+kd])
					coldBase := row * cfg.MaxSeqLen * kd
					for i := range actualK[layer] {
						maxKV = max(maxKV, float32(math.Abs(float64(actualK[layer][i]-coldCache.K[layer].HostSlice()[coldBase+i]))))
						maxKV = max(maxKV, float32(math.Abs(float64(actualV[layer][i]-coldCache.V[layer].HostSlice()[coldBase+i]))))
					}
				}
			}
			if maxLogit > 1e-4 || maxKV > 1e-6 {
				t.Fatalf("max logit=%g KV=%g", maxLogit, maxKV)
			}
			t.Logf("width=%d full_logits=%d max_logit=%g all_kv=%d max_kv=%g prefix=%+v", width, len(coldResult.Logits), maxLogit, width*cfg.NumLayers*len(prompts[0])*kd*2, maxKV, runtime.Stats())
		})
	}
}

func TestProductionPrefixLeaseCancellationAndStale(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := newProductionPrefixRuntime(eng, cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()
	cache, err := eng.NewBatchedAttentionCache(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Free()
	prefix := []int32{3, 4, 5, 6}
	packed := packTestPrompts(t, cfg, [][]int32{prefix})
	chain, err := RunSegmentedLayerChain(t.Context(), eng, cfg, cache, packed)
	if err != nil {
		t.Fatal(err)
	}
	tail, err := eng.NewBatchedModelTail(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	result, err := tail.Run(eng, cfg, append([]float32(nil), chain.Hidden...), nil)
	tail.Free()
	if err != nil {
		t.Fatal(err)
	}
	if err := runtime.Put(t.Context(), prefix, result.Tokens[0], cache, 0); err != nil {
		t.Fatal(err)
	}
	lease, _, err := runtime.Acquire(append(append([]int32(nil), prefix...), 7))
	if err != nil || lease == nil {
		t.Fatalf("Acquire: %v", err)
	}

	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = runtime.Run(canceled, eng, cfg, cache, []*productionPrefixLease{lease}, []int32{7}, []int{len(prefix)}, []int{0})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Run: %v", err)
	}
	if _, err := runtime.Run(t.Context(), eng, cfg, cache, []*productionPrefixLease{lease}, []int32{7}, []int{len(prefix)}, []int{0}); err != nil {
		t.Fatalf("lease after cancellation: %v", err)
	}
	lease.Release()
	_, err = runtime.Run(t.Context(), eng, cfg, cache, []*productionPrefixLease{lease}, []int32{7}, []int{len(prefix)}, []int{0})
	if !errors.Is(err, kvcache.ErrStale) {
		t.Fatalf("released lease Run: %v", err)
	}
}

func packTestPrompts(t *testing.T, cfg Config, prompts [][]int32) PackedPrefill {
	t.Helper()
	segments := make([]PrefillSegment, len(prompts))
	for row, tokens := range prompts {
		segments[row] = PrefillSegment{RequestID: fmt.Sprintf("row-%d", row), Row: row, Tokens: tokens}
	}
	var packer PrefillPacker
	packed, err := packer.Pack(t.Context(), segments, len(prompts)*cfg.MaxSeqLen, len(prompts), cfg.VocabSize, cfg.MaxSeqLen)
	if err != nil {
		t.Fatal(err)
	}
	return packed
}
