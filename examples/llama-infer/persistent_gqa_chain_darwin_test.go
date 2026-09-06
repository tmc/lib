//go:build darwin

package main

import (
	"context"
	"errors"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/apple/x/metal/metalattn"
)

func enablePersistentGQATestRoute(t *testing.T, eng *ModelEngine) {
	t.Helper()
	lib, err := eng.dev.CompileLibrary(metalattn.PersistentGQAMSL)
	if err != nil {
		t.Fatal(err)
	}
	pso, err := eng.dev.NewComputePipeline(lib, "PersistentGQABlockDecode")
	if err != nil {
		t.Fatal(err)
	}
	eng.testPersistentGQA, eng.testPersistentGQAPSO = true, pso
	t.Cleanup(func() {
		eng.testPersistentGQA = false
		eng.testPersistentGQAPSO = nil
		eng.testChainCompletionFail = nil
		eng.testRequireGPUTimestamps = false
	})
}

func TestPersistentGQAUninterruptedRealModelChain(t *testing.T) {
	model := os.Getenv("LLAMA_INFER_TEST_MODEL")
	if model == "" {
		t.Skip("LLAMA_INFER_TEST_MODEL is not set")
	}
	cfg := Llama3_2_1BConfig(model)
	cfg.Precision, cfg.MaxSeqLen = "f16", 64
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	enablePersistentGQATestRoute(t, eng)

	for _, width := range []int{1, 2, 4} {
		t.Run("B"+itoaTest(width), func(t *testing.T) {
			runtime, err := newProductionPrefixRuntime(eng, cfg, width)
			if err != nil {
				t.Fatal(err)
			}
			defer runtime.Close()
			prefixes := make([][]int32, width)
			leases := make([]*productionPrefixLease, width)
			positions, rows := make([]int, width), make([]int, width)
			first := make([]int32, width)
			history := make([][]int32, width)
			for row := range width {
				n := 17 + row
				prefixes[row] = make([]int32, n)
				prefixes[row][0] = cfg.BOSTokenID
				for i := 1; i < n; i++ {
					prefixes[row][i] = int32(100 + i + row*31)
				}
				source, err := eng.NewBatchedAttentionCache(cfg, 1)
				if err != nil {
					t.Fatal(err)
				}
				packed := packTestPrompts(t, cfg, [][]int32{prefixes[row]})
				chain, err := RunSegmentedLayerChain(t.Context(), eng, cfg, source, packed)
				if err != nil {
					t.Fatal(err)
				}
				tail, _ := eng.NewBatchedModelTail(cfg, 1)
				result, err := tail.Run(eng, cfg, append([]float32(nil), chain.Hidden...), nil)
				tail.Free()
				if err != nil {
					t.Fatal(err)
				}
				if err := runtime.Put(t.Context(), prefixes[row], result.Tokens[0], source, 0); err != nil {
					t.Fatal(err)
				}
				source.Free()
				first[row], positions[row], rows[row] = result.Tokens[0], n, row
				history[row] = append([]int32(nil), prefixes[row]...)
				lease, _, err := runtime.Acquire(append(append([]int32(nil), prefixes[row]...), first[row]))
				if err != nil || lease == nil {
					t.Fatalf("acquire row %d: %v", row, err)
				}
				leases[row] = lease
				defer lease.Release()
			}
			candidateCache, _ := eng.NewBatchedAttentionCache(cfg, width)
			defer candidateCache.Free()
			baselineCache, _ := eng.NewBatchedAttentionCache(cfg, width)
			defer baselineCache.Free()
			candidateTail, _ := eng.NewBatchedModelTail(cfg, width)
			defer candidateTail.Free()
			baselineTail, _ := eng.NewBatchedModelTail(cfg, width)
			defer baselineTail.Free()
			tokens := append([]int32(nil), first...)
			for step := 0; step < 2; step++ {
				for row := range width {
					history[row] = append(history[row], tokens[row])
				}
				eng.testPersistentGQA = true
				candidate, err := runtime.Run(t.Context(), eng, cfg, candidateCache, leases, tokens, positions, rows)
				if err != nil {
					t.Fatal(err)
				}
				candidateHidden := append([]float32(nil), candidate.Hidden...)
				candidateResult, err := candidateTail.Run(eng, cfg, candidateHidden, nil)
				if err != nil {
					t.Fatal(err)
				}
				eng.testPersistentGQA = false
				baseline, err := runtime.Run(t.Context(), eng, cfg, baselineCache, leases, tokens, positions, rows)
				if err != nil {
					t.Fatal(err)
				}
				baselineHidden := append([]float32(nil), baseline.Hidden...)
				baselineResult, err := baselineTail.Run(eng, cfg, baselineHidden, nil)
				if err != nil {
					t.Fatal(err)
				}
				checkVector(t, "hidden", candidateHidden, baselineHidden, 4e-4)
				checkVector(t, "logits", candidateResult.Logits, baselineResult.Logits, 8e-4)
				if candidate.CandidateAttentionDispatches != cfg.NumLayers || candidate.CandidateAttentionGroups != cfg.NumLayers*width*cfg.NumKVHeads {
					t.Fatalf("receipts dispatches=%d groups=%d", candidate.CandidateAttentionDispatches, candidate.CandidateAttentionGroups)
				}
				if baseline.CandidateAttentionDispatches != 0 {
					t.Fatal("default route selected candidate")
				}
				kd := cfg.NumKVHeads * cfg.HeadDim
				coldCache, err := eng.NewBatchedAttentionCache(cfg, width)
				if err != nil {
					t.Fatal(err)
				}
				coldPacked := packTestPrompts(t, cfg, history)
				cold, err := RunSegmentedLayerChain(t.Context(), eng, cfg, coldCache, coldPacked)
				if err != nil {
					t.Fatal(err)
				}
				coldTail, _ := eng.NewBatchedModelTail(cfg, width)
				coldResult, err := coldTail.Run(eng, cfg, append([]float32(nil), cold.Hidden...), nil)
				coldTail.Free()
				if err != nil {
					t.Fatal(err)
				}
				checkVector(t, "cold logits", candidateResult.Logits, coldResult.Logits, 8e-4)
				for layer := range cfg.NumLayers {
					for row := range width {
						end := (row*cfg.MaxSeqLen + positions[row] + 1) * kd
						start := end - (positions[row]-len(prefixes[row])+1)*kd
						checkVector(t, "K", candidateCache.K[layer].HostSlice()[start:end], baselineCache.K[layer].HostSlice()[start:end], 1e-6)
						checkVector(t, "V", candidateCache.V[layer].HostSlice()[start:end], baselineCache.V[layer].HostSlice()[start:end], 1e-6)
						actualK := make([][]float32, cfg.NumLayers)
						actualV := make([][]float32, cfg.NumLayers)
						for l := range cfg.NumLayers {
							actualK[l] = make([]float32, len(history[row])*kd)
							actualV[l] = make([]float32, len(history[row])*kd)
						}
						if err := leases[row].lease.Restore(actualK, actualV, len(prefixes[row])); err != nil {
							t.Fatal(err)
						}
						copy(actualK[layer][len(prefixes[row])*kd:], candidateCache.K[layer].HostSlice()[start:end])
						copy(actualV[layer][len(prefixes[row])*kd:], candidateCache.V[layer].HostSlice()[start:end])
						coldBase := row * cfg.MaxSeqLen * kd
						checkVector(t, "cold K", actualK[layer], coldCache.K[layer].HostSlice()[coldBase:coldBase+len(history[row])*kd], 4e-5)
						checkVector(t, "cold V", actualV[layer], coldCache.V[layer].HostSlice()[coldBase:coldBase+len(history[row])*kd], 4e-5)
					}
				}
				coldCache.Free()
				if candidateResult.Tokens[rowFirstMismatch(candidateResult.Tokens, baselineResult.Tokens)] != baselineResult.Tokens[rowFirstMismatch(candidateResult.Tokens, baselineResult.Tokens)] {
					t.Fatalf("tokens differ: %v %v", candidateResult.Tokens, baselineResult.Tokens)
				}
				tokens = append(tokens[:0], candidateResult.Tokens...)
				for i := range positions {
					positions[i]++
				}
				t.Logf("step=%d B=%d full_logits=%d hidden=%d kv_coordinates=%d chain_dispatches=%d candidate_attention_dispatches=%d candidate_groups=%d", step, width, len(candidateResult.Logits), len(candidateHidden), width*cfg.NumLayers*(positions[0]-len(prefixes[0]))*kd*2, candidate.Dispatches, candidate.CandidateAttentionDispatches, candidate.CandidateAttentionGroups)
			}
		})
	}
}

func TestPersistentGQAChainFailureAndLifetime(t *testing.T) {
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
	enablePersistentGQATestRoute(t, eng)
	runtime, err := newProductionPrefixRuntime(eng, cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()
	prefix := []int32{cfg.BOSTokenID, 101, 102}
	source, _ := eng.NewBatchedAttentionCache(cfg, 1)
	defer source.Free()
	packed := packTestPrompts(t, cfg, [][]int32{prefix})
	_, err = RunSegmentedLayerChain(t.Context(), eng, cfg, source, packed)
	if err != nil {
		t.Fatal(err)
	}
	if err := runtime.Put(t.Context(), prefix, 7, source, 0); err != nil {
		t.Fatal(err)
	}
	lease, _, err := runtime.Acquire(append(prefix, 7))
	if err != nil || lease == nil {
		t.Fatal(err)
	}
	defer lease.Release()
	cache, _ := eng.NewBatchedAttentionCache(cfg, 1)
	defer cache.Free()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := runtime.Run(ctx, eng, cfg, cache, []*productionPrefixLease{lease}, []int32{7}, []int{len(prefix)}, []int{0}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel error=%v", err)
	}
	injected := errors.New("test completion")
	eng.testChainCompletionFail = injected
	if _, err := runtime.Run(t.Context(), eng, cfg, cache, []*productionPrefixLease{lease}, []int32{7}, []int{len(prefix)}, []int{0}); !errors.Is(err, injected) {
		t.Fatalf("completion error=%v", err)
	}
	eng.testChainCompletionFail = nil
	pin, err := lease.lease.PinLayer(0)
	if err != nil {
		t.Fatalf("lease after failure: %v", err)
	}
	pin.Release()
}

func checkVector(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s lengths", name)
	}
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > tol {
			t.Fatalf("%s[%d] diff=%g", name, i, d)
		}
	}
}
func itoaTest(v int) string {
	if v == 1 {
		return "1"
	}
	if v == 2 {
		return "2"
	}
	return "4"
}
func rowFirstMismatch(a, b []int32) int {
	for i := range a {
		if a[i] != b[i] {
			return i
		}
	}
	return 0
}
