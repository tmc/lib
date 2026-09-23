//go:build darwin

package main

import (
	"context"
	"os"
	"reflect"
	"testing"

	"github.com/tmc/lib/examples/internal/continuousbatch"
)

func TestSegmentedLayerChainRealFP16(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.Precision = "f16"
	cfg.MaxSeqLen = 128
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	gotCache, err := eng.NewBatchedAttentionCache(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer gotCache.Free()
	wantCache, err := eng.NewBatchedAttentionCache(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer wantCache.Free()
	segments := []PrefillSegment{
		{RequestID: "short", Row: 1, Tokens: []int32{128000}},
		{RequestID: "long", Row: 0, Tokens: []int32{128000, 22691}},
	}
	var packer PrefillPacker
	packed, err := packer.Pack(t.Context(), segments, 3, 2, cfg.VocabSize, cfg.MaxSeqLen)
	if err != nil {
		t.Fatal(err)
	}
	tail, err := eng.NewBatchedModelTail(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer tail.Free()
	chain, err := RunSegmentedLayerChain(t.Context(), eng, cfg, gotCache, packed)
	if err != nil {
		t.Fatal(err)
	}
	gotHidden := append([]float32(nil), chain.Hidden...)
	got, err := RunSegmentedModelTailFromChain(eng, cfg, tail, chain, nil)
	if err != nil {
		t.Fatal(err)
	}
	gotDiagnostic, err := tail.Run(eng, cfg, gotHidden, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Tokens, gotDiagnostic.Tokens) {
		t.Fatalf("tiled tokens=%v diagnostic tokens=%v", got.Tokens, gotDiagnostic.Tokens)
	}
	wantLogits := make([]float32, 0, 2*cfg.VocabSize)
	wantTokens := make([]int32, 0, 2)
	for _, segment := range segments {
		var result BatchedModelTailResult
		for i, token := range segment.Tokens {
			one, err := RunBatchedLayerChain(eng, cfg, wantCache, []int32{token}, []int{i}, []int{segment.Row})
			if err != nil {
				t.Fatal(err)
			}
			if i == len(segment.Tokens)-1 {
				result, err = tail.Run(eng, cfg, append([]float32(nil), one.Hidden...), nil)
				if err != nil {
					t.Fatal(err)
				}
			}
		}
		wantLogits = append(wantLogits, result.Logits...)
		wantTokens = append(wantTokens, result.Tokens...)
	}
	checkClose(t, "real logits", gotDiagnostic.Logits, wantLogits, 5e-4)
	if !reflect.DeepEqual(gotDiagnostic.Tokens, wantTokens) {
		t.Fatalf("tokens=%v want %v", gotDiagnostic.Tokens, wantTokens)
	}
	for layer := range cfg.NumLayers {
		checkClose(t, "real k cache", gotCache.K[layer].HostSlice(), wantCache.K[layer].HostSlice(), 5e-4)
		checkClose(t, "real v cache", gotCache.V[layer].HostSlice(), wantCache.V[layer].HostSlice(), 5e-4)
	}
	t.Logf("tokens=%v logits=%d layers=%d", got.Tokens, len(gotDiagnostic.Logits), cfg.NumLayers)
}

func TestScheduledSegmentedPrefillRealFP16(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.Name += "-scheduled-segmented-test"
	cfg.Precision = "f16"
	cfg.MaxSeqLen = 128
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	eng.prefixes = newPrefixCache(64<<20, 8)
	adapter, err := newScheduledEngineAdapter(eng, cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	states := []continuousbatch.State{
		{ID: "short", PromptTokenIDs: []int32{128000}},
		{ID: "long", PromptTokenIDs: []int32{128000, 22691}},
	}
	inputs := make([]continuousbatch.PrefillRequest, len(states))
	for i := range states {
		inputs[i] = continuousbatch.PrefillRequest{Context: t.Context(), State: &states[i]}
	}
	results := adapter.PrefillBatch(inputs)
	gotTokens := make([]int32, len(results))
	for i, result := range results {
		if result.Err != nil {
			t.Fatal(result.Err)
		}
		gotTokens[i] = result.Token.ID
	}
	gotHidden := append([]float32(nil), eng.prefillNormXMat.HostSlice()[:len(states)*cfg.Dim]...)
	gotDiagnostic, err := adapter.tail.Run(eng, cfg, gotHidden, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotTokens, gotDiagnostic.Tokens) {
		t.Fatalf("tiled tokens=%v diagnostic tokens=%v", gotTokens, gotDiagnostic.Tokens)
	}
	gotLogits := gotDiagnostic.Logits

	wantCache, err := eng.NewBatchedAttentionCache(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer wantCache.Free()
	wantTail, err := eng.NewBatchedModelTail(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer wantTail.Free()
	wantTokens := make([]int32, 0, len(states))
	wantLogits := make([]float32, 0, len(states)*cfg.VocabSize)
	for row, state := range states {
		var tail BatchedModelTailResult
		for position, token := range state.PromptTokenIDs {
			chain, err := RunBatchedLayerChain(eng, cfg, wantCache, []int32{token}, []int{position}, []int{row})
			if err != nil {
				t.Fatal(err)
			}
			if position == len(state.PromptTokenIDs)-1 {
				tail, err = wantTail.Run(eng, cfg, append([]float32(nil), chain.Hidden...), nil)
				if err != nil {
					t.Fatal(err)
				}
			}
		}
		wantTokens = append(wantTokens, tail.Tokens...)
		wantLogits = append(wantLogits, tail.Logits...)
	}
	if !reflect.DeepEqual(gotTokens, wantTokens) {
		t.Fatalf("tokens=%v want %v", gotTokens, wantTokens)
	}
	checkClose(t, "scheduled real logits", gotLogits, wantLogits, 5e-4)
	for layer := range cfg.NumLayers {
		checkClose(t, "scheduled real k cache", adapter.cache.K[layer].HostSlice(), wantCache.K[layer].HostSlice(), 5e-4)
		checkClose(t, "scheduled real v cache", adapter.cache.V[layer].HostSlice(), wantCache.V[layer].HostSlice(), 5e-4)
	}
	t.Logf("tokens=%v logits=%d layers=%d prefill=%+v", gotTokens, len(gotLogits), cfg.NumLayers, adapter.PrefillObservations())
}

func TestSegmentedLayerChainMatchesSequentialModel(t *testing.T) {
	for _, tc := range []struct {
		name     string
		segments []PrefillSegment
		maxSeq   int
	}{
		{"b1", []PrefillSegment{{RequestID: "a", Row: 1, Tokens: []int32{3, 4, 5}}}, 130},
		{"b2-unequal", []PrefillSegment{{RequestID: "a", Row: 2, Tokens: []int32{3}}, {RequestID: "b", Row: 0, Tokens: []int32{4, 5, 6}}}, 130},
		{"b4-unequal", []PrefillSegment{{RequestID: "a", Row: 3, Tokens: []int32{1}}, {RequestID: "b", Row: 0, Tokens: []int32{2, 3}}, {RequestID: "c", Row: 2, Tokens: []int32{4, 5, 6}}, {RequestID: "d", Row: 1, Tokens: []int32{7, 8, 9, 10}}}, 130},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultTinyLlamaConfig()
			cfg.MaxSeqLen = tc.maxSeq
			eng, err := GetOrCompileEngine(cfg)
			if err != nil {
				t.Fatal(err)
			}
			gotCache, err := eng.NewBatchedAttentionCache(cfg, 4)
			if err != nil {
				t.Fatal(err)
			}
			defer gotCache.Free()
			wantCache, err := eng.NewBatchedAttentionCache(cfg, 4)
			if err != nil {
				t.Fatal(err)
			}
			defer wantCache.Free()
			var packer PrefillPacker
			packed, err := packer.Pack(t.Context(), tc.segments, 16, 4, cfg.VocabSize, cfg.MaxSeqLen)
			if err != nil {
				t.Fatal(err)
			}
			tail, err := eng.NewBatchedModelTail(cfg, 4)
			if err != nil {
				t.Fatal(err)
			}
			defer tail.Free()
			gotChain, err := RunSegmentedLayerChain(t.Context(), eng, cfg, gotCache, packed)
			if err != nil {
				t.Fatal(err)
			}
			gotHidden := append([]float32(nil), gotChain.Hidden...)
			got, err := RunSegmentedModelTailFromChain(eng, cfg, tail, gotChain, nil)
			if err != nil {
				t.Fatal(err)
			}
			gotDiagnostic, err := tail.Run(eng, cfg, gotHidden, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.Tokens, gotDiagnostic.Tokens) {
				t.Fatalf("tiled tokens=%v diagnostic tokens=%v", got.Tokens, gotDiagnostic.Tokens)
			}
			wantLogits := make([]float32, 0, len(tc.segments)*cfg.VocabSize)
			wantTokens := make([]int32, 0, len(tc.segments))
			wantHidden := make([]float32, 0, len(tc.segments)*cfg.Dim)
			for _, segment := range tc.segments {
				var result BatchedModelTailResult
				var chain BatchedLayerChainResult
				for i, token := range segment.Tokens {
					chain, err = RunBatchedLayerChain(eng, cfg, wantCache, []int32{token}, []int{segment.AbsoluteStart + i}, []int{segment.Row})
					if err != nil {
						t.Fatal(err)
					}
					if i == len(segment.Tokens)-1 {
						result, err = tail.Run(eng, cfg, append([]float32(nil), chain.Hidden...), nil)
						if err != nil {
							t.Fatal(err)
						}
					}
				}
				wantLogits = append(wantLogits, result.Logits...)
				wantTokens = append(wantTokens, result.Tokens...)
				wantHidden = append(wantHidden, chain.Hidden...)
			}
			checkClose(t, "tail hidden", gotHidden, wantHidden, 3e-4)
			checkClose(t, "logits", gotDiagnostic.Logits, wantLogits, 3e-4)
			if !reflect.DeepEqual(got.Tokens, wantTokens) {
				t.Fatalf("tokens=%v want %v", got.Tokens, wantTokens)
			}
			for layer := range cfg.NumLayers {
				checkClose(t, "k cache", gotCache.K[layer].HostSlice(), wantCache.K[layer].HostSlice(), 3e-4)
				checkClose(t, "v cache", gotCache.V[layer].HostSlice(), wantCache.V[layer].HostSlice(), 3e-4)
			}
		})
	}
}

func TestSegmentedLayerChainRestoredPrefixBoundary(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.MaxSeqLen = 132
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	gotCache, err := eng.NewBatchedAttentionCache(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer gotCache.Free()
	wantCache, err := eng.NewBatchedAttentionCache(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer wantCache.Free()
	for pos := range 127 {
		token := int32(pos%11 + 1)
		for _, cache := range []*BatchedAttentionCache{gotCache, wantCache} {
			if _, err := RunBatchedLayerChain(eng, cfg, cache, []int32{token}, []int{pos}, []int{2}); err != nil {
				t.Fatalf("prefix position %d: %v", pos, err)
			}
		}
	}
	segments := []PrefillSegment{
		{RequestID: "boundary", Row: 2, Tokens: []int32{12, 13}, AbsoluteStart: 127, RestoredPrefix: 127},
		{RequestID: "fresh", Row: 0, Tokens: []int32{14, 15, 16}, AbsoluteStart: 0},
	}
	var packer PrefillPacker
	packed, err := packer.Pack(t.Context(), segments, 5, 3, cfg.VocabSize, cfg.MaxSeqLen)
	if err != nil {
		t.Fatal(err)
	}
	got, err := RunSegmentedLayerChain(t.Context(), eng, cfg, gotCache, packed)
	if err != nil {
		t.Fatal(err)
	}
	gotHidden := append([]float32(nil), got.Hidden...)
	wantHidden := make([]float32, 0, 2*cfg.Dim)
	for _, segment := range segments {
		var chain BatchedLayerChainResult
		for i, token := range segment.Tokens {
			chain, err = RunBatchedLayerChain(eng, cfg, wantCache, []int32{token}, []int{segment.AbsoluteStart + i}, []int{segment.Row})
			if err != nil {
				t.Fatal(err)
			}
		}
		wantHidden = append(wantHidden, chain.Hidden...)
	}
	checkClose(t, "boundary hidden", gotHidden, wantHidden, 3e-4)
	for layer := range cfg.NumLayers {
		checkClose(t, "boundary k cache", gotCache.K[layer].HostSlice(), wantCache.K[layer].HostSlice(), 3e-4)
		checkClose(t, "boundary v cache", gotCache.V[layer].HostSlice(), wantCache.V[layer].HostSlice(), 3e-4)
	}
}

func TestSegmentedLayerChainRejectsBeforeCommit(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	cache, err := eng.NewBatchedAttentionCache(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Free()
	before := append([]float32(nil), cache.K[0].HostSlice()...)
	bad := PackedPrefill{Tokens: []int32{1}, RequestIndex: []uint32{0}, AbsolutePositions: []uint32{1}, Offsets: []uint32{0}, Lengths: []uint32{1}, Rows: []uint32{0}, RestoredPrefixes: []uint32{0}, RequestIDs: []string{"bad"}}
	if _, err := RunSegmentedLayerChain(t.Context(), eng, cfg, cache, bad); err == nil {
		t.Fatal("malformed pack succeeded")
	}
	badToken := PackedPrefill{Tokens: []int32{int32(cfg.VocabSize)}, RequestIndex: []uint32{0}, AbsolutePositions: []uint32{0}, Offsets: []uint32{0}, Lengths: []uint32{1}, Rows: []uint32{0}, RestoredPrefixes: []uint32{0}, RequestIDs: []string{"bad-token"}}
	if _, err := RunSegmentedLayerChain(t.Context(), eng, cfg, cache, badToken); err == nil {
		t.Fatal("out-of-range token succeeded")
	}
	var packer PrefillPacker
	packed, err := packer.Pack(t.Context(), []PrefillSegment{{RequestID: "valid", Row: 0, Tokens: []int32{1}}}, 1, 2, cfg.VocabSize, cfg.MaxSeqLen)
	if err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := RunSegmentedLayerChain(canceled, eng, cfg, cache, packed); err == nil {
		t.Fatal("canceled pack succeeded")
	}
	if !reflect.DeepEqual(before, cache.K[0].HostSlice()) {
		t.Fatal("rejected chain mutated KV")
	}
}

func TestSegmentedAttentionMatchesSequentialRows(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.MaxSeqLen = 130
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	segmented, err := eng.NewBatchedAttentionCache(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer segmented.Free()
	sequential, err := eng.NewBatchedAttentionCache(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer sequential.Free()
	fillAttentionCache(segmented, cfg)
	for layer := range sequential.K {
		copy(sequential.K[layer].HostSlice(), segmented.K[layer].HostSlice())
		copy(sequential.V[layer].HostSlice(), segmented.V[layer].HostSlice())
	}
	var packer PrefillPacker
	packed, err := packer.Pack(t.Context(), []PrefillSegment{
		{RequestID: "long", Row: 2, Tokens: []int32{1, 2, 3}, AbsoluteStart: 125, RestoredPrefix: 125},
		{RequestID: "short", Row: 0, Tokens: []int32{4, 5}, AbsoluteStart: 0},
	}, 5, 3, cfg.VocabSize, cfg.MaxSeqLen)
	if err != nil {
		t.Fatal(err)
	}
	qd, kd := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	q, k, v := make([]float32, 5*qd), make([]float32, 5*kd), make([]float32, 5*kd)
	for i := range q {
		q[i] = float32(i%23-11) / 41
	}
	for i := range k {
		k[i] = float32(i%19-9) / 37
	}
	for i := range v {
		v[i] = float32(i%17-8) / 31
	}
	got, err := RunSegmentedAttention(t.Context(), eng, cfg, 0, segmented, packed, q, k, v)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 0, len(got.Attention))
	for request := range packed.Offsets {
		offset, length := int(packed.Offsets[request]), int(packed.Lengths[request])
		for i := range length {
			token := offset + i
			one, err := RunBatchedAttention(eng, cfg, 0, sequential,
				q[token*qd:(token+1)*qd], k[token*kd:(token+1)*kd], v[token*kd:(token+1)*kd],
				[]int{int(packed.AbsolutePositions[token])}, []int{int(packed.Rows[request])})
			if err != nil {
				t.Fatal(err)
			}
			want = append(want, one.Attention...)
		}
	}
	checkClose(t, "segmented attention", got.Attention, want, 2e-3)
	checkClose(t, "segmented k cache", segmented.K[0].HostSlice(), sequential.K[0].HostSlice(), 5e-4)
	checkClose(t, "segmented v cache", segmented.V[0].HostSlice(), sequential.V[0].HostSlice(), 0)
}

func TestSegmentedAttentionRejectsBeforeKVMutation(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	cache, err := eng.NewBatchedAttentionCache(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Free()
	before := append([]float32(nil), cache.K[0].HostSlice()...)
	malformed := PackedPrefill{Tokens: []int32{1}, RequestIndex: []uint32{0}, AbsolutePositions: []uint32{1}, Offsets: []uint32{0}, Lengths: []uint32{1}, Rows: []uint32{0}, RestoredPrefixes: []uint32{0}, RequestIDs: []string{"x"}}
	qd, kd := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	if _, err := RunSegmentedAttention(t.Context(), eng, cfg, 0, cache, malformed, make([]float32, qd), make([]float32, kd), make([]float32, kd)); err == nil {
		t.Fatal("malformed boundary succeeded")
	}
	if !reflect.DeepEqual(before, cache.K[0].HostSlice()) {
		t.Fatal("invalid pack mutated KV")
	}
	var packer PrefillPacker
	if _, err := packer.Pack(t.Context(), []PrefillSegment{{RequestID: "empty", Row: 0}}, 1, 2, cfg.VocabSize, cfg.MaxSeqLen); err == nil {
		t.Fatal("empty suffix succeeded")
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	valid, err := packer.Pack(t.Context(), []PrefillSegment{{RequestID: "valid", Row: 0, Tokens: []int32{1}}}, 1, 2, cfg.VocabSize, cfg.MaxSeqLen)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RunSegmentedAttention(canceled, eng, cfg, 0, cache, valid, make([]float32, qd), make([]float32, kd), make([]float32, kd)); err == nil {
		t.Fatal("canceled submission succeeded")
	}
	if !reflect.DeepEqual(before, cache.K[0].HostSlice()) {
		t.Fatal("canceled submission mutated KV")
	}
}
