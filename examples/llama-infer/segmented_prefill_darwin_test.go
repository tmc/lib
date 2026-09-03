//go:build darwin

package main

import (
	"context"
	"reflect"
	"testing"
)

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
