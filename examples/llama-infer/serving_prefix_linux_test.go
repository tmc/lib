//go:build linux && !tpu

package main

// Device qualification for LLAMA_INFER_PAGED_PREFIX=1.
//
// TestPagedPrefixServingReuse is the test prefix_cache_paged.go names: a
// second request that begins with a finished request's tokens must be
// prefilled from the stored blocks and produce the token a backend without
// the cache produces for the same prompt, and the pool must be whole after
// every sequence has left and the cache has been reset.
//
// It needs a CUDA device and LIB_CUDA_HARDWARE_TEST=1, like the other hardware
// tests here, and has never been executed by its author. A skip is not a pass.

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/examples/internal/continuousbatch"
)

func TestPagedPrefixServingReuse(t *testing.T) {
	requireCUDAHardware(t)
	cfg := Config{
		Name:       "PagedPrefixTestModel",
		Dim:        256,
		HiddenDim:  512,
		NumLayers:  2,
		NumHeads:   4,
		NumKVHeads: 2,
		HeadDim:    64,
		VocabSize:  256,
		MaxSeqLen:  64,
		Eps:        1e-5,
		RopeTheta:  10000.0,
		Precision:  "fp32",
		TestMode:   true,
	}
	t.Setenv("LLAMA_INFER_CUDA_GRAPH", "")
	t.Setenv("LLAMA_INFER_PAGED_KV", "1")
	t.Setenv("LLAMA_INFER_PAGED_KV_SEQS", "4")
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("paged engine: %v", err)
	}
	if eng.paged == nil {
		t.Fatal("engine is not paged")
	}
	blockSize := eng.paged.cfg.BlockSize
	freeAtStart := eng.paged.cache.NumFreeBlocks()

	newBackend := func(t *testing.T, prefix string) *cudaServingBackend {
		t.Helper()
		t.Setenv("LLAMA_INFER_PAGED_PREFIX", prefix)
		b, err := newPlatformServingBackend(eng, cfg, 2)
		if err != nil {
			t.Fatalf("backend with LLAMA_INFER_PAGED_PREFIX=%q: %v", prefix, err)
		}
		cb, ok := b.(*cudaServingBackend)
		if !ok {
			t.Fatalf("backend is %T, want *cudaServingBackend", b)
		}
		t.Cleanup(func() { cb.Close() })
		return cb
	}
	cached := newBackend(t, "1")
	plain := newBackend(t, "")
	if _, ok := cached.prefixCacheMetrics(); !ok {
		t.Fatal("backend with the switch reports no prefix cache")
	}
	if _, ok := plain.prefixCacheMetrics(); ok {
		t.Fatal("backend without the switch reports a prefix cache")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// First request: a prompt of two whole blocks and a fraction, then two
	// decode steps, so the stored sequence has generated tokens in it.
	prompt := make([]int32, 2*blockSize+3)
	for i := range prompt {
		prompt[i] = int32(10 + i)
	}
	first := &continuousbatch.State{ID: "first", PromptTokenIDs: prompt}
	tok, err := cached.Prefill(ctx, first)
	if err != nil {
		t.Fatalf("first prefill: %v", err)
	}
	written := append([]int32(nil), prompt...)
	first.Current = tok.ID
	for step := 0; step < 2; step++ {
		res := cached.DecodeBatch(ctx, []*continuousbatch.State{first})[0]
		if res.Err != nil {
			t.Fatalf("first decode step %d: %v", step, res.Err)
		}
		written = append(written, first.Current)
		first.Position++
		first.Current = res.Token.ID
	}
	cached.Release(first)
	m, _ := cached.prefixCacheMetrics()
	if m.Entries != 1 || m.Hits != 0 || m.Misses != 1 {
		t.Fatalf("after the first request: metrics %+v, want one entry, one miss", m)
	}
	wantBytes := int64(len(written)/blockSize) * eng.paged.blockBytes()
	if m.Bytes != wantBytes {
		t.Fatalf("after the first request: %d bytes retained, want %d (%d whole blocks)", m.Bytes, wantBytes, len(written)/blockSize)
	}

	// Second request: everything the first wrote plus a tail. The cache must
	// hand over the whole blocks and prefill only the rest.
	second := append(append([]int32(nil), written...), 200, 201, 202)
	reuse := &continuousbatch.State{ID: "second", PromptTokenIDs: second}
	got, err := cached.Prefill(ctx, reuse)
	if err != nil {
		t.Fatalf("second prefill: %v", err)
	}
	m, _ = cached.prefixCacheMetrics()
	wantSaved := int64((len(second) - 1) / blockSize * blockSize)
	if m.Hits != 1 || m.Misses != 1 || m.SavedTokens != wantSaved {
		t.Fatalf("after the second request: metrics %+v, want one hit and %d saved tokens", m, wantSaved)
	}
	if reuse.Position != len(second) {
		t.Fatalf("second request position %d, want %d", reuse.Position, len(second))
	}

	ref := &continuousbatch.State{ID: "reference", PromptTokenIDs: second}
	want, err := plain.Prefill(ctx, ref)
	if err != nil {
		t.Fatalf("reference prefill: %v", err)
	}
	if got.ID != want.ID {
		t.Fatalf("prefill over the stored prefix chose token %d, the full prefill chose %d", got.ID, want.ID)
	}

	// The reused sequence decodes on: its later blocks are its own, the
	// shared ones must not be written. Both backends must agree.
	reuse.Current = got.ID
	ref.Current = want.ID
	for step := 0; step < 3; step++ {
		a := cached.DecodeBatch(ctx, []*continuousbatch.State{reuse})[0]
		b := plain.DecodeBatch(ctx, []*continuousbatch.State{ref})[0]
		if a.Err != nil || b.Err != nil {
			t.Fatalf("decode step %d: cached %v, plain %v", step, a.Err, b.Err)
		}
		if a.Token.ID != b.Token.ID {
			t.Fatalf("decode step %d: cached token %d, plain token %d", step, a.Token.ID, b.Token.ID)
		}
		reuse.Position++
		reuse.Current = a.Token.ID
		ref.Position++
		ref.Current = b.Token.ID
	}
	cached.Release(reuse)
	plain.Release(ref)
	if err := eng.paged.CheckStatus(); err != nil {
		t.Fatal(err)
	}

	// Retained blocks are the only thing the pool is missing; reset returns
	// them all.
	m, _ = cached.prefixCacheMetrics()
	if free := eng.paged.cache.NumFreeBlocks(); int64(freeAtStart-free)*eng.paged.blockBytes() != m.Bytes {
		t.Fatalf("pool is short %d blocks with %d bytes retained", freeAtStart-free, m.Bytes)
	}
	if err := cached.resetBenchmarkPrefixes(); err != nil {
		t.Fatalf("reset: %v", err)
	}
	if free := eng.paged.cache.NumFreeBlocks(); free != freeAtStart {
		t.Fatalf("pool has %d free blocks after reset, had %d before the test", free, freeAtStart)
	}
	m, _ = cached.prefixCacheMetrics()
	if m.Entries != 0 || m.Bytes != 0 {
		t.Fatalf("after reset: metrics %+v, want nothing retained", m)
	}
}

// TestPagedPrefixRefusesDenseEngine pins the switch to paged engines. The
// backend constructor refuses before touching the device, but building the
// engine needs one.
func TestPagedPrefixRefusesDenseEngine(t *testing.T) {
	requireCUDAHardware(t)
	cfg := Config{Name: "PagedPrefixDenseRefusal", Dim: 64, HiddenDim: 128, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 64, VocabSize: 64, MaxSeqLen: 16, Eps: 1e-5, RopeTheta: 10000.0, Precision: "fp32", TestMode: true}
	t.Setenv("LLAMA_INFER_CUDA_GRAPH", "")
	t.Setenv("LLAMA_INFER_PAGED_KV", "")
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("dense engine: %v", err)
	}
	t.Setenv("LLAMA_INFER_PAGED_PREFIX", "1")
	_, err = newPlatformServingBackend(eng, cfg, 1)
	if err == nil || !strings.Contains(err.Error(), "LLAMA_INFER_PAGED_KV=1") {
		t.Fatalf("prefix switch on a dense engine: err = %v, want the paged-engine refusal", err)
	}
}
