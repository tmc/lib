//go:build darwin && arm64

package main

import (
	"math"
	"os"
	"testing"
)

func TestMetalRuntimeOwnsPinnedSessionState(t *testing.T) {
	path := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	if path == "" {
		t.Skip("profile model path not set")
	}
	r, err := newMetalRuntime(path, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if got := len(r.layers); got != ds4Layers {
		t.Fatalf("layers = %d, want %d", got, ds4Layers)
	}
	if got := r.state.layers[0].rawKV.Len(); got != ds4RawCap*ds4KVWidth {
		t.Fatalf("raw KV length = %d, want %d", got, ds4RawCap*ds4KVWidth)
	}
	if got := r.state.layers[2].attn.cap; got != 258 {
		t.Fatalf("layer 2 compressed capacity = %d, want 258", got)
	}
	if got := r.state.layers[3].attn.cap; got != 10 {
		t.Fatalf("layer 3 compressed capacity = %d, want 10", got)
	}
	if got := r.state.layers[2].indexer.width; got != 256 {
		t.Fatalf("layer 2 indexer width = %d, want 256", got)
	}
	if got := r.state.layers[3].indexer.cap; got != 0 {
		t.Fatalf("layer 3 indexer capacity = %d, want 0", got)
	}
	if got := r.state.layers[2].attn.cache.Len(); got != 258*ds4KVWidth {
		t.Fatalf("layer 2 attention cache length = %d, want %d", got, 258*ds4KVWidth)
	}
	if got := r.state.layers[2].attn.stateKV.Len(); got != 8*1024 {
		t.Fatalf("layer 2 attention state length = %d, want %d", got, 8*1024)
	}
	if got := r.state.layers[3].attn.stateKV.Len(); got != 128*ds4KVWidth {
		t.Fatalf("layer 3 attention state length = %d, want %d", got, 128*ds4KVWidth)
	}
	for i, value := range r.state.layers[2].attn.stateScore.HostSlice() {
		if !math.IsInf(float64(value), -1) {
			t.Fatalf("layer 2 attention score %d = %g, want -Inf", i, value)
		}
	}
	if r.output.weight.name != "output.weight" || r.output.norm.name != "output_norm.weight" {
		t.Fatal("output tensors are not bound")
	}
}
