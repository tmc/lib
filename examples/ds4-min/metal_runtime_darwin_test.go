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
	if got := r.state.layers[2].attn.cap; got != 256 {
		t.Fatalf("layer 2 compressed capacity = %d, want 256", got)
	}
	if got := r.state.layers[3].attn.cap; got != 8 {
		t.Fatalf("layer 3 compressed capacity = %d, want 8", got)
	}
	if got := r.state.layers[2].indexer.width; got != 256 {
		t.Fatalf("layer 2 indexer width = %d, want 256", got)
	}
	if got := r.state.layers[3].indexer.cap; got != 0 {
		t.Fatalf("layer 3 indexer capacity = %d, want 0", got)
	}
	for i, value := range r.state.layers[2].attn.score.HostSlice() {
		if !math.IsInf(float64(value), -1) {
			t.Fatalf("layer 2 attention score %d = %g, want -Inf", i, value)
		}
	}
	if r.output.weight.name != "output.weight" || r.output.norm.name != "output_norm.weight" {
		t.Fatal("output tensors are not bound")
	}
}
