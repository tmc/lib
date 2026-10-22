//go:build linux

package main

import (
	"errors"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Invalid batches must fail before evaluating a valid prefix. This runtime has
// no device, so reaching an encoder would also expose a preflight regression.
func TestCUDASessionBatchPreflight(t *testing.T) {
	for _, tt := range []struct {
		name     string
		tokens   []int
		position int
	}{
		{"negative suffix", []int{1, -1}, 3},
		{"large suffix", []int{1, 8}, 3},
		{"context overflow", []int{1, 2, 3}, 3},
		{"wrong position", []int{1}, 2},
		{"negative position", []int{1}, -1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := cudaRuntime{
				arena:  new(cudamem.Arena),
				model:  &cudaModel{},
				ctx:    5,
				pos:    3,
				logits: []float32{0.25, -0.5, 1.75},
			}
			r.model.output.embedding.dimensions = []uint64{4096, 8}
			r.state.layers[2].attn.n = 1
			want := slices.Clone(r.logits)
			if err := r.EvalBatch(tt.tokens, tt.position); err == nil {
				t.Fatal("accepted invalid batch")
			}
			if r.pos != 3 || r.failed != nil || r.state.layers[2].attn.n != 1 {
				t.Fatalf("preflight changed timeline: pos=%d failed=%v compressed rows=%d", r.pos, r.failed, r.state.layers[2].attn.n)
			}
			if !slices.Equal(r.Logits(), want) {
				t.Fatalf("preflight changed logits: got %v, want %v", r.Logits(), want)
			}
		})
	}
}

func TestCUDASessionBatchUnavailable(t *testing.T) {
	failed := errors.New("prior evaluation failed")
	for _, r := range []*cudaRuntime{nil, {}, {arena: new(cudamem.Arena), failed: failed}} {
		if err := r.EvalBatch(nil, 0); err == nil {
			t.Fatal("accepted batch on unavailable runtime")
		}
	}
}
