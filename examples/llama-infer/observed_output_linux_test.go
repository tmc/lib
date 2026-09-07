//go:build linux && !tpu

package main

import (
	"strings"
	"testing"

	"github.com/tmc/lib/examples/internal/continuousbatch"
	"github.com/tmc/lib/examples/internal/inferbench"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

func TestCUDAObservedOutputPreservesCompactedMembership(t *testing.T) {
	c := newObservedOutputCollector()
	states := []*continuousbatch.State{{ID: "request-0"}, {ID: "request-2"}}
	logits := []float32{1, 2, 3, 4, 5, 6}
	if err := c.observeLogits(states, logits, 3); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"request-0", "request-2"} {
		record := c.records[id]
		record.PopulatedKVSHA256 = hashFloat32([]float32{1})
		record.PopulatedKVElements = 2
		record.PopulatedTokens = 1
		record.Layers = []inferbench.ObservedKVLayer{{Layer: 0, KeySHA256: hashFloat32([]float32{1}), ValueSHA256: hashFloat32([]float32{2}), ElementsEach: 1}}
		c.records[id] = record
	}
	out, err := c.outputs([]string{"request-2", "request-0"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := out[0].FullLogitsSHA256, hashFloat32(logits[3:]); got != want {
		t.Fatalf("compacted request 2 logits = %q, want %q", got, want)
	}
	if got, want := out[1].FullLogitsSHA256, hashFloat32(logits[:3]); got != want {
		t.Fatalf("compacted request 0 logits = %q, want %q", got, want)
	}
}

func TestCUDAObservedOutputRejectsIncompleteRequest(t *testing.T) {
	c := newObservedOutputCollector()
	if _, err := c.outputs([]string{"canceled"}); err == nil {
		t.Fatal("incomplete canceled request succeeded")
	}
}

func TestCUDAObservedOutputRejectsCanceledPosition(t *testing.T) {
	c := newObservedOutputCollector()
	cache := &SequenceKVCache{Cache: new(cudakv.Cache)}
	sequence := cudaSequenceState{cache: cache, generation: 1}
	state := &continuousbatch.State{ID: "canceled", Position: 2}
	if err := c.observeCUDALogits([]*continuousbatch.State{state}, []cudaSequenceState{sequence}, []float32{1}, 1); err != nil {
		t.Fatal(err)
	}
	c.observeCUDAKV(state, sequence, Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1, MaxSeqLen: 4})
	if _, err := c.outputs([]string{state.ID}); err == nil || !strings.Contains(err.Error(), "stale or mismatched CUDA membership") {
		t.Fatalf("canceled position error = %v", err)
	}
}
