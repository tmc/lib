//go:build darwin && arm64

package main

import (
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestMetalStateAdapterSpeculationFailsClosed(t *testing.T) {
	a := &metalStateAdapter{}
	verifier, err := a.NewSpeculativeVerifier(nil)
	if verifier != nil {
		t.Fatal("NewSpeculativeVerifier returned a verifier for an incomplete adapter")
	}
	if !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("NewSpeculativeVerifier error = %v, want dsv4.ErrUnimplemented", err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := a.NewSpeculativeVerifier(nil); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("NewSpeculativeVerifier after Close = %v, want dsv4.ErrClosed", err)
	}
}

func TestMetalSpeculativeFrontierRequiresCompleteFiniteLogits(t *testing.T) {
	logits := make([]float32, Flash.Vocab)
	logits[123] = 1
	if got, err := metalSpeculativeFrontier(logits); err != nil || got != 123 {
		t.Fatalf("top=%d err=%v", got, err)
	}
	for _, values := range [][]float32{nil, logits[:len(logits)-1]} {
		if _, err := metalSpeculativeFrontier(values); err == nil {
			t.Fatal("partial frontier admitted")
		}
	}
	for _, index := range []int{0, Flash.Vocab - 1} {
		for _, poison := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			values := append([]float32(nil), logits...)
			values[index] = poison
			if _, err := metalSpeculativeFrontier(values); err == nil {
				t.Fatal("nonfinite frontier admitted")
			}
		}
	}
}

func TestMetalSpeculativeArgmaxUsesFirstIndexOnTies(t *testing.T) {
	if got := metalSpeculativeArgmax([]float32{1, 3, 3, 2}); got != 1 {
		t.Fatalf("metalSpeculativeArgmax tie = %d, want 1", got)
	}
}
