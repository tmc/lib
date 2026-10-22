//go:build linux

package main

import (
	"bytes"
	"os"
	"testing"
)

// TestCUDAStateGraphReplayGate exercises the legacy runtime's captured graph
// path across a reset. It is opt-in because it loads the registered model and
// must run under gpu-lock. A reset leaves the compiled graphs in place while
// clearing their backing state; replay must remain bit-identical to the first
// evaluation.
func TestCUDAStateGraphReplayGate(t *testing.T) {
	if os.Getenv("DS4_STATE_GRAPH_GATE") != "1" {
		t.Skip("DS4_STATE_GRAPH_GATE is not set")
	}
	model := os.Getenv("DS4_MODEL")
	if model == "" {
		t.Fatal("DS4_MODEL is required")
	}
	if os.Getenv("DS4_DISABLE_CUDA_GRAPH") == "1" {
		t.Fatal("DS4_DISABLE_CUDA_GRAPH must be unset for graph replay gate")
	}
	r, err := newCUDARuntime(model, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if err := r.Eval(0, 0); err != nil {
		t.Fatalf("initial graph capture/evaluation: %v", err)
	}
	want := append([]float32(nil), r.Logits()...)
	if len(want) == 0 {
		t.Fatal("initial evaluation returned no logits")
	}
	if err := r.Reset(); err != nil {
		t.Fatalf("reset: %v", err)
	}
	if err := r.Eval(0, 0); err != nil {
		t.Fatalf("graph replay evaluation: %v", err)
	}
	if got := r.Logits(); !bytes.Equal(float32Bytes(got), float32Bytes(want)) {
		t.Fatal("graph replay logits differ after reset")
	}
}
