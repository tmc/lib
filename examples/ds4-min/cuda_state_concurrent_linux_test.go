//go:build linux

package main

import (
	"bytes"
	"os"
	"sync"
	"testing"
)

// TestCUDAIndependentGraphConcurrencyGate compares simultaneously evaluated
// request-local states with their serial controls, including a graph replay.
// It requires the registered model and an externally held gpu-lock.
func TestCUDAIndependentGraphConcurrencyGate(t *testing.T) {
	if os.Getenv("DS4_STATE_CONCURRENT_GATE") != "1" {
		t.Skip("DS4_STATE_CONCURRENT_GATE is not set")
	}
	model := os.Getenv("DS4_MODEL")
	if model == "" || os.Getenv("DS4_DISABLE_CUDA_GRAPH") == "1" {
		t.Fatal("DS4_MODEL is required and graphs must be enabled")
	}
	r, err := newCUDARuntime(model, 128)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	var states [4]*cudaRuntime
	var want [4][]float32
	for i := range states {
		states[i], err = r.newIndependentState(128)
		if err != nil {
			t.Fatal(err)
		}
		defer states[i].Close()
		for pos := range 2 {
			if err := states[i].Eval(i+1, pos); err != nil {
				t.Fatal(err)
			}
		}
		want[i] = append([]float32(nil), states[i].Logits()...)
		if err := states[i].Close(); err != nil {
			t.Fatal(err)
		}
		// Fresh states force overlapping first capture, not only replay of
		// graphs captured serially by the control evaluations.
		states[i], err = r.newIndependentState(128)
		if err != nil {
			t.Fatal(err)
		}
		defer states[i].Close()
	}
	var wg sync.WaitGroup
	start := make(chan struct{})
	errs := make(chan error, len(states))
	for i, state := range states {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for pos := range 2 {
				if err := state.Eval(i+1, pos); err != nil {
					errs <- err
					return
				}
			}
		}()
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	for i, state := range states {
		qkv, ffn := 0, 0
		for _, graph := range state.graphs {
			if graph.qkv != nil {
				qkv++
			}
			if graph.ffn != nil {
				ffn++
			}
		}
		if qkv != cudaLayers || ffn != cudaLayers-3 {
			t.Fatalf("state %d capture witness: qkv=%d ffn=%d", i, qkv, ffn)
		}
		t.Logf("state %d: %d QKV and %d FFN graphs, serial logits identical", i, qkv, ffn)
		if !bytes.Equal(float32Bytes(state.Logits()), float32Bytes(want[i])) {
			t.Fatalf("state %d differs from serial control", i)
		}
	}
}
