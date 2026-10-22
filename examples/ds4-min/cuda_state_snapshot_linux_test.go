//go:build linux

package main

import (
	"bytes"
	"errors"
	"os"
	"sync"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// TestCUDAStateSnapshotRestoreGate checks the contract that matters to
// speculative decoding: after a request snapshot, a rejected evaluation can
// be rolled back and replayed with identical device output. The test is
// opt-in because it loads the registered model and must run under gpu-lock.
func TestCUDAStateSnapshotRestoreGate(t *testing.T) {
	if os.Getenv("DS4_STATE_SNAPSHOT_GATE") != "1" {
		t.Skip("DS4_STATE_SNAPSHOT_GATE is not set")
	}
	model := os.Getenv("DS4_MODEL")
	if model == "" {
		t.Fatal("DS4_MODEL is required")
	}
	r, err := newCUDARuntime(model, 2048)
	if err != nil {
		t.Fatal(err)
	}
	backend := r.stateBackend()
	defer backend.Close()
	state, err := backend.NewState(2048)
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()

	if err := backend.Eval(state, 0, 0); err != nil {
		t.Fatalf("seed evaluation: %v", err)
	}
	snapshot, err := state.Snapshot()
	if err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	defer snapshot.Close()

	if err := backend.Eval(state, 1, 1); err != nil {
		t.Fatalf("first replay evaluation: %v", err)
	}
	want := append([]float32(nil), backend.Logits(state)...)
	if len(want) == 0 {
		t.Fatal("first replay evaluation returned no logits")
	}
	if err := state.Restore(snapshot); err != nil {
		t.Fatalf("restore: %v", err)
	}
	if err := backend.Eval(state, 1, 1); err != nil {
		t.Fatalf("second replay evaluation: %v", err)
	}
	got := backend.Logits(state)
	if !bytes.Equal(float32Bytes(got), float32Bytes(want)) {
		t.Fatalf("replayed logits differ after restore")
	}

	other, err := backend.NewState(2048)
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	if err := other.Restore(snapshot); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("cross-state restore error = %v, want ownership error", err)
	}
}

// TestCUDAStateGraphSnapshotRestoreGate checks that a request-local graph
// remains usable when mutable device state is restored. This is distinct from
// the legacy runtime reset gate: Reset destroys graphs and captures them
// again, while Snapshot/Restore must preserve the request's captured graph
// executions and restore only their inputs.
func TestCUDAStateGraphSnapshotRestoreGate(t *testing.T) {
	if os.Getenv("DS4_STATE_GRAPH_SNAPSHOT_GATE") != "1" {
		t.Skip("DS4_STATE_GRAPH_SNAPSHOT_GATE is not set")
	}
	model := os.Getenv("DS4_MODEL")
	if model == "" {
		t.Fatal("DS4_MODEL is required")
	}
	r, err := newCUDARuntime(model, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	backend := r.stateBackend()
	defer backend.Close()
	state, err := backend.NewState(2048)
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()

	request, ok := state.(*cudaRequestState)
	if !ok || request.runtime == nil {
		t.Fatal("backend returned a non-CUDA request state")
	}
	if err := backend.Eval(state, 0, 0); err != nil {
		t.Fatalf("initial graph evaluation: %v", err)
	}
	graphs := cudaStateGraphCount(request.runtime)
	if graphs == 0 {
		t.Fatal("initial evaluation captured no request-local graphs")
	}
	snapshot, err := state.Snapshot()
	if err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	defer snapshot.Close()
	if err := backend.Eval(state, 1, 1); err != nil {
		t.Fatalf("first replay evaluation: %v", err)
	}
	want := append([]float32(nil), backend.Logits(state)...)
	if err := state.Restore(snapshot); err != nil {
		t.Fatalf("restore: %v", err)
	}
	if got := cudaStateGraphCount(request.runtime); got != graphs {
		t.Fatalf("restore changed captured graph count: got %d, want %d", got, graphs)
	}
	if err := backend.Eval(state, 1, 1); err != nil {
		t.Fatalf("graph replay after restore: %v", err)
	}
	if got := backend.Logits(state); !bytes.Equal(float32Bytes(got), float32Bytes(want)) {
		t.Fatal("request graph replay differs after restore")
	}
}

// TestCUDAStateConcurrentGraphReplayGate checks that request-local graphs can
// replay concurrently without sharing mutable state. Graphs are captured for
// each request before the concurrent section; serial states provide the
// reference logits for the same token streams.
//
// It is opt-in because it loads the registered model and must run under
// gpu-lock. This gate intentionally exercises only independent request
// streams. It does not claim that one request may call Eval concurrently.
func TestCUDAStateConcurrentGraphReplayGate(t *testing.T) {
	if os.Getenv("DS4_STATE_CONCURRENT_GRAPH_GATE") != "1" {
		t.Skip("DS4_STATE_CONCURRENT_GRAPH_GATE is not set")
	}
	model := os.Getenv("DS4_MODEL")
	if model == "" {
		t.Fatal("DS4_MODEL is required")
	}
	r, err := newCUDARuntime(model, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	backend := r.stateBackend()
	defer backend.Close()

	states := make([]dsv4.State, 4)
	for i := range states {
		states[i], err = backend.NewState(2048)
		if err != nil {
			for _, state := range states[:i] {
				_ = state.Close()
			}
			t.Fatal(err)
		}
	}
	defer func() {
		for _, state := range states {
			_ = state.Close()
		}
	}()

	requests := []struct {
		stateIndex int
		tokenBase  int
	}{
		{stateIndex: 0, tokenBase: 0},
		{stateIndex: 1, tokenBase: 17},
	}

	// Capture request-local graphs for both live requests and their serial
	// references before starting concurrent replay. A graph count is the
	// request's immutable execution shape and must not change during replay.
	for _, request := range requests {
		if err := backend.Eval(states[request.stateIndex], request.tokenBase, 0); err != nil {
			t.Fatalf("capture request %d: %v", request.stateIndex, err)
		}
		if err := backend.Eval(states[request.stateIndex+2], request.tokenBase, 0); err != nil {
			t.Fatalf("capture reference %d: %v", request.stateIndex+2, err)
		}
	}
	counts := [4]int{}
	for i, state := range states {
		request, ok := state.(*cudaRequestState)
		if !ok || request.runtime == nil {
			t.Fatalf("state %d is not a CUDA request state", i)
		}
		counts[i] = cudaStateGraphCount(request.runtime)
		if counts[i] == 0 {
			t.Fatalf("state %d captured no request-local graphs", i)
		}
	}

	for position := 1; position < 3; position++ {
		// Advance the serial references first and copy their host logits before
		// running the concurrent requests.
		want := make([][]byte, len(requests))
		for i, request := range requests {
			if err := backend.Eval(states[request.stateIndex+2], request.tokenBase+position, position); err != nil {
				t.Fatalf("serial reference %d position %d: %v", request.stateIndex+2, position, err)
			}
			want[i] = append([]byte(nil), float32Bytes(backend.Logits(states[request.stateIndex+2]))...)
		}

		errs := make([]error, len(requests))
		var wg sync.WaitGroup
		wg.Add(len(requests))
		for i, request := range requests {
			i, request := i, request
			go func() {
				defer wg.Done()
				errs[i] = backend.Eval(states[request.stateIndex], request.tokenBase+position, position)
			}()
		}
		wg.Wait()
		for i, err := range errs {
			if err != nil {
				t.Fatalf("concurrent request %d position %d: %v", i, position, err)
			}
			if got := float32Bytes(backend.Logits(states[requests[i].stateIndex])); !bytes.Equal(got, want[i]) {
				t.Fatalf("concurrent request %d logits differ from serial reference at position %d", i, position)
			}
		}
		for i, state := range states {
			request := state.(*cudaRequestState)
			if got := cudaStateGraphCount(request.runtime); got != counts[i] {
				t.Fatalf("concurrent replay changed state %d graph count: got %d, want %d", i, got, counts[i])
			}
		}
	}
}

func cudaStateGraphCount(r *cudaRuntime) int {
	if r == nil {
		return 0
	}
	count := 0
	for i := range r.graphs {
		if r.graphs[i].qkv != nil {
			count++
		}
		if r.graphs[i].ffn != nil {
			count++
		}
	}
	return count
}

// TestCUDAStateSaveLoadGate checks the serialized State contract on the
// device-backed implementation. It is opt-in because the payload contains
// the request-owned device buffers and must run under gpu-lock.
func TestCUDAStateSaveLoadGate(t *testing.T) {
	if os.Getenv("DS4_STATE_SAVELOAD_GATE") != "1" {
		t.Skip("DS4_STATE_SAVELOAD_GATE is not set")
	}
	model := os.Getenv("DS4_MODEL")
	if model == "" {
		t.Fatal("DS4_MODEL is required")
	}
	r, err := newCUDARuntime(model, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	backend := r.stateBackend()
	defer backend.Close()
	state, err := backend.NewState(2048)
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	if err := backend.Eval(state, 0, 0); err != nil {
		t.Fatalf("seed evaluation: %v", err)
	}
	if err := backend.Eval(state, 1, 1); err != nil {
		t.Fatalf("second evaluation: %v", err)
	}
	var saved bytes.Buffer
	if err := state.Save(&saved); err != nil {
		t.Fatalf("save: %v", err)
	}
	if saved.Len() == 0 {
		t.Fatal("save produced an empty payload")
	}
	if err := backend.Eval(state, 2, 2); err != nil {
		t.Fatalf("pre-load evaluation: %v", err)
	}
	want := append([]float32(nil), backend.Logits(state)...)
	if err := state.Load(bytes.NewReader(saved.Bytes())); err != nil {
		t.Fatalf("load: %v", err)
	}
	if err := backend.Eval(state, 2, 2); err != nil {
		t.Fatalf("post-load evaluation: %v", err)
	}
	if got := backend.Logits(state); !bytes.Equal(float32Bytes(got), float32Bytes(want)) {
		t.Fatal("serialized state replay differs")
	}
	t.Logf("serialized_state_bytes=%d logits=%d", saved.Len(), len(want))
}

// TestCUDAStateInterleavedIsolationGate checks that two request states remain
// independent when their evaluations are interleaved on the same backend.
// It is opt-in because it loads the registered model and must run under
// gpu-lock.
func TestCUDAStateInterleavedIsolationGate(t *testing.T) {
	if os.Getenv("DS4_STATE_MULTI_REQUEST_GATE") != "1" {
		t.Skip("DS4_STATE_MULTI_REQUEST_GATE is not set")
	}
	model := os.Getenv("DS4_MODEL")
	if model == "" {
		t.Fatal("DS4_MODEL is required")
	}
	r, err := newCUDARuntime(model, 2048)
	if err != nil {
		t.Fatal(err)
	}
	backend := r.stateBackend()
	defer backend.Close()
	states := make([]dsv4.State, 0, 4)
	for i := 0; i < 4; i++ {
		state, err := backend.NewState(2048)
		if err != nil {
			for _, prior := range states {
				_ = prior.Close()
			}
			t.Fatal(err)
		}
		states = append(states, state)
	}
	defer func() {
		for _, state := range states {
			_ = state.Close()
		}
	}()

	// Run two requests interleaved and two identical requests serially. The
	// serial states are independent references for the corresponding logits.
	const steps = 3
	for position := 0; position < steps; position++ {
		if err := backend.Eval(states[0], position, position); err != nil {
			t.Fatalf("interleaved state A position %d: %v", position, err)
		}
		if err := backend.Eval(states[1], position+17, position); err != nil {
			t.Fatalf("interleaved state B position %d: %v", position, err)
		}
		if err := backend.Eval(states[2], position, position); err != nil {
			t.Fatalf("serial reference A position %d: %v", position, err)
		}
		if err := backend.Eval(states[3], position+17, position); err != nil {
			t.Fatalf("serial reference B position %d: %v", position, err)
		}
		if got, want := float32Bytes(backend.Logits(states[0])), float32Bytes(backend.Logits(states[2])); !bytes.Equal(got, want) {
			t.Fatalf("state A changed under interleaving at position %d", position)
		}
		if got, want := float32Bytes(backend.Logits(states[1])), float32Bytes(backend.Logits(states[3])); !bytes.Equal(got, want) {
			t.Fatalf("state B changed under interleaving at position %d", position)
		}
	}

}
