//go:build linux

package main

import (
	"bytes"
	"errors"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestCUDASpeculativeEpochRejectedStateOperations(t *testing.T) {
	s := &cudaRequestState{runtime: &cudaRuntime{ctx: 16}, lifecycleEpoch: 7}
	if err := s.Load(bytes.NewReader(nil)); err == nil {
		t.Fatal("accepted missing State payload")
	}
	if got, err := s.speculativeEpoch(); err != nil || got != 7 {
		t.Fatalf("rejected Load changed epoch: %d %v", got, err)
	}
	foreign := &cudaStateSnapshot{owner: &cudaRuntime{}, pos: 0}
	if err := s.Restore(foreign); err == nil {
		t.Fatal("accepted foreign State snapshot")
	}
	if got, err := s.speculativeEpoch(); err != nil || got != 7 {
		t.Fatalf("rejected Restore changed epoch: %d %v", got, err)
	}
	invalid := &cudaStateSnapshot{owner: s.runtime, pos: 17}
	if err := s.Restore(invalid); err == nil {
		t.Fatal("accepted out-of-context State snapshot")
	}
	if got, err := s.speculativeEpoch(); err != nil || got != 7 {
		t.Fatalf("invalid frontier changed epoch: %d %v", got, err)
	}
}

func TestCUDASpeculativeEpochTransition(t *testing.T) {
	s := &cudaRequestState{runtime: &cudaRuntime{}}
	rejected := errors.New("rejected transition")
	for _, step := range []struct {
		err  error
		want uint64
	}{{nil, 1}, {rejected, 1}, {nil, 2}, {nil, 3}} {
		if err := s.completeSpeculativeTransition(step.err); !errors.Is(err, step.err) {
			t.Fatalf("transition: %v", err)
		}
		if got, err := s.speculativeEpoch(); err != nil || got != step.want {
			t.Fatalf("epoch %d %v, want %d", got, err, step.want)
		}
	}
	s.lifecycleEpoch = ^uint64(0)
	if err := s.completeSpeculativeTransition(nil); !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Fatalf("overflow: %v", err)
	}
	if _, err := s.speculativeEpoch(); !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Fatalf("poisoned epoch: %v", err)
	}
	if s.lifecycleEpoch != ^uint64(0) {
		t.Fatal("overflow wrapped epoch")
	}
}
