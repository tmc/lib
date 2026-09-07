//go:build linux

package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestEnsureBatchFirstRetriesAfterAllocationFailure(t *testing.T) {
	r := new(cudaRuntime)
	wantErr := errors.New("allocation failed")
	calls := 0
	allocate := func() (*cudamem.Arena, cudaTokenState, error) {
		calls++
		if calls == 1 {
			return new(cudamem.Arena), cudaTokenState{}, wantErr
		}
		return new(cudamem.Arena), cudaTokenState{}, nil
	}
	if err := r.ensureBatchFirst(allocate); !errors.Is(err, wantErr) {
		t.Fatalf("first allocation error = %v, want %v", err, wantErr)
	}
	if r.batchArena != nil {
		t.Fatal("failed allocation installed its arena")
	}
	if err := r.ensureBatchFirst(allocate); err != nil {
		t.Fatalf("retry: %v", err)
	}
	if r.batchArena == nil || calls != 2 {
		t.Fatalf("retry installed arena = %v, calls = %d", r.batchArena != nil, calls)
	}
	if err := r.ensureBatchFirst(allocate); err != nil {
		t.Fatalf("reuse: %v", err)
	}
	if calls != 2 {
		t.Fatalf("reuse called allocator; calls = %d", calls)
	}
}

func TestEnsureBatchFirstRejectsMissingArena(t *testing.T) {
	r := new(cudaRuntime)
	err := r.ensureBatchFirst(func() (*cudamem.Arena, cudaTokenState, error) {
		return nil, cudaTokenState{}, nil
	})
	if err == nil || r.batchArena != nil {
		t.Fatalf("ensureBatchFirst() = %v, arena = %v", err, r.batchArena)
	}
}
