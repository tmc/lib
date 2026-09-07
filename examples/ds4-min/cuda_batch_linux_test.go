//go:build linux

package main

import (
	"errors"
	"testing"
)

func TestEnsureBatchRetriesAfterAllocationFailure(t *testing.T) {
	r := new(cudaRuntime)
	wantErr := errors.New("allocation failed")
	calls := 0
	failedHC := &cudaHC{ready: true}
	allocate := func() (*cudaBatchResources, error) {
		calls++
		if calls == 1 {
			return &cudaBatchResources{hc: []*cudaHC{failedHC}}, wantErr
		}
		return &cudaBatchResources{
			rows: make([]cudaTokenState, cudaPrefillBatchWidth-1),
			hc:   make([]*cudaHC, cudaPrefillBatchWidth-1),
		}, nil
	}
	if err := r.ensureBatch(allocate); !errors.Is(err, wantErr) {
		t.Fatalf("first allocation error = %v, want %v", err, wantErr)
	}
	if r.batch != nil {
		t.Fatal("failed allocation installed its resources")
	}
	if failedHC.ready {
		t.Fatal("failed allocation did not close partial resources")
	}
	if err := r.ensureBatch(allocate); err != nil {
		t.Fatalf("retry: %v", err)
	}
	if r.batch == nil || calls != 2 {
		t.Fatalf("retry installed resources = %v, calls = %d", r.batch != nil, calls)
	}
	if err := r.ensureBatch(allocate); err != nil {
		t.Fatalf("reuse: %v", err)
	}
	if calls != 2 {
		t.Fatalf("reuse called allocator; calls = %d", calls)
	}
}

func TestEnsureBatchRejectsIncompleteResources(t *testing.T) {
	r := new(cudaRuntime)
	err := r.ensureBatch(func() (*cudaBatchResources, error) {
		return nil, nil
	})
	if err == nil || r.batch != nil {
		t.Fatalf("ensureBatch() = %v, resources = %v", err, r.batch)
	}
	err = r.ensureBatch(func() (*cudaBatchResources, error) {
		return &cudaBatchResources{
			rows: make([]cudaTokenState, cudaPrefillBatchWidth-2),
			hc:   make([]*cudaHC, cudaPrefillBatchWidth-1),
		}, nil
	})
	if err == nil || r.batch != nil {
		t.Fatalf("ensureBatch(short rows) = %v, resources = %v", err, r.batch)
	}
}

func TestCUDARejectsInvalidInternalBatchLengths(t *testing.T) {
	r := new(cudaRuntime)
	if err := r.evalBatch([]int{1}, 0, false); err == nil {
		t.Fatal("evalBatch accepted one token")
	}
	if err := r.evalBatch(make([]int, cudaPrefillBatchWidth+1), 0, false); err == nil {
		t.Fatal("evalBatch accepted too many tokens")
	}
	if err := r.evalLayerBatch(nil, nil, 0, nil, 0); err == nil {
		t.Fatal("evalLayerBatch accepted missing rows")
	}
}
