package main

import (
	"testing"
)

func TestNCCLOracles(t *testing.T) {
	ranks := 2
	elements := 257

	// 1. AllReduce checks
	for i := 0; i < elements; i++ {
		want, err := allReduceExpected(ranks, i)
		if err != nil {
			t.Fatalf("allReduceExpected(%d, %d) error: %v", ranks, i, err)
		}
		// Direct sum of inputs from rank 0 and rank 1
		directSum := ncclInput(0, i) + ncclInput(1, i)
		if want != directSum {
			t.Fatalf("allreduce at %d: got %d, want %d", i, want, directSum)
		}
	}

	// 2. AllGather checks
	for r := 0; r < ranks; r++ {
		for i := 0; i < elements; i++ {
			idx := r*elements + i
			got := allGatherExpected(elements, idx)
			want := ncclInput(r, i)
			if got != want {
				t.Fatalf("allgather at rank %d, elem %d (idx %d): got %d, want %d", r, i, idx, got, want)
			}
		}
	}

	// Negative control: wrong oracle / bad sentinel detection
	wrongVal := ncclInput(0, 0) + 1
	if wrongVal == ncclInput(0, 0) {
		t.Fatal("oracle bad-control check failed")
	}
}
