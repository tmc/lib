package egraphcore

import "testing"

func TestCandidateLatticeHeapOrder(t *testing.T) {
	buckets := []CandidateLatticeBucket[int]{
		{Finalists: []CandidateLatticeFinalist[int]{{Score: 0}, {Score: 1}, {Score: 2}}},
		{Finalists: []CandidateLatticeFinalist[int]{{Score: 0}, {Score: 1}, {Score: 2}}},
	}
	lattice, err := NewCandidateLattice(buckets, func(scores []float64) float64 {
		return scores[0] + scores[1]
	}, 10, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]int{{0, 0}, {0, 1}, {1, 0}, {0, 2}, {1, 1}, {2, 0}}
	for i, indices := range want {
		set, ok, err := lattice.Next()
		if err != nil {
			t.Fatalf("Next() at %d: %v", i, err)
		}
		if !ok {
			t.Fatalf("Next() at %d: no proposal", i)
		}
		if len(set.Indices) != len(indices) {
			t.Fatalf("proposal %d has indices %v, want %v", i, set.Indices, indices)
		}
		for j := range indices {
			if set.Indices[j] != indices[j] {
				t.Fatalf("proposal %d has indices %v, want %v", i, set.Indices, indices)
			}
		}
		if i < 3 {
			if err := lattice.Reject(set, "test rejection"); err != nil {
				t.Fatalf("Reject() at %d: %v", i, err)
			}
		} else if err := lattice.Accept(set); err != nil {
			t.Fatalf("Accept() at %d: %v", i, err)
		}
	}
	if set, ok, err := lattice.Next(); err != nil || ok || set != nil {
		t.Fatalf("Next() after exhaustion = (%v, %t, %v), want (nil, false, nil)", set, ok, err)
	}
}

func TestCandidateLatticeProposalLimit(t *testing.T) {
	buckets := []CandidateLatticeBucket[int]{
		{Finalists: []CandidateLatticeFinalist[int]{{Score: 0}, {Score: 1}}},
		{Finalists: []CandidateLatticeFinalist[int]{{Score: 0}, {Score: 1}}},
	}
	lattice, err := NewCandidateLattice(buckets, func(scores []float64) float64 {
		return scores[0] + scores[1]
	}, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	set, ok, err := lattice.Next()
	if err != nil || !ok {
		t.Fatalf("Next() = (%v, %t, %v), want a proposal", set, ok, err)
	}
	if err := lattice.Reject(set, "test rejection"); err != nil {
		t.Fatal(err)
	}
	if set, ok, err := lattice.Next(); err != nil || ok || set != nil || !lattice.LimitReached() {
		t.Fatalf("Next() at limit = (%v, %t, %v), limit reached %t", set, ok, err, lattice.LimitReached())
	}
}
