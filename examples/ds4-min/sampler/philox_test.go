package sampler

import "testing"

func TestSampleBatchPhiloxBatchInvariance(t *testing.T) {
	rows := [][]float32{
		{1.2, 2.4, 0.5, 3.1, -0.7, 2.9, 1.8, 0.2},
		{-0.4, 1.5, 2.2, 2.1, 0.9, -1.1, 2.0, 1.7},
		{0.1, 0.2, 0.3, 0.4, 2.5, 2.6, 2.4, -0.5},
	}
	seeds := []uint64{101, 202, 303}
	steps := []uint64{7, 19, 42}
	p := Params{Temperature: 0.8, TopK: 4, TopP: 0.9, MinP: 0.05}

	batched, err := SampleBatchPhilox(rows, p, seeds, steps)
	if err != nil {
		t.Fatalf("SampleBatchPhilox failed: %v", err)
	}
	for i := range rows {
		solo, err := SamplePhilox(rows[i], p, seeds[i], steps[i], nil)
		if err != nil {
			t.Fatalf("SamplePhilox(%d) failed: %v", i, err)
		}
		if solo != batched[i] {
			t.Fatalf("row %d batch invariance violation: solo=%d batched=%d", i, solo, batched[i])
		}
	}

	// Reversed batch order must produce identical per-sequence tokens.
	revRows := [][]float32{rows[2], rows[1], rows[0]}
	revSeeds := []uint64{seeds[2], seeds[1], seeds[0]}
	revSteps := []uint64{steps[2], steps[1], steps[0]}
	revBatched, err := SampleBatchPhilox(revRows, p, revSeeds, revSteps)
	if err != nil {
		t.Fatalf("reversed SampleBatchPhilox failed: %v", err)
	}
	for i := range rows {
		if revBatched[len(rows)-1-i] != batched[i] {
			t.Fatalf("row %d permutation mismatch: got %d, want %d", i, revBatched[len(rows)-1-i], batched[i])
		}
	}
}
