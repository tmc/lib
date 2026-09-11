//go:build darwin && arm64

package main

import (
	"math"
	"testing"
)

// serialIndexerScores is the loop indexerScores replaced, transcribed here so
// the parallel version is checked against the code it is meant to be identical
// to rather than against a recomputation of the same idea.
func serialIndexerScores(rows, q, weights []float32, nComp int) []float32 {
	scores := make([]float32, nComp)
	for c := range scores {
		row := rows[c*128 : (c+1)*128]
		var score float32
		for h := range 64 {
			var dot float32
			for i, value := range row {
				dot += value * q[h*128+i]
			}
			if dot > 0 {
				score += dot * weights[h]
			}
		}
		scores[c] = score
	}
	return scores
}

// reassociatedIndexerScores sums the heads in reverse. It is the mutation this
// test has to be able to see: splitting the rows across goroutines is exact,
// reordering an accumulation is not, and a test that cannot tell them apart
// would pass a change that silently moved the selection.
func reassociatedIndexerScores(rows, q, weights []float32, nComp int) []float32 {
	scores := make([]float32, nComp)
	for c := range scores {
		row := rows[c*128 : (c+1)*128]
		var score float32
		for h := 63; h >= 0; h-- {
			var dot float32
			for i, value := range row {
				dot += value * q[h*128+i]
			}
			if dot > 0 {
				score += dot * weights[h]
			}
		}
		scores[c] = score
	}
	return scores
}

// indexerScoreInputs builds inputs whose sums are not exactly representable,
// so reassociation actually changes bits. A fixture of small integers would
// make float addition associative and the mutation check vacuous.
func indexerScoreInputs(nComp int) (rows, q, weights []float32) {
	rows = make([]float32, nComp*128)
	for i := range rows {
		rows[i] = float32(math.Cos(float64(i)*0.37)) * 0.013
	}
	q = make([]float32, 64*128)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.11)) * 0.017
	}
	weights = make([]float32, 64)
	for i := range weights {
		weights[i] = float32(math.Sin(float64(i)*1.7)) / float32(math.Sqrt(64*128))
	}
	return rows, q, weights
}

func TestMetalIndexerScoresAreBitIdenticalToSerial(t *testing.T) {
	// 513 is the first count that reaches this code at all: below 512 rows
	// IndexerAllowed admits everything and never scores. The others are chosen
	// to land on and off a chunk boundary for any plausible CPU count.
	for _, nComp := range []int{513, 514, 1000, 1024, 2049} {
		rows, q, weights := indexerScoreInputs(nComp)
		want := serialIndexerScores(rows, q, weights, nComp)
		got := indexerScores(rows, q, weights, nComp)
		if len(got) != len(want) {
			t.Fatalf("nComp %d: got %d scores, want %d", nComp, len(got), len(want))
		}
		var nonzero int
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("nComp %d: score %d differs: parallel %v (%#08x), serial %v (%#08x)",
					nComp, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
			if want[i] != 0 {
				nonzero++
			}
		}
		// A loop that returned early would agree with a reference that also
		// returned early. Require that the comparison had something to compare.
		if nonzero < nComp/2 {
			t.Fatalf("nComp %d: only %d of %d scores are non-zero; the fixture is not exercising the accumulation", nComp, nonzero, nComp)
		}
	}
}

func TestMetalIndexerScoreComparisonSeesReassociation(t *testing.T) {
	// Receipt that the test above has power: the same comparison applied to a
	// reordered accumulation must fail.
	const nComp = 513
	rows, q, weights := indexerScoreInputs(nComp)
	want := serialIndexerScores(rows, q, weights, nComp)
	mutated := reassociatedIndexerScores(rows, q, weights, nComp)
	var differing int
	for i := range want {
		if math.Float32bits(mutated[i]) != math.Float32bits(want[i]) {
			differing++
		}
	}
	if differing == 0 {
		t.Fatal("summing the heads in reverse produced bit-identical scores, so the bit-identity check above cannot see reassociation and proves nothing")
	}
	t.Logf("reassociating the head sum moves %d of %d scores", differing, nComp)
}

func TestMetalSelectIndexerRowsBreaksTiesByLowestIndex(t *testing.T) {
	// The scan is `if allowed[i] == 0 && score > bestScore`, ascending, with a
	// strict comparison, so equal scores resolve to the lowest index. Any
	// future parallel selection has to reproduce that, not merely pick 512
	// large scores.
	scores := make([]float32, 600)
	for i := range scores {
		scores[i] = 1 // every score identical
	}
	allowed := selectIndexerRows(scores, 512)
	for i := range allowed {
		want := uint32(0)
		if i < 512 {
			want = 1
		}
		if allowed[i] != want {
			t.Fatalf("row %d: allowed %d, want %d -- ties must resolve to the lowest index", i, allowed[i], want)
		}
	}
}
