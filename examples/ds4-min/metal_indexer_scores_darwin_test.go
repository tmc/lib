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

// scanIndexerRows is the k-pass scan selectIndexerRows replaced, transcribed
// verbatim. The sort-based version is meant to be indistinguishable from it on
// every input, including the inputs the scan handles by accident.
func scanIndexerRows(scores []float32, k int) []uint32 {
	allowed := make([]uint32, len(scores))
	k = min(k, len(scores))
	for range k {
		best := 0
		bestScore := float32(-1e30)
		for i, score := range scores {
			if allowed[i] == 0 && score > bestScore {
				best = i
				bestScore = score
			}
		}
		allowed[best] = 1
	}
	return allowed
}

func TestMetalSelectIndexerRowsMatchesTheScan(t *testing.T) {
	// The cases that matter are the ones where a reasonable rewrite and the
	// scan part company: ties, values at or below the sentinel, NaN, and
	// fewer eligible rows than k. A pile of distinct random floats agrees
	// under any correct implementation and proves the least.
	cases := []struct {
		name   string
		k      int
		scores func() []float32
	}{
		{"distinct", 512, func() []float32 {
			s := make([]float32, 1000)
			for i := range s {
				s[i] = float32(math.Sin(float64(i)*1.7 + 0.3))
			}
			return s
		}},
		{"all tied", 512, func() []float32 {
			s := make([]float32, 700)
			for i := range s {
				s[i] = 2.5
			}
			return s
		}},
		{"ties straddling the cut", 512, func() []float32 {
			// 400 clear winners, then 300 identical scores competing for the
			// remaining 112 slots: the cut lands inside a tie group.
			s := make([]float32, 700)
			for i := range s {
				if i < 400 {
					s[i] = 10 + float32(i)
				} else {
					s[i] = 1
				}
			}
			return s
		}},
		{"below the sentinel", 512, func() []float32 {
			s := make([]float32, 600)
			for i := range s {
				s[i] = float32(math.Sin(float64(i)))
				if i%3 == 0 {
					s[i] = -2e30 // under the scan's floor: never picked
				}
			}
			return s
		}},
		{"fewer eligible than k", 512, func() []float32 {
			s := make([]float32, 600)
			for i := range s {
				s[i] = -2e30
			}
			for i := 100; i < 110; i++ {
				s[i] = float32(i)
			}
			return s
		}},
		{"none eligible", 512, func() []float32 {
			s := make([]float32, 600)
			for i := range s {
				s[i] = float32(math.Inf(-1))
			}
			return s
		}},
		{"nan", 512, func() []float32 {
			s := make([]float32, 600)
			for i := range s {
				s[i] = float32(math.Sin(float64(i)))
				if i%5 == 0 {
					s[i] = float32(math.NaN())
				}
			}
			return s
		}},
		{"k exceeds the rows", 512, func() []float32 {
			s := make([]float32, 100)
			for i := range s {
				s[i] = float32(i % 7)
			}
			return s
		}},
		{"deep regime shape", 512, func() []float32 {
			s := make([]float32, 16418)
			for i := range s {
				s[i] = float32(math.Cos(float64(i) * 0.011))
			}
			return s
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			scores := c.scores()
			want := scanIndexerRows(scores, c.k)
			got := selectIndexerRows(scores, c.k)
			if len(got) != len(want) {
				t.Fatalf("length %d, want %d", len(got), len(want))
			}
			admitted := 0
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("row %d (score %v): admitted %d, scan admitted %d", i, scores[i], got[i], want[i])
				}
				admitted += int(want[i])
			}
			t.Logf("%d rows, %d admitted", len(scores), admitted)
		})
	}
}

func TestMetalSelectIndexerRowComparisonSeesATieBreakFlip(t *testing.T) {
	// Mutation receipt for the test above: a selection that resolves ties to
	// the highest index instead of the lowest must be caught. Without this,
	// the agreement cases prove only that both implementations pick 512 large
	// scores.
	scores := make([]float32, 700)
	for i := range scores {
		scores[i] = 2.5
	}
	want := scanIndexerRows(scores, 512)
	flipped := make([]uint32, len(scores))
	for i := len(scores) - 512; i < len(scores); i++ {
		flipped[i] = 1
	}
	moved := 0
	for i := range want {
		if want[i] != flipped[i] {
			moved++
		}
	}
	if moved == 0 {
		t.Fatal("the highest-index tie-break is indistinguishable from the scan on this fixture; the agreement test above proves nothing about ties")
	}
	t.Logf("flipping the tie-break moves %d of %d rows", moved, len(scores))
}
