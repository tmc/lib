//go:build darwin && arm64

package main

import (
	"math"
	"strings"
	"testing"
)

// Independent float64 operator reference. The physical packet must compare
// emitted device IDs/probabilities against this, not another MSL invocation.
func metalDSparkHeadReference(logits, hidden, w1, w2, confidence []float64, rows, vocab, rank, seed int, threshold float64, resetPrevious bool) ([]int, []float64) {
	var ids []int
	var probabilities []float64
	previous := seed
	for row := 0; row < rows; row++ {
		best, bestValue := 0, math.Inf(-1)
		for token := 0; token < vocab; token++ {
			value := logits[row*vocab+token]
			for k := 0; k < rank; k++ {
				value += w2[token*rank+k] * w1[previous*rank+k]
			}
			if value > bestValue {
				best, bestValue = token, value
			}
		}
		z := float64(0)
		for k := 0; k < 4096; k++ {
			z += confidence[k] * hidden[row*4096+k]
		}
		for k := 0; k < rank; k++ {
			z += confidence[4096+k] * w1[previous*rank+k]
		}
		p := 1 / (1 + math.Exp(-z))
		probabilities = append(probabilities, p)
		if p < threshold {
			break
		}
		ids = append(ids, best)
		previous = best
		if resetPrevious {
			previous = seed
		}
	}
	return ids, probabilities
}

func TestMetalDSparkHeadReference(t *testing.T) {
	hidden := make([]float64, 3*4096)
	hidden[0], hidden[4096] = 1, -1
	confidence := make([]float64, 4098)
	confidence[0], confidence[4096], confidence[4097] = .5, .25, -.25
	w1 := []float64{1, 0, 0, 1, 1, 1}
	w2 := []float64{0, 0, 2, 0, 0, 2}
	for _, tc := range []struct {
		threshold float64
		want      []int
	}{
		{.2, []int{1, 2, 1}}, {.5, []int{1}},
	} {
		got, p := metalDSparkHeadReference(make([]float64, 9), hidden, w1, w2, confidence, 3, 3, 2, 0, tc.threshold, false)
		if len(got) != len(tc.want) {
			t.Fatalf("threshold %g IDs %v", tc.threshold, got)
		}
		for i, want := range tc.want {
			if got[i] != want {
				t.Fatalf("row %d ID %d want %d", i, got[i], want)
			}
		}
		if math.Abs(p[0]-1/(1+math.Exp(-.75))) > 1e-15 {
			t.Fatalf("confidence %v", p)
		}
	}
	mutant, _ := metalDSparkHeadReference(make([]float64, 9), hidden, w1, w2, confidence, 3, 3, 2, 0, .2, true)
	if len(mutant) != 3 || mutant[1] == 2 {
		t.Fatal("Markov previous-token mutant not discriminated")
	}
	if strings.Count(metalDSparkAttentionMSL, "previous=best;") != 1 {
		t.Fatal("head same-site mutant must be unique")
	}
}
