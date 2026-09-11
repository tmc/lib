//go:build linux

package main

import (
	"fmt"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// Reference transcription of the indexer scoring loop.
// Accumulation order is strictly serial over h and i.
func serialIndexerScoresReference(q, weights, cache []float32, count int) []float32 {
	scores := make([]float32, count)
	for c := 0; c < count; c++ {
		var score float32
		for h := 0; h < 64; h++ {
			var dot float32
			for i := 0; i < 128; i++ {
				dot += q[h*128+i] * cache[c*128+i]
			}
			score += float32(math.Max(0, float64(dot))) * weights[h] * 0.011048543456039805
		}
		scores[c] = score
	}
	return scores
}

func serialSelectIndexerRows(scores []float32, k int) []uint32 {
	allowed := make([]uint32, len(scores))
	k = min(k, len(scores))
	for range k {
		best := 0
		bestScore := float32(math.Inf(-1))
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

func TestCUDAIndexerScoresBitIdenticalToSerial(t *testing.T) {
	r := cudaSynthFoundation(t)

	for _, count := range []int{513, 514, 1000, 1024, 2049} {
		t.Run(fmt.Sprintf("nComp_%d", count), func(t *testing.T) {
			q := make([]float32, 64*128)
			weights := make([]float32, 64)
			for i := range q {
				q[i] = float32(math.Sin(float64(i)*0.013 + 0.1))
			}
			for h := range weights {
				weights[h] = float32(math.Cos(float64(h)*0.07 + 0.5))
			}
			cache := make([]float32, count*128)
			for c := range count {
				for i := range 128 {
					cache[c*128+i] = float32(math.Sin(float64(c*128+i)*0.0031 + 0.7))
				}
			}

			dq := cudaTestAlloc(t, r, len(q))
			dw := cudaTestAlloc(t, r, len(weights))
			dc := cudaTestAlloc(t, r, len(cache))
			ds := cudaTestAlloc(t, r, count)
			dout := cudaTestAlloc(t, r, len(q))
			allowed := cudaCompressedMask(t, r, count)

			cudaCompressedUpload(t, dq, q)
			cudaCompressedUpload(t, dw, weights)
			cudaCompressedUpload(t, dc, cache)

			if err := r.encodeIndexer(dq, dw, dc, count, dout, ds, allowed, 2051); err != nil {
				t.Fatalf("encodeIndexer: %v", err)
			}

			gotScores := cudaTestRead(t, ds)
			transformedQ := cudaTestRead(t, dout)
			wantScores := serialIndexerScoresReference(transformedQ, weights, cache, count)

			diffs := 0
			for i := 0; i < count; i++ {
				if math.Float32bits(gotScores[i]) != math.Float32bits(wantScores[i]) {
					diffs++
					if diffs <= 5 {
						t.Errorf("row %d score bit mismatch: got %08x (%f), want %08x (%f)",
							i, math.Float32bits(gotScores[i]), gotScores[i], math.Float32bits(wantScores[i]), wantScores[i])
					}
				}
			}
			if diffs > 0 {
				t.Fatalf("count %d: %d / %d scores differ in bit representation from serial reference", count, diffs, count)
			}

			gotMask := make([]uint32, count)
			if err := allowed.CopyToHost(gotMask); err != nil {
				t.Fatal(err)
			}
			wantMask := serialSelectIndexerRows(wantScores, 512)

			maskDiffs := 0
			for i := 0; i < count; i++ {
				if gotMask[i] != wantMask[i] {
					maskDiffs++
				}
			}
			if maskDiffs > 0 {
				t.Fatalf("count %d: %d / %d selection mask entries differ from serial reference", count, maskDiffs, count)
			}
			t.Logf("nComp=%d: %d scores and selection masks 100%% bit-exact with serial reference", count, count)
		})
	}
}

// Mutation gate: Verifies that reassociating the head summation (e.g. summing in reverse)
// alters float bits, proving that the bit-exactness test has genuine discriminating power.
func TestCUDAIndexerScoreComparisonSeesReassociation(t *testing.T) {
	const count = 513
	q := make([]float32, 64*128)
	weights := make([]float32, 64)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.013 + 0.1))
	}
	for h := range weights {
		weights[h] = float32(math.Cos(float64(h)*0.07 + 0.5))
	}
	cache := make([]float32, count*128)
	for c := range count {
		for i := range 128 {
			cache[c*128+i] = float32(math.Sin(float64(c*128+i)*0.0031 + 0.7))
		}
	}

	forward := serialIndexerScoresReference(q, weights, cache, count)

	// Reassociate: reverse head order
	reverse := make([]float32, count)
	for c := 0; c < count; c++ {
		var score float32
		for h := 63; h >= 0; h-- {
			var dot float32
			for i := 0; i < 128; i++ {
				dot += q[h*128+i] * cache[c*128+i]
			}
			score += float32(math.Max(0, float64(dot))) * weights[h] * 0.011048543456039805
		}
		reverse[c] = score
	}

	moved := 0
	for i := 0; i < count; i++ {
		if math.Float32bits(forward[i]) != math.Float32bits(reverse[i]) {
			moved++
		}
	}
	t.Logf("mutation receipt: reassociating head sum moved %d of %d scores (%.1f%%)", moved, count, 100*float64(moved)/count)
	if moved < count*4/5 {
		t.Fatalf("expected >80%% of scores to move under reassociation, got %d / %d", moved, count)
	}
}

func TestCUDAIndexerTieBreakingByLowestIndex(t *testing.T) {
	r := cudaSynthFoundation(t)
	const count = 600
	ds := cudaTestAlloc(t, r, count)
	allowed := cudaCompressedMask(t, r, count)

	// All equal scores
	scores := make([]float32, count)
	for i := range scores {
		scores[i] = 1.2345
	}
	cudaCompressedUpload(t, ds, scores)

	if err := r.loadMathKernels(); err != nil {
		t.Fatal(err)
	}
	err := r.device.WithContext(func() error {
		return cudalaunch.Kernel(r.indexerSelectFn).Grid(1).Block(256).On(r.device.DefaultStream()).Launch(ds, allowed, uint32(count))
	})
	if err != nil {
		t.Fatalf("indexerSelectFn: %v", err)
	}

	mask := make([]uint32, count)
	if err := allowed.CopyToHost(mask); err != nil {
		t.Fatal(err)
	}

	// First 512 must be 1, rest must be 0
	for i := 0; i < count; i++ {
		want := uint32(0)
		if i < 512 {
			want = 1
		}
		if mask[i] != want {
			t.Fatalf("tie-break failure at index %d: got %d, want %d", i, mask[i], want)
		}
	}
	t.Logf("tie-breaking gate: all %d equal scores admitted strictly indices 0..511", count)
}
