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

// serialSelectIndexerRows is the reference selection logic. Note on sentinel floor:
// The legacy CUDA scan (pre-970f5e19) floored at -inf (-__int_as_float(0x7f800000)).
// Commit 970f5e19 deliberately aligned CUDA with Metal's -1e30 sentinel
// (selectIndexerSentinel) to ensure cross-backend behavioral convergence.
func serialSelectIndexerRows(scores []float32, k int) []uint32 {
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
		return cudalaunch.Kernel(r.indexerSelectFn).Grid(1).Block(512).On(r.device.DefaultStream()).Launch(ds, allowed, uint32(count))
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

func TestCUDASelectIndexerRowsMatchesTheScan(t *testing.T) {
	r := cudaSynthFoundation(t)
	if err := r.loadMathKernels(); err != nil {
		t.Fatal(err)
	}

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
		{"sentinel divergence band (-inf, -1e30]", 512, func() []float32 {
			// Tests the boundary between the legacy CUDA -inf floor and the aligned
			// Metal -1e30 floor. Rows at -1e30, just below it, and -1e35 must NOT be
			// admitted under the -1e30 floor even when fewer than 512 rows exist above them.
			s := make([]float32, 600)
			for i := range s {
				s[i] = float32(math.Inf(-1))
			}
			for i := 0; i < 100; i++ {
				s[i] = float32(i + 1)
			}
			s[100] = -1e30
			s[101] = float32(math.Nextafter(float64(-1e30), math.Inf(-1)))
			s[102] = -1e35
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
			count := len(scores)
			want := serialSelectIndexerRows(scores, c.k)

			ds := cudaTestAlloc(t, r, count)
			allowed := cudaCompressedMask(t, r, count)
			cudaCompressedUpload(t, ds, scores)

			err := r.device.WithContext(func() error {
				return cudalaunch.Kernel(r.indexerSelectFn).Grid(1).Block(512).On(r.device.DefaultStream()).Launch(ds, allowed, uint32(count))
			})
			if err != nil {
				t.Fatalf("indexerSelectFn: %v", err)
			}

			got := make([]uint32, count)
			if err := allowed.CopyToHost(got); err != nil {
				t.Fatal(err)
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

func TestCUDASentinelFloorDivergenceReceipt(t *testing.T) {
	// Mutation receipt demonstrating the divergence between the legacy CUDA -inf
	// floor and the aligned -1e30 floor.
	s := make([]float32, 600)
	for i := range s {
		s[i] = float32(math.Inf(-1))
	}
	for i := 0; i < 10; i++ {
		s[i] = float32(i + 1)
	}
	s[10] = -1e30
	s[11] = float32(math.Nextafter(float64(-1e30), math.Inf(-1)))
	s[12] = -1e35

	// Legacy CUDA scan (-inf floor):
	legacyScan := func(scores []float32, k int) []uint32 {
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

	legacyWant := legacyScan(s, 512)
	alignedWant := serialSelectIndexerRows(s, 512)

	// In legacy scan, rows 10, 11, 12 were admitted because they are > -inf:
	if legacyWant[10] != 1 || legacyWant[11] != 1 || legacyWant[12] != 1 {
		t.Fatalf("expected legacy -inf scan to admit rows in (-inf, -1e30]")
	}
	// In aligned scan, rows 10, 11, 12 are excluded by the strict > -1e30 floor:
	if alignedWant[10] != 0 || alignedWant[11] != 0 || alignedWant[12] != 0 {
		t.Fatalf("expected aligned -1e30 scan to exclude rows in (-inf, -1e30]")
	}
	t.Logf("receipt confirmed: legacy CUDA -inf admitted rows in (-inf, -1e30], aligned Metal/CUDA -1e30 strictly excludes them")
}
