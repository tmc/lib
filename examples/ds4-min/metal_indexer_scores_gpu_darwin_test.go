//go:build darwin && arm64

package main

import (
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// TestMetalIndexerScoresGPUIsBitIdenticalToCPU checks the indexer_scores kernel
// against indexerScores, which is the oracle -- not the CUDA kernel, which
// computes a different expression (fmaxf(0,dot)*w*k, three multiplies with the
// scale applied per head) and is not bit-equivalent to either.
//
// The bar is zero changed float32 bits. The scores order the rows
// selectIndexerRows admits, so a one-ULP move is a different set of admitted
// rows, not a rounding detail.
func TestMetalIndexerScoresGPUIsBitIdenticalToCPU(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()

	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()

	// 513 is the first count that reaches the scoring loop at all; 514 and
	// 2049 sit one past a threadgroup-sized boundary, and 1024 lands exactly
	// on one. The kernel's only shape-dependent line is its n_comp guard.
	for _, nComp := range []int{513, 514, 1024, 2049} {
		rows, q, weights := indexerScoreInputs(nComp)
		want := indexerScores(rows, q, weights, nComp)

		cache, err := metalmem.AllocFromSlice(dev, rows)
		if err != nil {
			t.Fatal(err)
		}
		got, err := ops.IndexerScores(cache, q, weights, nComp)
		cache.Free()
		if err != nil {
			t.Fatalf("nComp %d: %v", nComp, err)
		}
		if len(got) != len(want) {
			t.Fatalf("nComp %d: got %d scores, want %d", nComp, len(got), len(want))
		}
		var nonzero int
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("nComp %d: score %d differs: GPU %v (%#08x), CPU %v (%#08x)",
					nComp, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
			if want[i] != 0 {
				nonzero++
			}
		}
		// A kernel that wrote nothing would agree with an all-zero reference.
		if nonzero < nComp/2 {
			t.Fatalf("nComp %d: only %d of %d scores are non-zero; the fixture is not exercising the accumulation", nComp, nonzero, nComp)
		}
		t.Logf("nComp %d: %d scores bit-identical, %d non-zero", nComp, len(want), nonzero)
	}
}
