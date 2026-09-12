//go:build darwin && arm64

package main

import (
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// BenchmarkIndexerScores compares the CPU loop against the GPU kernel at the
// three counts that bracket production: 513 is the first count past the
// position-2051 boundary, 16418 is a full 65,665-token context.
//
// The scan runs once per even layer, so multiply a per-call figure by 21 to
// read it as a per-token cost.
func BenchmarkIndexerScores(b *testing.B) {
	for _, nComp := range []int{513, 4096, 16418} {
		rows, q, weights := indexerScoreInputs(nComp)

		b.Run("cpu/"+itoa(nComp), func(b *testing.B) {
			for b.Loop() {
				indexerScores(rows, q, weights, nComp)
			}
		})

		b.Run("gpu/"+itoa(nComp), func(b *testing.B) {
			dev, err := libmetal.NewDevice()
			if err != nil {
				b.Fatal(err)
			}
			defer dev.Close()
			ops, err := newMetalPinnedOps(dev)
			if err != nil {
				b.Fatal(err)
			}
			defer ops.Close()
			cache, err := metalmem.AllocFromSlice(dev, rows)
			if err != nil {
				b.Fatal(err)
			}
			defer cache.Free()
			b.ResetTimer()
			for b.Loop() {
				if _, err := ops.IndexerScores(cache, q, weights, nComp); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
