//go:build darwin && arm64

package main

import (
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// BenchmarkIndexerScoresCrossover sweeps the count at which the GPU scan
// overtakes the CPU loop. Bit-exactness holds on both sides -- see
// TestMetalIndexerScoresGPUIsBitIdenticalToCPU -- so the choice between them is
// purely one of cost, and a threshold changes no output bit.
func BenchmarkIndexerScoresCrossover(b *testing.B) {
	for _, nComp := range []int{513, 1024, 2048, 4096, 8192, 16418} {
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

// BenchmarkIndexerScoresCost decomposes the fixed per-call cost of
// IndexerScores. The GPU arm of BenchmarkIndexerScores is nearly flat from
// nComp=513 to nComp=16418 -- a 32x range in scan work -- so almost all of it
// is overhead. These arms say which overhead.
func BenchmarkIndexerScoresCost(b *testing.B) {
	const nComp = 513
	rows, q, weights := indexerScoreInputs(nComp)
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

	b.Run("submit-empty", func(b *testing.B) {
		for b.Loop() {
			cb := ops.commandBuffer()
			enc := cb.ComputeCommandEncoder()
			enc.EndEncoding()
			if err := ops.submitAndWait("bench-empty", cb); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("alloc-three", func(b *testing.B) {
		for b.Loop() {
			qbuf, err := metalmem.AllocFromSlice(dev, q)
			if err != nil {
				b.Fatal(err)
			}
			wbuf, err := metalmem.AllocFromSlice(dev, weights)
			if err != nil {
				b.Fatal(err)
			}
			out, err := metalmem.Alloc[float32](dev, nComp)
			if err != nil {
				b.Fatal(err)
			}
			out.Free()
			wbuf.Free()
			qbuf.Free()
		}
	})

	b.Run("full", func(b *testing.B) {
		for b.Loop() {
			if _, err := ops.IndexerScores(cache, q, weights, nComp); err != nil {
				b.Fatal(err)
			}
		}
	})
}
