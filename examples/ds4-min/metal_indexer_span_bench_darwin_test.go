//go:build darwin && arm64

package main

import (
	"sort"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// BenchmarkIndexerScoresGPUSpan reports the kernel's own device-time span
// rather than wall-clock, using the GPU clock the command buffer reports.
//
// Wall-clock on a shared desktop is not a usable instrument here: consecutive
// identical samples of this kernel have differed by 15x while other processes
// held the GPU. The device span excludes host scheduling, so it survives a
// loaded host, and it is what the kernel's shape actually controls.
//
// The reported gpu-ms/op is a median over the run's spans, not a mean: one
// preemption is larger than the effect being measured.
func BenchmarkIndexerScoresGPUSpan(b *testing.B) {
	for _, nComp := range []int{513, 2048, 16418} {
		b.Run(itoa(nComp), func(b *testing.B) {
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
			log := &metalRegionLog{}
			ops.regions = log
			defer func() { ops.regions = nil }()
			b.ResetTimer()
			for b.Loop() {
				if _, err := ops.IndexerScores(cache, q, weights, nComp); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			spans := log.all()
			if len(spans) == 0 {
				b.Fatal("no GPU spans recorded")
			}
			ms := make([]float64, 0, len(spans))
			for _, s := range spans {
				ms = append(ms, s.Seconds()*1e3)
			}
			sort.Float64s(ms)
			b.ReportMetric(ms[len(ms)/2], "gpu-ms/op")
			b.ReportMetric(ms[0], "gpu-ms-min")
		})
	}
}
