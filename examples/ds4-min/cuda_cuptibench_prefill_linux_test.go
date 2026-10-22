//go:build linux

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

// BenchmarkCUDAPrefill evaluates prompt chunks from a fresh session using
// cuptibench to measure prefill launches, wall time, and GPU duration.
func BenchmarkCUDAPrefill(b *testing.B) {
	if !*ds4CUPTIBench {
		b.Skip("enable with -ds4-cuptibench under gpu-lock")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		b.Fatal("gpu-lock is required")
	}
	for _, chunk := range cudaBenchSizes(b, "DS4_BENCH_CHUNKS", []int{32, 64, 128, 256}) {
		b.Run(fmt.Sprintf("tokens=%d", chunk), func(b *testing.B) {
			r := cudaPrefillBenchRuntime(b)
			if chunk >= r.ctx {
				b.Skipf("chunk %d does not fit context %d", chunk, r.ctx)
			}
			tokens := cudaBenchTokens(chunk)
			if err := r.Reset(); err != nil {
				b.Fatal(err)
			}
			if err := r.EvalBatch(tokens, 0); err != nil {
				b.Fatal(err)
			}
			if err := r.Reset(); err != nil {
				b.Fatal(err)
			}
			metrics, err := cuptibench.Resolve("launches")
			if err != nil {
				b.Fatal(err)
			}
			plan, err := cuptibench.New(r.device, cuptibench.Config{
				Metrics:     metrics,
				Mode:        cuptibench.Whole,
				Operations:  1,
				CachePolicy: cuptirange.CachePreserve,
				Limits: cuptirange.Limits{
					MaxRanges:         64,
					MaxRangeTreeNodes: 128,
					MaxCounterBytes:   16 << 20,
					MaxPasses:         32,
				},
				MaxConfigBytes: 4 << 20,
			})
			if plan != nil {
				b.Cleanup(func() {
					if err := plan.Close(); err != nil {
						b.Error(err)
					}
				})
			}
			if err != nil {
				b.Fatal(err)
			}
			var hostLaunches uint64
			work := cuptibench.Workload{
				View:        "full",
				Streams:     []*cuda.Stream{r.stream()},
				Preparation: "reset runtime state before prefill",
				Prepare:     func() error { return r.Reset() },
				Synchronize: func() error { return syncStream(r.stream()) },
				Run: func(s *cuptibench.Scope) error {
					before := cudalaunch.TotalLaunches()
					err := r.EvalBatch(tokens, 0)
					hostLaunches = cudalaunch.TotalLaunches() - before
					return err
				},
			}
			res := plan.Measure(b, work)
			b.ReportMetric(float64(chunk), "tokens/op")
			b.ReportMetric(float64(hostLaunches), "host-launches/op")
			if res != nil {
				counters := res
				if res.Aggregate != nil {
					counters = res.Aggregate
				}
				for _, rg := range counters.Ranges {
					for _, m := range rg.Metrics {
						if m.Name == "sm__ctas_launched.sum" {
							b.ReportMetric(m.Value, "launches/op")
						}
					}
				}
			}
		})
	}
}
