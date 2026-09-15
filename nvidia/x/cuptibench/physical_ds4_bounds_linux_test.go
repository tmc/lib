//go:build linux

package cuptibench_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

// Use the DS4 profiling limits unchanged on a small known pipeline. Direct and
// retained graph execution distinguish capacity configuration from graph replay.
// This is a native synthetic control, not a DS4 implementation or qualification.
func TestPhysicalDS4KernelReplayBounds(t *testing.T) {
	p := newPhysicalPipeline(t, 4)
	for _, graphMode := range []bool{false, true} {
		name := "direct"
		if graphMode {
			name = "graph"
		}
		t.Run(name, func(t *testing.T) {
			run := func() error {
				if err := cudalaunch.Kernel(p.project).Grid(4).Block(physicalBlock).On(p.streams[0]).Launch(p.state, p.mid); err != nil {
					return err
				}
				return cudalaunch.Kernel(p.decode).Grid(4).Block(physicalBlock).On(p.streams[0]).Launch(p.state, p.mid, p.out)
			}
			var graphs []*cudagraph.Exec
			if graphMode {
				var captured error
				graph, err := cudagraph.Capture(p.streams[0], func() { captured = run() })
				physicalCheck(t, err)
				physicalClose(t, graph.Destroy)
				physicalCheck(t, captured)
				executable, err := graph.InstantiateWithOptions(cudagraph.InstantiateOptions{RetainKernelOrder: true})
				physicalCheck(t, err)
				physicalClose(t, executable.Destroy)
				physicalCheck(t, graph.Destroy())
				graphs = []*cudagraph.Exec{executable}
				run = func() error { return executable.Launch(p.streams[0]) }
			}
			config := cuptibench.Config{Mode: cuptibench.Kernels, ReplayMode: cuptirange.KernelReplay, Operations: 1, CachePolicy: cuptirange.CachePreserve, Metrics: []string{physicalMetric}, MaxConfigBytes: 16 << 20, Limits: cuptirange.Limits{MaxRanges: 16384, MaxRangeTreeNodes: 32768, MaxCounterBytes: 64 << 20, MaxPasses: 64}}
			raw, err := json.Marshal(config)
			physicalCheck(t, err)
			t.Logf("DS4 bounds synthetic config=%s", raw)
			plan, err := cuptibench.New(p.dev, config)
			physicalCheck(t, err)
			physicalClose(t, plan.Close)
			prepares, runs := 0, 0
			w := cuptibench.Workload{View: "full", Graphs: graphs, Streams: p.streams[:1], Synchronize: p.synchronize, Prepare: func() error { prepares++; return p.prepare() }, Run: func(*cuptibench.Scope) error { runs++; return run() }, Preparation: "reset all persistent buffers before collection; project4 then decode4 on one stream; DS4 capacity limits unchanged; cache preserve"}
			result, err := plan.Collect(w)
			raw, encodeErr := json.Marshal(result)
			physicalCheck(t, encodeErr)
			t.Logf("DS4 bounds synthetic result=%s error=%v", raw, err)
			physicalCheck(t, err)
			if prepares != 1 || runs != 1 || result.Quality.PassesStarted != 1 || result.Quality.PassesStopped != 1 || !result.Quality.AllPassesSubmitted || !result.Quality.Decoded || result.Quality.DroppedRanges != 0 {
				t.Fatalf("callbacks=%d/%d quality=%+v", prepares, runs, result.Quality)
			}
			if len(result.Ranges) != 2 {
				t.Fatalf("ranges=%+v", result.Ranges)
			}
			for i, v := range result.Ranges {
				want, ordinal := "fixture_project", "0"
				if i == 1 {
					want, ordinal = "fixture_decode", "1"
				}
				if !strings.Contains(v.Name, want) || v.NativeName != ordinal || physicalValue(t, v) != 4 {
					t.Fatalf("range[%d]=%+v", i, v)
				}
			}
			checkPhysicalOutput(t, p, "", 1)
		})
	}
}
