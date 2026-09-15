//go:build linux

package cuptibench_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

// Graph memory nodes have GPU-ordered lifetimes, not kernel symbols. This
// physical control tests the additional proposition that this driver/CUPTI
// combination exposes exactly the two declared kernels as automatic ranges.
// Any hidden profiler range fails the independent count gate. The allocation
// stays live from project through decode, and is freed within each graph run.
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#graph-memory-nodes
func TestPhysicalGraphAllocationIdentity(t *testing.T) {
	p := newPhysicalPipeline(t, 5)
	var captured error
	graph, err := cudagraph.Capture(p.streams[0], func() {
		captured = p.dev.WithContext(func() error {
			mid, err := p.streams[0].Raw().MemAllocAsync(5 * physicalBlock * 4)
			if err != nil {
				return err
			}
			err = cudalaunch.Kernel(p.project).Grid(5).Block(physicalBlock).On(p.streams[0]).Launch(p.state, mid)
			if err == nil {
				err = cudalaunch.Kernel(p.decode).Grid(2).Block(physicalBlock).On(p.streams[0]).Launch(p.state, mid, p.out)
			}
			return errors.Join(err, mid.MemFreeAsync(p.streams[0].Raw()))
		})
	})
	physicalCheck(t, err)
	physicalClose(t, graph.Destroy)
	physicalCheck(t, captured)
	var nodes []driver.GraphNode
	physicalCheck(t, p.dev.WithContext(func() error {
		var err error
		nodes, err = graph.Raw().NodesBounded(65536)
		return err
	}))
	kinds := make(map[driver.GraphNodeType]int)
	physicalCheck(t, p.dev.WithContext(func() error {
		for _, node := range nodes {
			kind, err := node.GetType()
			if err != nil {
				return err
			}
			kinds[kind]++
		}
		return nil
	}))
	if len(nodes) != 4 || kinds[driver.GraphNodeTypeMemAlloc] != 1 || kinds[driver.GraphNodeTypeMemFree] != 1 || kinds[driver.GraphNodeTypeKernel] != 2 {
		t.Fatalf("captured node kinds=%v", kinds)
	}
	executable, err := graph.InstantiateWithOptions(cudagraph.InstantiateOptions{RetainKernelOrder: true})
	physicalCheck(t, err)
	physicalClose(t, executable.Destroy)
	physicalCheck(t, graph.Destroy())
	_, names, err := executable.KernelOrder()
	physicalCheck(t, err)
	if len(names) != 2 || !strings.Contains(names[0], "fixture_project") || !strings.Contains(names[1], "fixture_decode") {
		t.Fatalf("snapshot=%v", names)
	}
	for _, mode := range []cuptibench.Mode{cuptibench.Whole, cuptibench.Kernels} {
		replay := cuptirange.UserReplay
		if mode == cuptibench.Kernels {
			replay = cuptirange.KernelReplay
		}
		t.Run(replay.String(), func(t *testing.T) {
			plan, err := cuptibench.New(p.dev, cuptibench.Config{Mode: mode, ReplayMode: replay, Operations: 1, CachePolicy: cuptirange.CachePreserve, Metrics: []string{physicalMetric}, MaxConfigBytes: 16 << 20, Limits: cuptirange.Limits{MaxRanges: 2, MaxRangeTreeNodes: 4, MaxCounterBytes: 32 << 20, MaxPasses: 8}})
			physicalCheck(t, err)
			physicalClose(t, plan.Close)
			prepares, runs := 0, 0
			w := cuptibench.Workload{View: "full", Graphs: []*cudagraph.Exec{executable}, Streams: p.streams[:1], Synchronize: p.synchronize, Preparation: "reset persistent state and output before each host pass; graph allocates temporary mid, launches project5 then decode2, frees mid; cache preserve", Prepare: func() error { prepares++; return p.prepare() }, Run: func(*cuptibench.Scope) error { runs++; return executable.Launch(p.streams[0]) }}
			for repeat := 0; repeat < 2; repeat++ {
				beforePrepare, beforeRun := prepares, runs
				r, err := plan.Collect(w)
				physicalCheck(t, err)
				raw, err := json.Marshal(r)
				physicalCheck(t, err)
				t.Logf("allocation graph result=%s", raw)
				q := r.Quality
				if !q.Decoded || !q.AllPassesSubmitted || q.DroppedRanges != 0 || q.PassesStarted == 0 || q.PassesStopped != q.PassesStarted || prepares-beforePrepare != int(q.PassesStarted) || runs-beforeRun != int(q.PassesStarted) {
					t.Fatalf("quality=%+v callbacks=%d/%d", q, prepares-beforePrepare, runs-beforeRun)
				}
				if mode == cuptibench.Whole {
					if len(r.Ranges) != 1 || physicalValue(t, r.Ranges[0]) != 7 {
						t.Fatalf("whole ranges=%+v", r.Ranges)
					}
				} else {
					if q.PassesStarted != 1 || len(r.Ranges) != 2 {
						t.Fatalf("kernel quality=%+v ranges=%+v", q, r.Ranges)
					}
					for i, v := range r.Ranges {
						wantName, wantOrdinal, wantCount := "fixture_project", "0", float64(5)
						if i == 1 {
							wantName, wantOrdinal, wantCount = "fixture_decode", "1", 2
						}
						if !strings.Contains(v.Name, wantName) || v.NativeName != wantOrdinal || physicalValue(t, v) != wantCount {
							t.Fatalf("kernel[%d]=%+v", i, v)
						}
					}
				}
				physicalCheck(t, p.synchronize())
				state, out := make([]uint32, physicalCapacity), make([]uint32, physicalCapacity)
				physicalCheck(t, p.state.CopyToHost(state))
				physicalCheck(t, p.out.CopyToHost(out))
				for i := range state {
					wantState, wantOut := uint32(i), uint32(0)
					if i < 2*physicalBlock {
						wantState++
						wantOut = uint32(2 * (i + 3))
					}
					if state[i] != wantState || out[i] != wantOut {
						t.Fatalf("output[%d]=%d/%d want=%d/%d", i, state[i], out[i], wantState, wantOut)
					}
				}
			}
		})
	}
}
