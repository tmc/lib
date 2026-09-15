//go:build linux

package cuptibench_test

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

// Run each selector alone in a fresh process with -test.run exactly this test.
// The late arm performs its graph warm launch before any CUPTI/Plan API call.
// The early arm creates the plan first; graph construction and warm work match.
func TestPhysicalGraphWarmBeforeProfiling(t *testing.T) {
	order := os.Getenv("CUPTIBENCH_GRAPH_WARM_ORDER")
	if order == "" {
		t.Skip("fresh-process diagnostic: CUPTIBENCH_GRAPH_WARM_ORDER=early or late required")
	}
	if order != "early" && order != "late" {
		t.Fatal("unknown graph warm order")
	}
	streamMode := os.Getenv("CUPTIBENCH_GRAPH_STREAM")
	if streamMode == "" {
		streamMode = "nonblocking"
	}
	if streamMode != "blocking" && streamMode != "nonblocking" {
		t.Fatal("unknown graph stream mode")
	}
	p := newPhysicalPipeline(t, 5)
	if streamMode == "blocking" {
		stream, err := p.dev.NewStream()
		physicalCheck(t, err)
		physicalClose(t, stream.Destroy)
		p.streams[0] = stream
	}
	t.Logf("graph warm order=%s stream=%s", order, streamMode)
	var captured error
	graph, err := cudagraph.Capture(p.streams[0], func() {
		captured = cudalaunch.Kernel(p.project).Grid(5).Block(physicalBlock).On(p.streams[0]).Launch(p.state, p.mid)
		if captured == nil {
			captured = cudalaunch.Kernel(p.decode).Grid(2).Block(physicalBlock).On(p.streams[0]).Launch(p.state, p.mid, p.out)
		}
	})
	physicalCheck(t, err)
	physicalClose(t, graph.Destroy)
	physicalCheck(t, captured)
	executable, err := graph.InstantiateWithOptions(cudagraph.InstantiateOptions{RetainKernelOrder: true})
	physicalCheck(t, err)
	physicalClose(t, executable.Destroy)
	physicalCheck(t, graph.Destroy())
	config := cuptibench.Config{Mode: cuptibench.Kernels, ReplayMode: cuptirange.KernelReplay, Operations: 1, CachePolicy: cuptirange.CachePreserve, Metrics: []string{physicalMetric}, MaxConfigBytes: 16 << 20, Limits: cuptirange.Limits{MaxRanges: 16384, MaxRangeTreeNodes: 32768, MaxCounterBytes: 64 << 20, MaxPasses: 64}}
	var plan *cuptibench.Plan
	newPlan := func() {
		t.Logf("graph warm order=%s Plan.New", order)
		var err error
		plan, err = cuptibench.New(p.dev, config)
		if plan != nil {
			physicalClose(t, plan.Close)
		}
		physicalCheck(t, err)
	}
	if order == "early" {
		newPlan()
	}
	t.Logf("graph warm order=%s first graph launch", order)
	physicalCheck(t, p.prepare())
	physicalCheck(t, executable.Launch(p.streams[0]))
	physicalCheck(t, p.synchronize())
	checkGraphWarmOutput(t, p)
	if order == "late" {
		newPlan()
	}
	prepares, runs := 0, 0
	w := cuptibench.Workload{View: "full", Graphs: []*cudagraph.Exec{executable}, Streams: p.streams[:1], Synchronize: p.synchronize, Preparation: "reset state/mid/output after one preprofiling graph launch; project5 then decode2; plan initialization " + order + " relative to warm launch; stream " + streamMode + "; cache preserve", Prepare: func() error { prepares++; return p.prepare() }, Run: func(*cuptibench.Scope) error { runs++; return executable.Launch(p.streams[0]) }}
	result, collectErr := plan.Collect(w)
	raw, err := json.Marshal(result)
	physicalCheck(t, err)
	t.Logf("graph warm order=%s prepares=%d runs=%d result=%s native_error=%v", order, prepares, runs, raw, collectErr)
	// Record output before rejecting an incomplete counter capture. A successful
	// graph launch does not itself establish that native Stop/Decode succeeded.
	checkGraphWarmOutput(t, p)
	physicalCheck(t, collectErr)
	q := result.Quality
	if prepares != 1 || runs != 1 || q.PassesStarted != 1 || q.PassesStopped != 1 || !q.Decoded || !q.AllPassesSubmitted || q.DroppedRanges != 0 {
		t.Fatalf("callbacks=%d/%d quality=%+v", prepares, runs, q)
	}
	if len(result.Ranges) != 2 {
		t.Fatalf("ranges=%+v", result.Ranges)
	}
	for i, v := range result.Ranges {
		symbol, ordinal, count := "fixture_project", "0", float64(5)
		if i == 1 {
			symbol, ordinal, count = "fixture_decode", "1", 2
		}
		if !strings.Contains(v.Name, symbol) || v.NativeName != ordinal || physicalValue(t, v) != count {
			t.Fatalf("range[%d]=%+v", i, v)
		}
	}
}

func checkGraphWarmOutput(t *testing.T, p *physicalPipeline) {
	t.Helper()
	physicalCheck(t, p.synchronize())
	state, mid, out := make([]uint32, physicalCapacity), make([]uint32, physicalCapacity), make([]uint32, physicalCapacity)
	physicalCheck(t, p.state.CopyToHost(state))
	physicalCheck(t, p.mid.CopyToHost(mid))
	physicalCheck(t, p.out.CopyToHost(out))
	for i := range state {
		wantState, wantMid, wantOut := uint32(i), uint32(0), uint32(0)
		if i < 5*physicalBlock {
			wantMid = uint32(i + 3)
		}
		if i < 2*physicalBlock {
			wantState++
			wantOut = 2 * wantMid
		}
		if state[i] != wantState || mid[i] != wantMid || out[i] != wantOut {
			t.Fatalf("output[%d]=%d/%d/%d want=%d/%d/%d", i, state[i], mid[i], out[i], wantState, wantMid, wantOut)
		}
	}
	t.Log("warm graph output verified")
}
