package cudahostemu

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// gemm is a main loop whose MMA (512 cycles on the sm_100a model) outlasts
// its 48 KiB TMA load (384 cycles), so a deep enough pipeline is
// tensor-bound.
func gemm(stages int, specialized, waitInLoop bool) GEMMTimeline {
	return GEMMTimeline{M: 128, N: 256, K: 64, KSteps: 32, Stages: stages, Specialized: specialized, WaitLoadInLoop: waitInLoop}
}

func simulate(t *testing.T, g GEMMTimeline) TimelineSummary {
	t.Helper()
	p, err := GEMMTimelineProgram(g)
	if err != nil {
		t.Fatal(err)
	}
	s, err := SimulateTimeline(p, "sm_100a")
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func hasFinding(s TimelineSummary, kind string) bool {
	for _, f := range s.Findings {
		if f.Kind == kind {
			return true
		}
	}
	return false
}

func TestTimelinePipelineOverlap(t *testing.T) {
	serial := simulate(t, gemm(1, false, false))
	if serial.TMAOverlapPct != 0 {
		t.Errorf("1-stage TMAOverlapPct = %.1f, want 0", serial.TMAOverlapPct)
	}
	if !hasFinding(serial, "consumer-stall") {
		t.Errorf("1-stage: no consumer-stall finding (stall %d of %d cycles)", serial.ConsumerStallCyc, serial.TotalCycles)
	}

	piped := simulate(t, gemm(3, true, false))
	if piped.TMAOverlapPct < 80 {
		t.Errorf("3-stage TMAOverlapPct = %.1f, want >= 80", piped.TMAOverlapPct)
	}
	if piped.TensorUtilPct < 75 {
		t.Errorf("3-stage TensorUtilPct = %.1f, want >= 75", piped.TensorUtilPct)
	}
	if len(piped.Findings) != 0 {
		t.Errorf("3-stage findings: %+v", piped.Findings)
	}
	if serial.ConsumerStallCyc <= 4*piped.ConsumerStallCyc {
		t.Errorf("consumer stall: 1-stage %d, 3-stage %d", serial.ConsumerStallCyc, piped.ConsumerStallCyc)
	}

	// Deeper pipelines never take longer.
	var prev uint64
	for _, g := range []GEMMTimeline{gemm(1, false, false), gemm(2, false, false), gemm(2, true, false), gemm(3, true, false)} {
		s := simulate(t, g)
		t.Logf("stages=%d specialized=%v: %d cycles, tensor %.1f%%, overlap %.1f%%, stall %d",
			g.Stages, g.Specialized, s.TotalCycles, s.TensorUtilPct, s.TMAOverlapPct, s.ConsumerStallCyc)
		if prev != 0 && s.TotalCycles > prev {
			t.Errorf("stages=%d specialized=%v: %d cycles, more than %d", g.Stages, g.Specialized, s.TotalCycles, prev)
		}
		prev = s.TotalCycles
	}
}

// TestTimelineWaitLoadInLoop is the mutant check: reading the accumulator
// inside the K loop serializes the consumer on every MMA.
func TestTimelineWaitLoadInLoop(t *testing.T) {
	good := simulate(t, gemm(3, true, false))
	bad := simulate(t, gemm(3, true, true))
	t.Logf("consumer stall %d, with wait::ld in loop %d", good.ConsumerStallCyc, bad.ConsumerStallCyc)
	if bad.ConsumerStallCyc < 2*good.ConsumerStallCyc {
		t.Errorf("stall with wait::ld in loop = %d, want >= 2 x %d", bad.ConsumerStallCyc, good.ConsumerStallCyc)
	}
	if !hasFinding(bad, "wait-ld-after-mma") {
		t.Error("no wait-ld-after-mma finding")
	}
	if hasFinding(good, "wait-ld-after-mma") {
		t.Error("wait-ld-after-mma finding without the mutation")
	}
}

// TestTimelineCycles checks the model's arithmetic on one load and one MMA.
func TestTimelineCycles(t *testing.T) {
	p := TimelineProgram{
		Barriers: []int{1, 1},
		Warps: []TimelineWarp{{ID: 0, Role: PipeConsumerWarp, Ops: []TimelineOp{
			{Kind: OpTMALoad, Bar: 0, Bytes: 1024},
			{Kind: OpWait, Bar: 0},
			{Kind: OpMMA, M: 64, N: 64, K: 64},
			{Kind: OpCommit, Bar: 1},
			{Kind: OpWait, Bar: 1},
		}}},
	}
	s, err := SimulateTimeline(p, "sm_100a")
	if err != nil {
		t.Fatal(err)
	}
	// TMA: issued at 16, done at 16+380. The wait wakes at 396 and
	// costs 2; the MMA issues at 406 and runs 2*64^3/8192 = 64 cycles.
	// The commit arrives at 470; the final wait ends at 472. Only the
	// first wait is a bubble: the MMA is in flight during the second.
	if s.TotalCycles != 472 || s.TensorCoreActiveCyc != 64 || s.TMAActiveCyc != 380 {
		t.Errorf("total %d, tensor %d, tma %d; want 472, 64, 380", s.TotalCycles, s.TensorCoreActiveCyc, s.TMAActiveCyc)
	}
	if s.ConsumerStallCyc != 380 {
		t.Errorf("stall = %d, want 380", s.ConsumerStallCyc)
	}
	if len(s.Flows) != 2 || s.Flows[0].FromTrack != TrackTMAEngine || s.Flows[0].ToCycle != 396 || s.Flows[1].FromTrack != TrackTensorCore {
		t.Errorf("flows = %+v", s.Flows)
	}
}

func TestTimelineTMEM(t *testing.T) {
	warp := func(ops ...TimelineOp) TimelineProgram {
		return TimelineProgram{Warps: []TimelineWarp{{ID: 4, Role: PipeConsumerWarp, Ops: ops}}}
	}
	s, err := SimulateTimeline(warp(
		TimelineOp{Kind: OpAlloc, Cols: 128, Label: "a"},
		TimelineOp{Kind: OpAlloc, Cols: 256, Label: "b"},
		TimelineOp{Kind: OpDealloc, Label: "a"},
		TimelineOp{Kind: OpAlloc, Cols: 64, Label: "c"},
		TimelineOp{Kind: OpCompute, Cycles: 100},
	), "sm_100a")
	if err != nil {
		t.Fatal(err)
	}
	got := fmt.Sprint(s.TMEMMap)
	want := "[{0 128 0 8 a} {256 256 4 116 b} {0 64 12 116 c}]"
	if got != want {
		t.Errorf("TMEMMap = %s, want %s", got, want)
	}
	if s.PeakTMEMCols != 384 {
		t.Errorf("PeakTMEMCols = %d, want 384", s.PeakTMEMCols)
	}

	errs := []struct {
		name string
		p    TimelineProgram
		arch string
	}{
		{"full", warp(TimelineOp{Kind: OpAlloc, Cols: 512, Label: "a"}, TimelineOp{Kind: OpAlloc, Cols: 32, Label: "b"}), "sm_100a"},
		{"not power of two", warp(TimelineOp{Kind: OpAlloc, Cols: 96}), "sm_100a"},
		{"too small", warp(TimelineOp{Kind: OpAlloc, Cols: 16}), "sm_100a"},
		{"no tmem", warp(TimelineOp{Kind: OpAlloc, Cols: 32}), "sm_90a"},
		{"unallocated", warp(TimelineOp{Kind: OpDealloc, Label: "x"}), "sm_100a"},
		{"deadlock", warp(TimelineOp{Kind: OpWait, Bar: 0}), "sm_100a"},
		{"bad barrier", warp(TimelineOp{Kind: OpArrive, Bar: 3}), "sm_100a"},
		{"bad arch", warp(), "sm_75"},
	}
	for _, e := range errs {
		if e.name == "deadlock" {
			e.p.Barriers = []int{1}
		}
		if _, err := SimulateTimeline(e.p, e.arch); err == nil {
			t.Errorf("%s: SimulateTimeline succeeded", e.name)
		}
	}
}

func TestTimelinePerfettoJSON(t *testing.T) {
	s := simulate(t, gemm(3, true, false))
	var buf bytes.Buffer
	if err := s.WritePerfettoJSON(&buf); err != nil {
		t.Fatal(err)
	}
	var doc struct {
		TraceEvents []struct {
			Name string         `json:"name"`
			Ph   string         `json:"ph"`
			Pid  int            `json:"pid"`
			Tid  int            `json:"tid"`
			Ts   uint64         `json:"ts"`
			Dur  *uint64        `json:"dur"`
			ID   uint64         `json:"id"`
			Args map[string]any `json:"args"`
		} `json:"traceEvents"`
		OtherData map[string]any `json:"otherData"`
	}
	if err := json.Unmarshal(buf.Bytes(), &doc); err != nil {
		t.Fatal(err)
	}
	count := map[string]int{}
	starts := map[uint64]uint64{}
	threads := map[string]bool{}
	for _, e := range doc.TraceEvents {
		count[e.Ph]++
		switch e.Ph {
		case "X":
			if e.Dur == nil {
				t.Errorf("X event %q without dur", e.Name)
			}
		case "s":
			starts[e.ID] = e.Ts
		case "f":
			ts, ok := starts[e.ID]
			if !ok {
				t.Errorf("flow %d finishes before it starts", e.ID)
			} else if e.Ts < ts {
				t.Errorf("flow %d: finish %d before start %d", e.ID, e.Ts, ts)
			}
		case "M":
			if e.Name == "thread_name" {
				threads[e.Args["name"].(string)] = true
			}
		case "C":
			if _, ok := e.Args["allocated"]; !ok {
				t.Errorf("counter without allocated value")
			}
		}
	}
	for _, ph := range []string{"X", "s", "f", "C", "M"} {
		if count[ph] == 0 {
			t.Errorf("no %q events", ph)
		}
	}
	if count["s"] != count["f"] {
		t.Errorf("%d flow starts, %d finishes", count["s"], count["f"])
	}
	for _, name := range []string{"Producer Warp 0", "Consumer Warp 4", "TMA Engine", "Tensor Core"} {
		if !threads[name] {
			t.Errorf("no thread named %q", name)
		}
	}
	if !strings.Contains(buf.String(), "STALL_MBARRIER") {
		t.Error("no stall spans")
	}
	if doc.OtherData["time_unit"] != "cycle" {
		t.Errorf("otherData = %v", doc.OtherData)
	}
}

func TestIntervals(t *testing.T) {
	u := union([]interval{{5, 10}, {0, 3}, {2, 4}, {10, 12}, {20, 20}})
	if fmt.Sprint(u) != "[{0 4} {5 12}]" {
		t.Errorf("union = %v", u)
	}
	x := intersect(u, []interval{{3, 6}, {11, 30}})
	if fmt.Sprint(x) != "[{3 4} {5 6} {11 12}]" || length(x) != 3 {
		t.Errorf("intersect = %v", x)
	}
}

func ExampleSimulateTimeline() {
	p, err := GEMMTimelineProgram(GEMMTimeline{M: 128, N: 256, K: 64, KSteps: 32, Stages: 3, Specialized: true})
	if err != nil {
		panic(err)
	}
	s, err := SimulateTimeline(p, "sm_100a")
	if err != nil {
		panic(err)
	}
	fmt.Printf("%d cycles, tensor core %.0f%% busy, %.0f%% of TMA time overlapped\n", s.TotalCycles, s.TensorUtilPct, s.TMAOverlapPct)
	// Output: 16823 cycles, tensor core 97% busy, 97% of TMA time overlapped
}
