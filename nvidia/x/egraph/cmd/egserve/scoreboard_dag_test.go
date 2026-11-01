package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestScoreboardDAGTrajectoryAndPipeClassification(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    2,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	iters, err := srv.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		t.Fatalf("BuildKernelStreamIterations(simple, 1): %v", err)
	}
	if len(iters) != 24 {
		t.Fatalf("len(iters) = %d, want 24", len(iters))
	}

	validPipes := map[string]bool{
		"LSU":        true,
		"MIO":        true,
		"SFU/MUFU":   true,
		"FMA/TENSOR": true,
		"CBU":        true,
	}
	validWriteBarriers := map[string]bool{
		"SB0": true,
		"SB1": true,
		"SB2": true,
		"SB3": true,
		"-":   true,
	}

	dags := make([]CandidateScoreboardDAG, len(iters))
	for i, it := range iters {
		dag := BuildCandidateScoreboardDAG(it)
		dags[i] = dag

		if dag.Iteration != i {
			t.Fatalf("iter %d: dag.Iteration = %d, want %d", i, dag.Iteration, i)
		}
		if dag.Phase != it.Phase {
			t.Fatalf("iter %d: dag.Phase = %q, want %q", i, dag.Phase, it.Phase)
		}
		if len(dag.Nodes) == 0 {
			t.Fatalf("iter %d: expected non-empty SASS schedule nodes", i)
		}
		if len(dag.Edges) == 0 {
			t.Fatalf("iter %d: expected non-empty scoreboard dependency edges", i)
		}
		if dag.BottleneckEdgeSummary == "" {
			t.Fatalf("iter %d: expected non-empty BottleneckEdgeSummary", i)
		}
		if dag.ActiveScoreboardBarriers <= 0 {
			t.Fatalf("iter %d: ActiveScoreboardBarriers = %d, want > 0", i, dag.ActiveScoreboardBarriers)
		}

		var sumStalls int
		var critNodes int
		var prevPC uint64
		var prevIssue int
		for nIdx, n := range dag.Nodes {
			if nIdx > 0 && n.PC <= prevPC {
				t.Fatalf("iter %d node %d: PC 0x%x not strictly increasing after 0x%x", i, nIdx, n.PC, prevPC)
			}
			prevPC = n.PC
			if nIdx > 0 && n.IssueCycle < prevIssue {
				t.Fatalf("iter %d node %d: IssueCycle %d < previous %d", i, nIdx, n.IssueCycle, prevIssue)
			}
			prevIssue = n.IssueCycle
			if n.ReadyCycle <= n.IssueCycle {
				t.Fatalf("iter %d node %d (%s): ReadyCycle %d <= IssueCycle %d", i, nIdx, n.Mnemonic, n.ReadyCycle, n.IssueCycle)
			}
			if !validPipes[n.PipeUnit] {
				t.Fatalf("iter %d node %d (%s): unexpected PipeUnit %q", i, nIdx, n.Mnemonic, n.PipeUnit)
			}
			if !validWriteBarriers[n.WriteBarrier] {
				t.Fatalf("iter %d node %d (%s): unexpected WriteBarrier %q", i, nIdx, n.Mnemonic, n.WriteBarrier)
			}
			sumStalls += n.StallCycles
			if n.OnCriticalPath {
				critNodes++
			}
		}
		if sumStalls != dag.TotalStallCycles {
			t.Fatalf("iter %d: sum(node.StallCycles)=%d != TotalStallCycles=%d", i, sumStalls, dag.TotalStallCycles)
		}
		if critNodes < 2 {
			t.Fatalf("iter %d: critNodes = %d, want >= 2", i, critNodes)
		}

		var critEdges int
		for eIdx, e := range dag.Edges {
			if e.ToPC <= e.FromPC {
				t.Fatalf("iter %d edge %d: ToPC 0x%x <= FromPC 0x%x", i, eIdx, e.ToPC, e.FromPC)
			}
			if e.LatencyCycles <= 0 {
				t.Fatalf("iter %d edge %d: LatencyCycles = %d, want > 0", i, eIdx, e.LatencyCycles)
			}
			if e.BarrierID == "" || e.ResolutionAction == "" {
				t.Fatalf("iter %d edge %d: empty BarrierID=%q or ResolutionAction=%q", i, eIdx, e.BarrierID, e.ResolutionAction)
			}
			if e.IsCriticalPath {
				critEdges++
			}
		}
		if critEdges == 0 {
			t.Fatalf("iter %d: expected at least 1 critical-path edge", i)
		}
	}

	// Verify #00 baseline_scalar_f32: TotalCriticalCycles ~ 420, PipelineOverlapPct ~ 18.5%.
	d0 := dags[0]
	if d0.TotalCriticalCycles < 400 || d0.TotalCriticalCycles > 440 {
		t.Fatalf("#00 TotalCriticalCycles = %d, want ~420 (400..440)", d0.TotalCriticalCycles)
	}
	if d0.PipelineOverlapPct < 16.0 || d0.PipelineOverlapPct > 21.0 {
		t.Fatalf("#00 PipelineOverlapPct = %.1f%%, want ~18.5%%", d0.PipelineOverlapPct)
	}
	if !strings.Contains(d0.BottleneckEdgeSummary, "SB0 (LOAD_GLOBAL)") {
		t.Fatalf("#00 BottleneckEdgeSummary = %q, want SB0 (LOAD_GLOBAL) bottleneck", d0.BottleneckEdgeSummary)
	}

	// Verify #03 local register spill candidate: TotalCriticalCycles > 650.
	d3 := dags[3]
	if d3.TotalCriticalCycles <= 650 {
		t.Fatalf("#03 TotalCriticalCycles = %d, want > 650 on register spill candidate", d3.TotalCriticalCycles)
	}
	if !strings.Contains(d3.BottleneckEdgeSummary, "SB0 (LOAD_GLOBAL)") {
		t.Fatalf("#03 BottleneckEdgeSummary = %q, want SB0 (LOAD_GLOBAL) local spill bottleneck", d3.BottleneckEdgeSummary)
	}

	// Verify #18..#23 pipelined candidates: TotalCriticalCycles <= 96 and PipelineOverlapPct >= 84.0%.
	for idx := 18; idx <= 23; idx++ {
		d := dags[idx]
		if d.TotalCriticalCycles > 96 {
			t.Fatalf("#%02d (%s) TotalCriticalCycles = %d, want <= 96", idx, d.Phase, d.TotalCriticalCycles)
		}
		if d.PipelineOverlapPct < 84.0 {
			t.Fatalf("#%02d (%s) PipelineOverlapPct = %.1f%%, want >= 84.0%%", idx, d.Phase, d.PipelineOverlapPct)
		}
		hasSB3 := false
		for _, e := range d.Edges {
			if e.BarrierID == "SB3 (CP_ASYNC)" {
				hasSB3 = true
				break
			}
		}
		if !hasSB3 {
			t.Fatalf("#%02d (%s) expected active SB3 (CP_ASYNC) edge", idx, d.Phase)
		}
	}
}

func TestScoreboardDAGStagesOnlyAndSpill32(t *testing.T) {
	// Construct a synthetic iteration with only Stages populated (LineAnnotations nil)
	// and explicit STL.32 / LDL.32 spill instructions to verify fallback collection and pipe/barrier mapping.
	it := KernelStreamIteration{
		Iteration:       3,
		Phase:           "unroll_16_reg_spill_104",
		TotalSpillBytes: 104,
		Stages: []AnnotatedStageSummary{
			{
				StageIndex: 0,
				StageName:  "s0_rmsnorm",
				StartLine:  1,
				EndLine:    14,
				SASSInsts: []AnnotatedSASSInst{
					{PC: 0x1000, Mnemonic: "S2R", Operands: "R0, SR_TID.X", WaitMask: "--:-:-", ControlStall: 2, SourceLine: 2},
					{PC: 0x1010, Mnemonic: "STL.32", Operands: "[R1+0x10], R4", WaitMask: "--:-:1", ControlStall: 6, ScoreboardWeight: 12, SourceLine: 6},
					{PC: 0x1020, Mnemonic: "LDL.32", Operands: "R4, [R1+0x10]", WaitMask: "01:-:1", ControlStall: 8, ScoreboardWeight: 14, SourceLine: 7},
					{PC: 0x1030, Mnemonic: "LDS.128", Operands: "R8, [R3]", WaitMask: "--:-:2", ControlStall: 4, ScoreboardWeight: 8, SourceLine: 9},
					{PC: 0x1040, Mnemonic: "MUFU.RSQ", Operands: "R10, R8", WaitMask: "02:-:1", ControlStall: 4, ScoreboardWeight: 6, SourceLine: 11},
					{PC: 0x1050, Mnemonic: "CP.ASYNC.CG", Operands: "[R3], [R2.64]", WaitMask: "--:-:1", ControlStall: 2, ScoreboardWeight: 4, SourceLine: 12},
					{PC: 0x1060, Mnemonic: "FFMA", Operands: "R12, R4, R10, R12", WaitMask: "03:-:-", ControlStall: 2, ScoreboardWeight: 4, SourceLine: 13},
					{PC: 0x1070, Mnemonic: "EXIT", Operands: "", WaitMask: "01:-:-", ControlStall: 5, SourceLine: 14},
				},
			},
		},
	}
	dag := BuildCandidateScoreboardDAG(it)
	if dag.TotalCriticalCycles <= 650 {
		t.Fatalf("TotalCriticalCycles = %d, want > 650", dag.TotalCriticalCycles)
	}
	if len(dag.Nodes) != 8 {
		t.Fatalf("len(dag.Nodes) = %d, want 8", len(dag.Nodes))
	}
	if dag.ActiveScoreboardBarriers != 4 {
		t.Fatalf("ActiveScoreboardBarriers = %d, want 4 (SB0, SB1, SB2, SB3)", dag.ActiveScoreboardBarriers)
	}
}

func TestScoreboardDAGHTTPHandler(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    2,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/kernels/scoreboard-dag", srv.handleKernelsScoreboardDAG)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// 1. Default request -> Champion #23.
	resp, err := ts.Client().Get(ts.URL + "/v1/kernels/scoreboard-dag")
	if err != nil {
		t.Fatalf("GET /v1/kernels/scoreboard-dag: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var dag23 CandidateScoreboardDAG
	if err := json.NewDecoder(resp.Body).Decode(&dag23); err != nil {
		t.Fatalf("decode CandidateScoreboardDAG: %v", err)
	}
	if dag23.Iteration != 23 {
		t.Fatalf("default Iteration = %d, want 23", dag23.Iteration)
	}
	if dag23.TotalCriticalCycles > 96 || dag23.PipelineOverlapPct < 84.0 {
		t.Fatalf("champion #23 metrics out of bounds: crit=%d overlap=%.1f%%", dag23.TotalCriticalCycles, dag23.PipelineOverlapPct)
	}
	if len(dag23.AllSummaries) != 24 {
		t.Fatalf("len(AllSummaries) = %d, want 24", len(dag23.AllSummaries))
	}

	// 2. Explicit ?iter=3 -> spill candidate #03.
	resp3, err := ts.Client().Get(ts.URL + "/v1/kernels/scoreboard-dag?iter=3")
	if err != nil {
		t.Fatalf("GET ?iter=3: %v", err)
	}
	defer resp3.Body.Close()
	var dag3 CandidateScoreboardDAG
	if err := json.NewDecoder(resp3.Body).Decode(&dag3); err != nil {
		t.Fatalf("decode ?iter=3: %v", err)
	}
	if dag3.Iteration != 3 || dag3.TotalCriticalCycles <= 650 {
		t.Fatalf("?iter=3 got Iteration=%d TotalCriticalCycles=%d, want 3 and >650", dag3.Iteration, dag3.TotalCriticalCycles)
	}

	// 3. ?all=1 -> top-level CandidateScoreboardDAG plus AllDAGs (len 24).
	respAll, err := ts.Client().Get(ts.URL + "/v1/kernels/scoreboard-dag?all=1")
	if err != nil {
		t.Fatalf("GET ?all=1: %v", err)
	}
	defer respAll.Body.Close()
	var dagAll CandidateScoreboardDAG
	if err := json.NewDecoder(respAll.Body).Decode(&dagAll); err != nil {
		t.Fatalf("decode ?all=1: %v", err)
	}
	if len(dagAll.AllDAGs) != 24 {
		t.Fatalf("len(AllDAGs) = %d, want 24", len(dagAll.AllDAGs))
	}
	if dagAll.AllDAGs[0].TotalCriticalCycles < 400 || dagAll.AllDAGs[23].TotalCriticalCycles > 96 {
		t.Fatalf("unexpected trajectory bounds in AllDAGs: #00=%d, #23=%d",
			dagAll.AllDAGs[0].TotalCriticalCycles, dagAll.AllDAGs[23].TotalCriticalCycles)
	}
}
