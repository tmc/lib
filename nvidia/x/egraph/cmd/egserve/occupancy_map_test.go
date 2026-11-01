package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOccupancyMapCliffTableAndConstants(t *testing.T) {
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
		t.Fatalf("BuildKernelStreamIterations: %v", err)
	}

	rep, err := BuildOccupancyPhaseSpaceReport("sm_121a", iters)
	if err != nil {
		t.Fatalf("BuildOccupancyPhaseSpaceReport: %v", err)
	}

	if rep.Arch != "sm_121a" {
		t.Fatalf("Arch = %q, want sm_121a", rep.Arch)
	}
	if rep.MaxRegistersPerSM != 65536 {
		t.Fatalf("MaxRegistersPerSM = %d, want 65536", rep.MaxRegistersPerSM)
	}
	if rep.MaxSharedMemPerSM != 233472 {
		t.Fatalf("MaxSharedMemPerSM = %d, want 233472", rep.MaxSharedMemPerSM)
	}
	if rep.MaxWarpsPerSM != 64 {
		t.Fatalf("MaxWarpsPerSM = %d, want 64", rep.MaxWarpsPerSM)
	}
	if rep.ThreadsPerCTA != 256 {
		t.Fatalf("ThreadsPerCTA = %d, want 256", rep.ThreadsPerCTA)
	}

	wantCliffs := []OccupancyCliffStep{
		{MaxRegsPerThread: 32, MaxSharedBytesCTA: 29184, ActiveCTAsPerSM: 8, ActiveWarpsPerSM: 64, OccupancyPct: 100.0, LimitingResource: "WARP_SLOTS"},
		{MaxRegsPerThread: 40, MaxSharedBytesCTA: 38912, ActiveCTAsPerSM: 6, ActiveWarpsPerSM: 48, OccupancyPct: 75.0, LimitingResource: "REGISTERS"},
		{MaxRegsPerThread: 64, MaxSharedBytesCTA: 58368, ActiveCTAsPerSM: 4, ActiveWarpsPerSM: 32, OccupancyPct: 50.0, LimitingResource: "REGISTERS"},
		{MaxRegsPerThread: 80, MaxSharedBytesCTA: 77824, ActiveCTAsPerSM: 3, ActiveWarpsPerSM: 24, OccupancyPct: 37.5, LimitingResource: "REGISTERS"},
		{MaxRegsPerThread: 128, MaxSharedBytesCTA: 116736, ActiveCTAsPerSM: 2, ActiveWarpsPerSM: 16, OccupancyPct: 25.0, LimitingResource: "REGISTERS"},
		{MaxRegsPerThread: 255, MaxSharedBytesCTA: 233472, ActiveCTAsPerSM: 1, ActiveWarpsPerSM: 8, OccupancyPct: 12.5, LimitingResource: "REGISTERS"},
	}

	if len(rep.CliffTable) != len(wantCliffs) {
		t.Fatalf("len(CliffTable) = %d, want %d", len(rep.CliffTable), len(wantCliffs))
	}
	for i, want := range wantCliffs {
		got := rep.CliffTable[i]
		if got != want {
			t.Fatalf("CliffTable[%d] = %+v, want %+v", i, got, want)
		}
	}
}

func TestOccupancyMapWarpSpecProfiles(t *testing.T) {
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
		t.Fatalf("BuildKernelStreamIterations: %v", err)
	}

	rep, err := BuildOccupancyPhaseSpaceReport("sm_121a", iters)
	if err != nil {
		t.Fatalf("BuildOccupancyPhaseSpaceReport: %v", err)
	}

	if len(rep.WarpSpecProfiles) != 3 {
		t.Fatalf("len(WarpSpecProfiles) = %d, want 3", len(rep.WarpSpecProfiles))
	}

	p0 := rep.WarpSpecProfiles[0]
	p1 := rep.WarpSpecProfiles[1]
	p2 := rep.WarpSpecProfiles[2]

	if p0.ScheduleID != "symmetric_8w" || p0.ProducerDMAWarps != 0 || p0.ConsumerMathWarps != 8 || p0.SpeedupVsSymmetric != 1.00 {
		t.Fatalf("unexpected symmetric_8w profile: %+v", p0)
	}

	if p1.ScheduleID != "ws_1dma_7math" || p1.ProducerDMAWarps != 1 || p1.ConsumerMathWarps != 7 {
		t.Fatalf("unexpected ws_1dma_7math profile: %+v", p1)
	}
	if p1.RegBudgetProducer != 40 || p1.RegBudgetConsumer != 72 {
		t.Fatalf("ws_1dma_7math reg budgets = (%d, %d), want (40, 72)", p1.RegBudgetProducer, p1.RegBudgetConsumer)
	}
	if !strings.Contains(p1.Label, "setmaxnreg.dec.sync.aligned.u32 40") ||
		!strings.Contains(p1.Label, "setmaxnreg.inc.sync.aligned.u32 72") ||
		!strings.Contains(p1.Label, "bar.sync 1, 224") {
		t.Fatalf("ws_1dma_7math label %q missing expected PTX directives", p1.Label)
	}
	if p1.SpeedupVsSymmetric <= 1.0 || p1.EstimatedLatencyUs >= p0.EstimatedLatencyUs {
		t.Fatalf("ws_1dma_7math latency/speedup = (%.2fus, %.2fx) should improve over symmetric (%.2fus)",
			p1.EstimatedLatencyUs, p1.SpeedupVsSymmetric, p0.EstimatedLatencyUs)
	}

	if p2.ScheduleID != "ws_2dma_6math_pingpong" || p2.ProducerDMAWarps != 2 || p2.ConsumerMathWarps != 6 {
		t.Fatalf("unexpected ws_2dma_6math_pingpong profile: %+v", p2)
	}
	if p2.SpeedupVsSymmetric <= p1.SpeedupVsSymmetric || p2.EstimatedLatencyUs >= p1.EstimatedLatencyUs {
		t.Fatalf("ws_2dma_6math_pingpong latency/speedup = (%.2fus, %.2fx) should improve over ws_1dma_7math (%.2fus, %.2fx)",
			p2.EstimatedLatencyUs, p2.SpeedupVsSymmetric, p1.EstimatedLatencyUs, p1.SpeedupVsSymmetric)
	}
}

func TestOccupancyMap24CandidatesAndMargins(t *testing.T) {
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
		t.Fatalf("BuildKernelStreamIterations: %v", err)
	}
	if len(iters) != 24 {
		t.Fatalf("len(iters) = %d, want 24", len(iters))
	}

	rep, err := BuildOccupancyPhaseSpaceReport("sm_121a", iters)
	if err != nil {
		t.Fatalf("BuildOccupancyPhaseSpaceReport: %v", err)
	}
	if len(rep.CandidateCoordinates) != 24 {
		t.Fatalf("len(CandidateCoordinates) = %d, want 24", len(rep.CandidateCoordinates))
	}

	// #00: 64 regs -> 4 CTAs/SM, 32 warps/SM, 50.0% occ, 0 regs margin to 64 cliff.
	c0 := rep.CandidateCoordinates[0]
	if c0.RegsPerThread != 64 || c0.ActiveCTAsPerSM != 4 || c0.ActiveWarpsPerSM != 32 || c0.OccupancyPct != 50.0 || c0.MarginToNextCliffRegs != 0 {
		t.Fatalf("Candidate #00 = %+v, want 64 regs, 4 CTAs, 32 warps, 50%%, margin 0", c0)
	}

	// #03: 168 regs (over-unrolled 16x spill) -> 1 CTA/SM, 8 warps/SM, 12.5% occ, 255-168 = 87 margin.
	c3 := rep.CandidateCoordinates[3]
	if c3.RegsPerThread != 168 || c3.ActiveCTAsPerSM != 1 || c3.ActiveWarpsPerSM != 8 || c3.OccupancyPct != 12.5 || c3.MarginToNextCliffRegs != 87 {
		t.Fatalf("Candidate #03 = %+v, want 168 regs, 1 CTA, 8 warps, 12.5%%, margin 87", c3)
	}

	// #17: 112 regs -> 2 CTAs/SM, 16 warps/SM, 25.0% occ, 128-112 = 16 margin.
	c17 := rep.CandidateCoordinates[17]
	if c17.RegsPerThread != 112 || c17.ActiveCTAsPerSM != 2 || c17.ActiveWarpsPerSM != 16 || c17.OccupancyPct != 25.0 || c17.MarginToNextCliffRegs != 16 {
		t.Fatalf("Candidate #17 = %+v, want 112 regs, 2 CTAs, 16 warps, 25.0%%, margin 16", c17)
	}

	// #19: 62 regs -> 4 CTAs/SM, 32 warps/SM, 50.0% occ, 64-62 = 2 margin.
	c19 := rep.CandidateCoordinates[19]
	if c19.RegsPerThread != 62 || c19.ActiveCTAsPerSM != 4 || c19.ActiveWarpsPerSM != 32 || c19.OccupancyPct != 50.0 || c19.MarginToNextCliffRegs != 2 {
		t.Fatalf("Candidate #19 = %+v, want 62 regs, 4 CTAs, 32 warps, 50.0%%, margin 2", c19)
	}

	// #23: 72 regs -> 3 CTAs/SM, 24 warps/SM, 37.5% occ, 80-72 = 8 margin.
	c23 := rep.CandidateCoordinates[23]
	if c23.RegsPerThread != 72 || c23.ActiveCTAsPerSM != 3 || c23.ActiveWarpsPerSM != 24 || c23.OccupancyPct != 37.5 || c23.MarginToNextCliffRegs != 8 {
		t.Fatalf("Candidate #23 = %+v, want 72 regs, 3 CTAs, 24 warps, 37.5%%, margin 8", c23)
	}

	// Verify shared-memory limiting resource when SharedBytesPerCTA exceeds step budget.
	smemBoundIter := KernelStreamIteration{
		Iteration: 99,
		Phase:     "synthetic_smem_bound",
		Promoted:  true,
		MaxRegs:   32,
		Stages: []AnnotatedStageSummary{
			{RegistersPerThread: 32, SharedBytes: 120000},
		},
	}
	smemRep, err := BuildOccupancyPhaseSpaceReport("sm_121a", []KernelStreamIteration{smemBoundIter})
	if err != nil {
		t.Fatalf("BuildOccupancyPhaseSpaceReport(smemBound): %v", err)
	}
	sc := smemRep.CandidateCoordinates[0]
	if sc.LimitingResource != "SHARED_MEM" || sc.ActiveCTAsPerSM != 1 || sc.OccupancyPct != 12.5 {
		t.Fatalf("synthetic smem-bound coord = %+v, want SHARED_MEM, 1 CTA/SM, 12.5%%", sc)
	}
}

func TestOccupancyMapHTTPHandler(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    2,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/kernels/occupancy-map?arch=sm_121a&mode=simple&bucket=1", nil)
	rec := httptest.NewRecorder()
	srv.handleKernelsOccupancyMap(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}

	var rep OccupancyPhaseSpaceReport
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if rep.Arch != "sm_121a" {
		t.Fatalf("Arch = %q, want sm_121a", rep.Arch)
	}
	if len(rep.CliffTable) != 6 {
		t.Fatalf("len(CliffTable) = %d, want 6", len(rep.CliffTable))
	}
	if len(rep.WarpSpecProfiles) != 3 {
		t.Fatalf("len(WarpSpecProfiles) = %d, want 3", len(rep.WarpSpecProfiles))
	}
	if len(rep.CandidateCoordinates) != 24 {
		t.Fatalf("len(CandidateCoordinates) = %d, want 24", len(rep.CandidateCoordinates))
	}

	// Verify POST returns 405 Method Not Allowed.
	postReq := httptest.NewRequest(http.MethodPost, "/v1/kernels/occupancy-map", nil)
	postRec := httptest.NewRecorder()
	srv.handleKernelsOccupancyMap(postRec, postReq)
	if postRec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST status = %d, want 405", postRec.Code)
	}
}
