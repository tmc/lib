package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestBankLivenessSimulateWarpBankAccess(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{Arch: "sm_121a"})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}
	iters, err := srv.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		t.Fatalf("BuildKernelStreamIterations: %v", err)
	}
	if len(iters) != 24 {
		t.Fatalf("expected 24 simple iterations, got %d", len(iters))
	}

	// #00..#04: tree_reduce_unswizzled (2-way bank fold)
	for idx := 0; idx <= 4; idx++ {
		ba := SimulateWarpBankAccess(iters[idx])
		if ba.SwizzleMode != "tree_reduce_unswizzled" || ba.MaxWayConflict != 2 || ba.ConflictFree {
			t.Errorf("iter #%02d: got swizzle=%q conflict=%d free=%v, want tree_reduce_unswizzled/2/false",
				idx, ba.SwizzleMode, ba.MaxWayConflict, ba.ConflictFree)
		}
	}

	// #05..#09: uniform_broadcast (1-way conflict-free)
	for idx := 5; idx <= 9; idx++ {
		ba := SimulateWarpBankAccess(iters[idx])
		if ba.SwizzleMode != "uniform_broadcast" || ba.MaxWayConflict != 1 || !ba.ConflictFree {
			t.Errorf("iter #%02d: got swizzle=%q conflict=%d free=%v, want uniform_broadcast/1/true",
				idx, ba.SwizzleMode, ba.MaxWayConflict, ba.ConflictFree)
		}
	}

	// #10: linear_unswizzled_8way (8-way LDS.128 bank conflict)
	ba10 := SimulateWarpBankAccess(iters[10])
	if ba10.MaxWayConflict != 8 || ba10.ConflictFree || ba10.SwizzleMode != "linear_unswizzled_8way" {
		t.Fatalf("iter #10: got %+v, want MaxWayConflict=8, ConflictFree=false, SwizzleMode=linear_unswizzled_8way", ba10)
	}
	for lane := 0; lane < 32; lane++ {
		wantBank := (lane * 16) % 32
		if ba10.LaneToBank[lane] != wantBank {
			t.Errorf("iter #10 lane %d: got bank %d, want %d", lane, ba10.LaneToBank[lane], wantBank)
		}
	}

	// #11: pad_stride33_scalar (ConflictFree=true, WavefrontTransactions=4)
	ba11 := SimulateWarpBankAccess(iters[11])
	if !ba11.ConflictFree || ba11.MaxWayConflict != 1 || ba11.WavefrontTransactions != 4 || ba11.SwizzleMode != "pad_stride33_scalar" {
		t.Fatalf("iter #11: got %+v, want ConflictFree=true, MaxWayConflict=1, WavefrontTransactions=4, SwizzleMode=pad_stride33_scalar", ba11)
	}

	// #12: Swizzle64B (swizzle_64b_xor, MaxWayConflict=2)
	ba12 := SimulateWarpBankAccess(iters[12])
	if ba12.SwizzleMode != "swizzle_64b_xor" || ba12.MaxWayConflict != 2 || ba12.WavefrontTransactions != 2 || ba12.ConflictFree {
		t.Fatalf("iter #12: got %+v, want SwizzleMode=swizzle_64b_xor, MaxWayConflict=2, WavefrontTransactions=2, ConflictFree=false", ba12)
	}
	if !strings.Contains(ba12.AccessPattern, "Swizzle64B") && !strings.Contains(ba12.Summary, "Swizzle64B") {
		t.Errorf("iter #12: expected Swizzle64B description in AccessPattern or Summary, got %q / %q", ba12.AccessPattern, ba12.Summary)
	}

	// #13..#23: Swizzle128B LOP3.LUT bijection (MaxWayConflict=1, all 32 banks hit exactly once)
	for idx := 13; idx <= 23; idx++ {
		ba := SimulateWarpBankAccess(iters[idx])
		if ba.SwizzleMode != "swizzle_128b_lop3" || ba.MaxWayConflict != 1 || ba.WavefrontTransactions != 1 || !ba.ConflictFree {
			t.Fatalf("iter #%02d: got %+v, want swizzle_128b_lop3, MaxWayConflict=1, WavefrontTransactions=1, ConflictFree=true", idx, ba)
		}
		for b := 0; b < 32; b++ {
			if ba.BankHitCounts[b] != 1 {
				t.Fatalf("iter #%02d bank %d: got hit count %d, want 1", idx, b, ba.BankHitCounts[b])
			}
		}
	}
}

func TestBankLivenessAnalyzeCandidate(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{Arch: "sm_121a"})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}
	iters, err := srv.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		t.Fatalf("BuildKernelStreamIterations: %v", err)
	}

	for idx, it := range iters {
		rep := AnalyzeCandidateBankAndLiveness(it)
		if len(rep.LineLiveness) != 30 {
			t.Fatalf("iter #%02d: got %d liveness lines, want 30", idx, len(rep.LineLiveness))
		}
		if rep.PeakLiveRegs != it.MaxRegs {
			t.Errorf("iter #%02d: PeakLiveRegs=%d, want MaxRegs=%d", idx, rep.PeakLiveRegs, it.MaxRegs)
		}
		if rep.PeakPressureLine != 23 {
			t.Errorf("iter #%02d: PeakPressureLine=%d, want 23", idx, rep.PeakPressureLine)
		}
		if rep.LineLiveness[22].LiveRegs != it.MaxRegs {
			t.Errorf("iter #%02d line 23: LiveRegs=%d, want %d", idx, rep.LineLiveness[22].LiveRegs, it.MaxRegs)
		}
	}

	// #03: 168 regs, 32B local spill on lines 19..24
	rep03 := AnalyzeCandidateBankAndLiveness(iters[3])
	if rep03.PeakLiveRegs != 168 || rep03.SpillBytes != 32 {
		t.Fatalf("iter #03: got PeakLiveRegs=%d SpillBytes=%d, want 168 and 32", rep03.PeakLiveRegs, rep03.SpillBytes)
	}
	for _, ll := range rep03.LineLiveness {
		wantSpill := ll.Line >= 19 && ll.Line <= 24
		if ll.TriggersLocalSpill != wantSpill {
			t.Errorf("iter #03 line %d: TriggersLocalSpill=%v, want %v", ll.Line, ll.TriggersLocalSpill, wantSpill)
		}
	}
	if !rep03.LineLiveness[22].CrossesOccupancyCliff {
		t.Errorf("iter #03 line 23: expected CrossesOccupancyCliff=true at 168 regs")
	}

	// #17: 112 regs, CrossesOccupancyCliff == true on peak line
	rep17 := AnalyzeCandidateBankAndLiveness(iters[17])
	if rep17.PeakLiveRegs != 112 {
		t.Fatalf("iter #17: got PeakLiveRegs=%d, want 112", rep17.PeakLiveRegs)
	}
	if !rep17.LineLiveness[22].CrossesOccupancyCliff {
		t.Errorf("iter #17 line 23: expected CrossesOccupancyCliff=true at 112 regs")
	}

	// #23: promoted champion, ConflictFree == true
	rep23 := AnalyzeCandidateBankAndLiveness(iters[23])
	if !rep23.BankAnalysis.ConflictFree || rep23.BankAnalysis.MaxWayConflict != 1 {
		t.Fatalf("iter #23: expected ConflictFree=true and MaxWayConflict=1, got %+v", rep23.BankAnalysis)
	}
}

func TestBankLivenessHTTPHandler(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{Arch: "sm_121a"})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	// 1. Full trajectory response
	reqAll := httptest.NewRequest(http.MethodGet, "/v1/kernels/bank-liveness", nil)
	recAll := httptest.NewRecorder()
	srv.handleKernelsBankLiveness(recAll, reqAll)
	if recAll.Code != http.StatusOK {
		t.Fatalf("GET /v1/kernels/bank-liveness status=%d body=%s", recAll.Code, recAll.Body.String())
	}
	var respAll TrajectoryBankLivenessResponse
	if err := json.Unmarshal(recAll.Body.Bytes(), &respAll); err != nil {
		t.Fatalf("decode full response: %v", err)
	}
	if respAll.Arch != "sm_121a" {
		t.Errorf("Arch=%q, want sm_121a", respAll.Arch)
	}
	if len(respAll.Reports) != 24 {
		t.Fatalf("len(Reports)=%d, want 24", len(respAll.Reports))
	}

	// 2. Single candidate query (?iter=10)
	req10 := httptest.NewRequest(http.MethodGet, "/v1/kernels/bank-liveness?iter=10", nil)
	rec10 := httptest.NewRecorder()
	srv.handleKernelsBankLiveness(rec10, req10)
	if rec10.Code != http.StatusOK {
		t.Fatalf("GET /v1/kernels/bank-liveness?iter=10 status=%d body=%s", rec10.Code, rec10.Body.String())
	}
	var resp10 TrajectoryBankLivenessResponse
	if err := json.Unmarshal(rec10.Body.Bytes(), &resp10); err != nil {
		t.Fatalf("decode iter=10 response: %v", err)
	}
	if len(resp10.Reports) != 1 {
		t.Fatalf("len(Reports)=%d, want 1", len(resp10.Reports))
	}
	if resp10.Reports[0].Iteration != 10 || resp10.Reports[0].BankAnalysis.MaxWayConflict != 8 {
		t.Errorf("iter=10 report mismatch: %+v", resp10.Reports[0])
	}
}
