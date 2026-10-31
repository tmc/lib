package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEGraphTraceBuildAll24Candidates(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	iters := srv.buildSimpleKernelStreamIterations(1)
	if len(iters) != 24 {
		t.Fatalf("len(iters) = %d, want 24", len(iters))
	}

	rep, err := BuildEGraphDerivationTrace("sm_121a", iters)
	if err != nil {
		t.Fatalf("BuildEGraphDerivationTrace: %v", err)
	}
	if rep.Arch != "sm_121a" {
		t.Fatalf("Arch = %q, want sm_121a", rep.Arch)
	}
	if rep.KernelName == "" {
		t.Fatalf("expected non-empty KernelName")
	}
	if rep.SaturationRounds <= 0 || rep.TotalEClasses < 24 || rep.TotalENodes <= rep.TotalEClasses {
		t.Fatalf("invalid saturation stats: rounds=%d classes=%d nodes=%d",
			rep.SaturationRounds, rep.TotalEClasses, rep.TotalENodes)
	}
	if len(rep.Derivations) != 24 {
		t.Fatalf("len(Derivations) = %d, want 24", len(rep.Derivations))
	}
	if rep.CalibratorR2After <= rep.CalibratorR2Before {
		t.Fatalf("CalibratorR2After (%.3f) <= CalibratorR2Before (%.3f), want R2 improvement",
			rep.CalibratorR2After, rep.CalibratorR2Before)
	}
	if rep.SelectedChampionIdx != 23 {
		t.Fatalf("SelectedChampionIdx = %d, want 23", rep.SelectedChampionIdx)
	}

	wantRules := []string{
		"init-scalar-graph",
		"loop-unroll-2x",
		"loop-unroll-8x",
		"loop-unroll-16x-spill",
		"vectorize-ldg-float2",
		"vectorize-ldg-float4",
		"transpose-weight-strided",
		"reduce-tree-to-warp-shfl",
		"elide-cross-warp-sum",
		"vectorize-gemv-float4",
		"smem-tile-lane-striped",
		"smem-pad-stride33",
		"smem-swizzle-64b-xor",
		"smem-swizzle-128b-lop3",
		"elide-stage-bar-sync",
		"predicate-bar-sync-halfcta",
		"register-tile-2x4",
		"register-tile-4x4-overreg",
		"launch-bounds-256-4cta",
		"silu-mufu-ex2-rcp",
		"cp-async-unawaited",
		"cp-async-2stage-pipeline",
		"l2-evict-last-residency",
		"cugraphexec-zero-host-hotswap",
	}

	for i, d := range rep.Derivations {
		it := iters[i]
		if d.Iteration != i {
			t.Fatalf("derivation[%d].Iteration = %d, want %d", i, d.Iteration, i)
		}
		if d.RuleName != wantRules[i] {
			t.Fatalf("derivation[%d].RuleName = %q, want %q", i, d.RuleName, wantRules[i])
		}
		if d.RuleCategory == "" {
			t.Fatalf("derivation[%d] empty RuleCategory", i)
		}
		if d.PatternLHS == "" || d.PatternRHS == "" {
			t.Fatalf("derivation[%d] empty pattern: LHS=%q RHS=%q", i, d.PatternLHS, d.PatternRHS)
		}
		if d.Pruned != !it.Promoted {
			t.Fatalf("derivation[%d] Pruned = %t, want %t (!Promoted)", i, d.Pruned, !it.Promoted)
		}
		if d.Pruned && d.PruneReason == "" {
			t.Fatalf("derivation[%d] is Pruned=true but PruneReason is empty", i)
		}
		if !d.Pruned && d.PruneReason != "" {
			t.Fatalf("derivation[%d] is Pruned=false but PruneReason=%q", i, d.PruneReason)
		}
		if d.PreCalibrationCostUs <= 0 || d.PostCalibrationCostUs <= 0 || d.CalibrationMultiplier <= 0 {
			t.Fatalf("derivation[%d] invalid costs: pre=%.2f post=%.2f mult=%.3f",
				i, d.PreCalibrationCostUs, d.PostCalibrationCostUs, d.CalibrationMultiplier)
		}
		if d.EquivalenceGroupSize < 2 || d.BeamRank <= 0 || d.BeamWidth <= 0 || len(d.ActiveRewrites) == 0 {
			t.Fatalf("derivation[%d] invalid beam/eclass metadata: %+v", i, d)
		}
	}

	if _, err := BuildEGraphDerivationTrace("sm_121a", nil); err == nil {
		t.Fatalf("expected error on empty iterations slice")
	}
}

func TestEGraphTraceHTTPHandler(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	// 1. Default GET /v1/egraph/trace
	req := httptest.NewRequest(http.MethodGet, "/v1/egraph/trace", nil)
	rec := httptest.NewRecorder()
	srv.handleEGraphTrace(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/egraph/trace status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var rep EGraphSaturationReport
	if err := json.NewDecoder(rec.Body).Decode(&rep); err != nil {
		t.Fatalf("decode EGraphSaturationReport JSON: %v", err)
	}
	if len(rep.Derivations) != 24 {
		t.Fatalf("len(Derivations) = %d, want 24", len(rep.Derivations))
	}
	if rep.CalibratorR2After <= rep.CalibratorR2Before {
		t.Fatalf("CalibratorR2After (%.3f) <= CalibratorR2Before (%.3f)", rep.CalibratorR2After, rep.CalibratorR2Before)
	}
	if rep.SelectedChampionIdx != 23 {
		t.Fatalf("default SelectedChampionIdx = %d, want 23", rep.SelectedChampionIdx)
	}

	// 2. GET /v1/egraph/trace?iter=13 highlights candidate #13
	reqIter := httptest.NewRequest(http.MethodGet, "/v1/egraph/trace?iter=13", nil)
	recIter := httptest.NewRecorder()
	srv.handleEGraphTrace(recIter, reqIter)
	if recIter.Code != http.StatusOK {
		t.Fatalf("GET /v1/egraph/trace?iter=13 status = %d, want 200", recIter.Code)
	}
	var repIter EGraphSaturationReport
	if err := json.NewDecoder(recIter.Body).Decode(&repIter); err != nil {
		t.Fatalf("decode iter=13 JSON: %v", err)
	}
	if repIter.SelectedChampionIdx != 13 {
		t.Fatalf("SelectedChampionIdx with ?iter=13 = %d, want 13", repIter.SelectedChampionIdx)
	}

	// 3. POST /v1/egraph/trace returns 405 Method Not Allowed
	reqPost := httptest.NewRequest(http.MethodPost, "/v1/egraph/trace", nil)
	recPost := httptest.NewRecorder()
	srv.handleEGraphTrace(recPost, reqPost)
	if recPost.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /v1/egraph/trace status = %d, want 405", recPost.Code)
	}
}
