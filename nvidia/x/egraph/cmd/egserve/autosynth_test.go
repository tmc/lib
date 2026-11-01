package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAutoSynthRunAutonomousBeamSynthesis(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	baseIters, err := srv.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		t.Fatalf("BuildKernelStreamIterations before synth: %v", err)
	}
	if len(baseIters) != 24 {
		t.Fatalf("len(baseIters) = %d, want 24", len(baseIters))
	}

	resp, err := RunAutonomousBeamSynthesis(context.Background(), srv, AutoSynthRequest{
		SeedIteration: 0,
		MaxRounds:     4,
		BeamWidth:     2,
	})
	if err != nil {
		t.Fatalf("RunAutonomousBeamSynthesis: %v", err)
	}

	if resp.Status != "completed" {
		t.Fatalf("Status = %q, want completed", resp.Status)
	}
	if resp.SeedIteration != 0 {
		t.Fatalf("SeedIteration = %d, want 0", resp.SeedIteration)
	}
	if resp.RoundsExecuted != 4 {
		t.Fatalf("RoundsExecuted = %d, want 4", resp.RoundsExecuted)
	}
	if resp.CandidatesExplored != len(resp.Steps) || len(resp.Steps) < 5 {
		t.Fatalf("CandidatesExplored = %d (len(Steps)=%d), want >= 5", resp.CandidatesExplored, len(resp.Steps))
	}
	if resp.PrunedCount < 1 {
		t.Fatalf("PrunedCount = %d, want >= 1 pruned exploratory mutant", resp.PrunedCount)
	}
	if resp.PromotedCount < 3 {
		t.Fatalf("PromotedCount = %d, want >= 3 promoted beam candidates", resp.PromotedCount)
	}
	if resp.FinalBestSpeedup < 5.8 {
		t.Fatalf("FinalBestSpeedup = %.2f, want >= 5.8", resp.FinalBestSpeedup)
	}
	if resp.FinalChampionIter < 24 {
		t.Fatalf("FinalChampionIter = %d, want >= 24", resp.FinalChampionIter)
	}

	for i, step := range resp.Steps {
		if step.Round < 1 || step.Round > 4 {
			t.Fatalf("step %d invalid Round = %d", i, step.Round)
		}
		if step.UCB1Score <= 0 {
			t.Fatalf("step %d invalid UCB1Score = %.3f", i, step.UCB1Score)
		}
		if step.Candidate.Iteration != 24+i {
			t.Fatalf("step %d Candidate.Iteration = %d, want %d", i, step.Candidate.Iteration, 24+i)
		}
		if len(step.Candidate.CUDALines) != 30 {
			t.Fatalf("step %d Candidate.CUDALines = %d, want 30", i, len(step.Candidate.CUDALines))
		}
		if len(step.Candidate.LineAnnotations) == 0 || len(step.Candidate.Stages) != 2 {
			t.Fatalf("step %d Candidate missing lineinfo annotations or stages", i)
		}
		if !step.Promoted && step.RejectReason == "" {
			t.Fatalf("step %d rejected candidate missing RejectReason", i)
		}
	}

	afterIters, err := srv.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		t.Fatalf("BuildKernelStreamIterations after synth: %v", err)
	}
	wantLen := 24 + len(resp.Steps)
	if len(afterIters) != wantLen {
		t.Fatalf("len(afterIters) = %d, want %d", len(afterIters), wantLen)
	}
}

func TestAutoSynthHTTPHandler(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	// 1. Verify GET is rejected with 405 Method Not Allowed.
	getReq := httptest.NewRequest(http.MethodGet, "/v1/kernels/synthesize", nil)
	getRec := httptest.NewRecorder()
	srv.handleKernelsSynthesize(getRec, getReq)
	if getRec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET status = %d, want %d", getRec.Code, http.StatusMethodNotAllowed)
	}

	// 2. Verify POST with empty body uses defaults and succeeds.
	emptyReq := httptest.NewRequest(http.MethodPost, "/v1/kernels/synthesize", nil)
	emptyRec := httptest.NewRecorder()
	srv.handleKernelsSynthesize(emptyRec, emptyReq)
	if emptyRec.Code != http.StatusOK {
		t.Fatalf("POST empty body status = %d, want 200; body=%s", emptyRec.Code, emptyRec.Body.String())
	}
	var resp AutoSynthResponse
	if err := json.NewDecoder(emptyRec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode AutoSynthResponse: %v", err)
	}
	if resp.RoundsExecuted != 4 || resp.PrunedCount < 1 || resp.PromotedCount < 2 || resp.FinalBestSpeedup < 5.8 {
		t.Fatalf("unexpected default AutoSynthResponse: %+v", resp)
	}

	// 3. Verify POST with explicit JSON payload.
	bodyBytes, err := json.Marshal(AutoSynthRequest{
		SeedIteration: 5,
		MaxRounds:     4,
		BeamWidth:     2,
	})
	if err != nil {
		t.Fatalf("marshal AutoSynthRequest: %v", err)
	}
	postReq := httptest.NewRequest(http.MethodPost, "/v1/kernels/synthesize", bytes.NewReader(bodyBytes))
	postRec := httptest.NewRecorder()
	srv.handleKernelsSynthesize(postRec, postReq)
	if postRec.Code != http.StatusOK {
		t.Fatalf("POST JSON body status = %d, want 200; body=%s", postRec.Code, postRec.Body.String())
	}
	var resp2 AutoSynthResponse
	if err := json.NewDecoder(postRec.Body).Decode(&resp2); err != nil {
		t.Fatalf("decode second AutoSynthResponse: %v", err)
	}
	if resp2.SeedIteration != 5 || resp2.FinalBestSpeedup < 5.8 {
		t.Fatalf("unexpected explicit AutoSynthResponse: %+v", resp2)
	}

	// 4. Verify invalid JSON returns 400 Bad Request.
	badReq := httptest.NewRequest(http.MethodPost, "/v1/kernels/synthesize", strings.NewReader("{invalid"))
	badRec := httptest.NewRecorder()
	srv.handleKernelsSynthesize(badRec, badReq)
	if badRec.Code != http.StatusBadRequest {
		t.Fatalf("POST invalid JSON status = %d, want %d", badRec.Code, http.StatusBadRequest)
	}
}
