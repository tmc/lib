package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
)

func TestSandboxRewriteRulesAndApplyKnobs(t *testing.T) {
	rules := AvailableSandboxRewriteRules()
	if len(rules) != 8 {
		t.Fatalf("len(AvailableSandboxRewriteRules()) = %d, want 8", len(rules))
	}
	wantIDs := []string{
		"vec4_ldg128",
		"warp_shfl_bfly",
		"swizzle_128b",
		"reg_tile_2x4",
		"launch_bounds_4cta",
		"fast_math_mufu",
		"cp_async_2stage",
		"l2_evict_last",
	}
	for i, wantID := range wantIDs {
		if rules[i].ID != wantID {
			t.Fatalf("rule[%d].ID = %q, want %q", i, rules[i].ID, wantID)
		}
		if rules[i].Description == "" || rules[i].TargetInstruction == "" {
			t.Fatalf("rule %q missing description or target instruction: %+v", wantID, rules[i])
		}
	}

	// Apply all 8 knobs starting from empty baseLines (#00 default).
	mutated, applied := ApplySandboxRewriteKnobs(nil, wantIDs)
	if len(mutated) != 30 {
		t.Fatalf("len(mutated) = %d, want 30", len(mutated))
	}
	if !slices.Equal(applied, wantIDs) {
		t.Fatalf("applied = %v, want %v", applied, wantIDs)
	}

	joined := strings.Join(mutated, "\n")
	for _, wantSub := range []string{
		"__ldca(",
		"__shfl_xor_sync(0xffffffffu, sq, mask)",
		"tid ^ ((tid >> 5) << 2)",
		"k4 ^ ((k4 >> 5) << 2)",
		"k4 += 2",
		"__launch_bounds__(256, 4)",
		"__frcp_rn(1.0f+__expf(",
		"__ldcs(",
	} {
		if !strings.Contains(joined, wantSub) {
			t.Fatalf("mutated 30-line kernel missing feature %q:\n%s", wantSub, joined)
		}
	}

	ver := VerifySimpleKernelCandidate(mutated)
	if !ver.Passed() {
		t.Fatalf("expected all-8-knobs candidate to pass VerifySimpleKernelCandidate, got %+v", ver)
	}
}

func TestSandboxEvaluateCandidatePromotedAndRejectedGates(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	ctx := context.Background()

	// 1. All-8-rules combination should be promoted and beat/tie the #23 champion (6.05x).
	allRules := []string{
		"vec4_ldg128",
		"warp_shfl_bfly",
		"swizzle_128b",
		"reg_tile_2x4",
		"launch_bounds_4cta",
		"fast_math_mufu",
		"cp_async_2stage",
		"l2_evict_last",
	}
	resPromoted, err := EvaluateSandboxCandidate(ctx, srv, SandboxEvalRequest{
		BaseIteration: 0,
		AppliedRules:  allRules,
		CandidateName: "All-8 Rewrite Synthesis",
	})
	if err != nil {
		t.Fatalf("EvaluateSandboxCandidate(promoted): %v", err)
	}
	if !resPromoted.Verification.Passed() {
		t.Fatalf("expected verification to pass, got %+v", resPromoted.Verification)
	}
	if !resPromoted.Candidate.Promoted || !resPromoted.PromotedNew || resPromoted.Status != "promoted" {
		t.Fatalf("expected candidate to be promoted: status=%q promoted=%v promotedNew=%v reject=%q",
			resPromoted.Status, resPromoted.Candidate.Promoted, resPromoted.PromotedNew, resPromoted.Candidate.RejectReason)
	}
	if resPromoted.Candidate.HoldoutSpeedup < 6.05 || resPromoted.Candidate.TotalSpillBytes != 0 {
		t.Fatalf("unexpected promoted metrics: speedup=%.2f spills=%d",
			resPromoted.Candidate.HoldoutSpeedup, resPromoted.Candidate.TotalSpillBytes)
	}
	if resPromoted.Candidate.Iteration != 24 || resPromoted.TotalFrames != 25 {
		t.Fatalf("iteration=%d totalFrames=%d, want 24 and 25",
			resPromoted.Candidate.Iteration, resPromoted.TotalFrames)
	}

	// 2. Broken candidate with missing cross-stage __syncthreads() in Stage s1 -> rejected by formal RAW barrier verifier.
	brokenLines := slices.Clone(resPromoted.Candidate.CUDALines)
	brokenLines[16] = "    /* __syncthreads() removed by user */"
	resBroken, err := EvaluateSandboxCandidate(ctx, srv, SandboxEvalRequest{
		CUDALines:     brokenLines,
		CandidateName: "Missing Barrier Edit",
	})
	if err != nil {
		t.Fatalf("EvaluateSandboxCandidate(broken barrier): %v", err)
	}
	if resBroken.Verification.RAWBarrierOK || resBroken.Verification.Passed() {
		t.Fatalf("expected RAWBarrierOK=false for missing __syncthreads(), got %+v", resBroken.Verification)
	}
	if resBroken.Candidate.Promoted || resBroken.PromotedNew || resBroken.Status != "rejected" {
		t.Fatalf("expected broken candidate to be rejected, got status=%q promoted=%v",
			resBroken.Status, resBroken.Candidate.Promoted)
	}
	if !strings.Contains(resBroken.Candidate.RejectReason, "RAW") {
		t.Fatalf("expected RAW hazard in RejectReason, got %q", resBroken.Candidate.RejectReason)
	}
	if resBroken.Candidate.Iteration != 25 || resBroken.TotalFrames != 26 {
		t.Fatalf("iteration=%d totalFrames=%d, want 25 and 26",
			resBroken.Candidate.Iteration, resBroken.TotalFrames)
	}

	// 3. Over-unrolled #pragma unroll 16 candidate -> rejected by PTXAS register-spill gate.
	spillLines := slices.Clone(resPromoted.Candidate.CUDALines)
	spillLines[18] = "    #pragma unroll 16 // force register spill"
	resSpill, err := EvaluateSandboxCandidate(ctx, srv, SandboxEvalRequest{
		CUDASource:    strings.Join(spillLines, "\n"),
		CandidateName: "Over-Unrolled 16x Spill",
	})
	if err != nil {
		t.Fatalf("EvaluateSandboxCandidate(spill): %v", err)
	}
	if resSpill.Candidate.Promoted || resSpill.Candidate.TotalSpillBytes <= 0 {
		t.Fatalf("expected PTXAS spill rejection with TotalSpillBytes > 0, got promoted=%v spills=%d",
			resSpill.Candidate.Promoted, resSpill.Candidate.TotalSpillBytes)
	}
	if !strings.Contains(resSpill.Candidate.RejectReason, "spill") {
		t.Fatalf("expected spill gate in RejectReason, got %q", resSpill.Candidate.RejectReason)
	}
	if resSpill.Candidate.Iteration != 26 || resSpill.TotalFrames != 27 {
		t.Fatalf("iteration=%d totalFrames=%d, want 26 and 27",
			resSpill.Candidate.Iteration, resSpill.TotalFrames)
	}
}

func TestSandboxHTTPHandlers(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	// 1. Test GET /v1/kernels/rewrites handler.
	reqRewrites := httptest.NewRequest(http.MethodGet, "/v1/kernels/rewrites", nil)
	recRewrites := httptest.NewRecorder()
	srv.handleKernelsRewrites(recRewrites, reqRewrites)
	if recRewrites.Code != http.StatusOK {
		t.Fatalf("GET /v1/kernels/rewrites status = %d, want 200", recRewrites.Code)
	}
	var rewritePayload struct {
		Arch  string               `json:"arch"`
		Rules []SandboxRewriteRule `json:"rules"`
	}
	if err := json.NewDecoder(recRewrites.Body).Decode(&rewritePayload); err != nil {
		t.Fatalf("decode /v1/kernels/rewrites: %v", err)
	}
	if rewritePayload.Arch != "sm_121a" || len(rewritePayload.Rules) != 8 {
		t.Fatalf("unexpected rewrites response: arch=%q len=%d", rewritePayload.Arch, len(rewritePayload.Rules))
	}

	// 2. Test POST /v1/kernels/eval handler.
	evalBody, err := json.Marshal(SandboxEvalRequest{
		BaseIteration: 0,
		AppliedRules: []string{
			"vec4_ldg128",
			"warp_shfl_bfly",
			"swizzle_128b",
			"reg_tile_2x4",
			"launch_bounds_4cta",
			"fast_math_mufu",
			"cp_async_2stage",
			"l2_evict_last",
		},
		CandidateName: "HTTP Eval All Knobs",
	})
	if err != nil {
		t.Fatalf("marshal eval request: %v", err)
	}
	reqEval := httptest.NewRequest(http.MethodPost, "/v1/kernels/eval", bytes.NewReader(evalBody))
	reqEval.Header.Set("Content-Type", "application/json")
	recEval := httptest.NewRecorder()
	srv.handleKernelsEval(recEval, reqEval)
	if recEval.Code != http.StatusOK {
		t.Fatalf("POST /v1/kernels/eval status = %d, want 200; body=%s", recEval.Code, recEval.Body.String())
	}
	var evalResp SandboxEvalResponse
	if err := json.NewDecoder(recEval.Body).Decode(&evalResp); err != nil {
		t.Fatalf("decode /v1/kernels/eval response: %v", err)
	}
	if evalResp.Status != "promoted" || !evalResp.Candidate.Promoted || !evalResp.PromotedNew {
		t.Fatalf("expected promoted evalResp, got %+v", evalResp)
	}
	if len(evalResp.AppliedRules) != 8 || evalResp.TotalFrames != 25 {
		t.Fatalf("unexpected AppliedRules=%d TotalFrames=%d", len(evalResp.AppliedRules), evalResp.TotalFrames)
	}

	// Verify subsequent BuildKernelStreamIterations("simple", 1) includes the appended frame #24.
	iters, err := srv.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		t.Fatalf("BuildKernelStreamIterations: %v", err)
	}
	if len(iters) != 25 {
		t.Fatalf("len(iters) after sandbox eval = %d, want 25", len(iters))
	}
}
