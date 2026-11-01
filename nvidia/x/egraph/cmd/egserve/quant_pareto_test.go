package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestQuantParetoReportBuild(t *testing.T) {
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
	if len(iters) == 0 {
		t.Fatal("expected non-empty simple iterations")
	}

	promoted := iters[len(iters)-1]
	rep, err := BuildQuantParetoReport("sm_121a", promoted, 0)
	if err != nil {
		t.Fatalf("BuildQuantParetoReport: %v", err)
	}

	if rep.Arch != "sm_121a" {
		t.Errorf("Arch = %q, want sm_121a", rep.Arch)
	}
	if rep.SourceIteration != promoted.Iteration {
		t.Errorf("SourceIteration = %d, want %d", rep.SourceIteration, promoted.Iteration)
	}
	if rep.MaxAllowedPPLDelta != 0.15 {
		t.Errorf("MaxAllowedPPLDelta = %v, want 0.15 default", rep.MaxAllowedPPLDelta)
	}
	if rep.RecommendedFormat != "nvfp4_sparse_2to4" {
		t.Errorf("RecommendedFormat = %q, want nvfp4_sparse_2to4", rep.RecommendedFormat)
	}

	wantFormats := []struct {
		id     string
		bits   float64
		pareto bool
	}{
		{"fp32", 32.0, true},
		{"bf16", 16.0, true},
		{"int8_wo", 8.5, false},
		{"fp8_e4m3", 8.25, true},
		{"nvfp4_e2m1", 4.5, true},
		{"nvfp4_sparse_2to4", 2.5, true},
	}
	if len(rep.Formats) != len(wantFormats) {
		t.Fatalf("len(Formats) = %d, want %d", len(rep.Formats), len(wantFormats))
	}

	for i, wf := range wantFormats {
		got := rep.Formats[i]
		if got.FormatID != wf.id {
			t.Errorf("Formats[%d].FormatID = %q, want %q", i, got.FormatID, wf.id)
		}
		if got.EffectiveBitsPerWeight != wf.bits {
			t.Errorf("Formats[%d] (%s) EffectiveBitsPerWeight = %v, want %v", i, got.FormatID, got.EffectiveBitsPerWeight, wf.bits)
		}
		if got.ParetoOptimal != wf.pareto {
			t.Errorf("Formats[%d] (%s) ParetoOptimal = %v, want %v", i, got.FormatID, got.ParetoOptimal, wf.pareto)
		}
		if got.QualificationVerdict != "PROMOTED_SLA_OK" {
			t.Errorf("Formats[%d] (%s) QualificationVerdict = %q, want PROMOTED_SLA_OK", i, got.FormatID, got.QualificationVerdict)
		}
		if strings.TrimSpace(got.TensorCoreMnemonic) == "" {
			t.Errorf("Formats[%d] (%s) empty TensorCoreMnemonic", i, got.FormatID)
		}
		if strings.TrimSpace(got.PTXLoweringSnippet) == "" {
			t.Errorf("Formats[%d] (%s) empty PTXLoweringSnippet", i, got.FormatID)
		}
		if got.WeightBytesTransferred <= 0 {
			t.Errorf("Formats[%d] (%s) WeightBytesTransferred = %d, want > 0", i, got.FormatID, got.WeightBytesTransferred)
		}
		if got.LatencyUs <= 0 {
			t.Errorf("Formats[%d] (%s) LatencyUs = %v, want > 0", i, got.FormatID, got.LatencyUs)
		}
	}

	// Verify SLA threshold sweep selects expected Pareto-optimal format.
	slaCases := []struct {
		maxPPL          float64
		wantRecommended string
		wantSparseSLA   string
	}{
		{0.15, "nvfp4_sparse_2to4", "PROMOTED_SLA_OK"},
		{0.10, "nvfp4_e2m1", "ACCURACY_FLOOR_BREACH"},
		{0.035, "fp8_e4m3", "ACCURACY_FLOOR_BREACH"},
		{0.010, "bf16", "ACCURACY_FLOOR_BREACH"},
		{0.001, "fp32", "ACCURACY_FLOOR_BREACH"},
	}
	for _, tc := range slaCases {
		r, err := BuildQuantParetoReport("sm_121a", promoted, tc.maxPPL)
		if err != nil {
			t.Fatalf("BuildQuantParetoReport(maxPPL=%v): %v", tc.maxPPL, err)
		}
		if r.RecommendedFormat != tc.wantRecommended {
			t.Errorf("maxPPL=%v: RecommendedFormat = %q, want %q", tc.maxPPL, r.RecommendedFormat, tc.wantRecommended)
		}
		sparseFmt := r.Formats[len(r.Formats)-1]
		if sparseFmt.QualificationVerdict != tc.wantSparseSLA {
			t.Errorf("maxPPL=%v: nvfp4_sparse_2to4 verdict = %q, want %q", tc.maxPPL, sparseFmt.QualificationVerdict, tc.wantSparseSLA)
		}
	}
}

func TestQuantParetoHTTPHandler(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    2,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/kernels/quant-pareto?iter=23&max_ppl_delta=0.10", nil)
	rec := httptest.NewRecorder()
	srv.handleKernelsQuantPareto(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("handleKernelsQuantPareto status = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}

	var rep QuantParetoReport
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if rep.SourceIteration != 23 {
		t.Errorf("SourceIteration = %d, want 23", rep.SourceIteration)
	}
	if rep.MaxAllowedPPLDelta != 0.10 {
		t.Errorf("MaxAllowedPPLDelta = %v, want 0.10", rep.MaxAllowedPPLDelta)
	}
	if rep.RecommendedFormat != "nvfp4_e2m1" {
		t.Errorf("RecommendedFormat = %q, want nvfp4_e2m1", rep.RecommendedFormat)
	}
	if len(rep.Formats) != 6 {
		t.Fatalf("len(Formats) = %d, want 6", len(rep.Formats))
	}

	postReq := httptest.NewRequest(http.MethodPost, "/v1/kernels/quant-pareto", nil)
	postRec := httptest.NewRecorder()
	srv.handleKernelsQuantPareto(postRec, postReq)
	if postRec.Code != http.StatusMethodNotAllowed {
		t.Errorf("POST handleKernelsQuantPareto status = %d, want %d", postRec.Code, http.StatusMethodNotAllowed)
	}
}
