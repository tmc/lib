package main

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCostCalibrateAll24Candidates(t *testing.T) {
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

	rep, err := CalibrateEGraphCostModel("sm_121a", iters)
	if err != nil {
		t.Fatalf("CalibrateEGraphCostModel: %v", err)
	}

	t.Logf("PriorMAPE=%.2f%% PosteriorMAPE=%.2f%% PriorRho=%.3f PosteriorRho=%.3f InversionsResolved=%d Champion=#%02d",
		rep.PriorMAPEPct, rep.PosteriorMAPEPct, rep.PriorSpearmanRho, rep.PosteriorSpearmanRho,
		rep.InversionsResolved, rep.ExtractedChampionIter)
	t.Logf("PriorWeights=%+v PosteriorWeights=%+v", rep.PriorWeights, rep.PosteriorWeights)

	if rep.Arch != "sm_121a" {
		t.Fatalf("Arch = %q, want sm_121a", rep.Arch)
	}
	if rep.SampleCount != 24 {
		t.Fatalf("SampleCount = %d, want 24", rep.SampleCount)
	}
	if len(rep.Residuals) != 24 {
		t.Fatalf("len(Residuals) = %d, want 24", len(rep.Residuals))
	}

	// Verify PriorWeights are unit analytical weights (1.00).
	if rep.PriorWeights.HBMBandwidthWeight != 1.00 ||
		rep.PriorWeights.L2ResidencyWeight != 1.00 ||
		rep.PriorWeights.SharedBankPenalty != 1.00 ||
		rep.PriorWeights.RegSpillPenalty != 1.00 ||
		rep.PriorWeights.ScoreboardWaitWeight != 1.00 ||
		rep.PriorWeights.BarrierSyncWeight != 1.00 {
		t.Fatalf("unexpected PriorWeights: %+v", rep.PriorWeights)
	}

	// Verify PosteriorWeights match hardware-calibrated expectations (~1.24, 0.72, 2.35, 3.10, 1.18, 0.88).
	if math.Abs(rep.PosteriorWeights.HBMBandwidthWeight-1.24) > 0.05 ||
		math.Abs(rep.PosteriorWeights.L2ResidencyWeight-0.72) > 0.05 ||
		math.Abs(rep.PosteriorWeights.SharedBankPenalty-2.35) > 0.05 ||
		math.Abs(rep.PosteriorWeights.RegSpillPenalty-3.10) > 0.05 ||
		math.Abs(rep.PosteriorWeights.ScoreboardWaitWeight-1.18) > 0.05 ||
		math.Abs(rep.PosteriorWeights.BarrierSyncWeight-0.88) > 0.05 {
		t.Fatalf("unexpected PosteriorWeights: %+v", rep.PosteriorWeights)
	}

	// Verify PriorMAPEPct ~ 18.4% and PosteriorMAPEPct <= 2.5%.
	if rep.PriorMAPEPct < 15.0 || rep.PriorMAPEPct > 22.0 {
		t.Fatalf("PriorMAPEPct = %.2f%%, want ~18.4%% (in [15%%, 22%%])", rep.PriorMAPEPct)
	}
	if rep.PosteriorMAPEPct > 2.5 {
		t.Fatalf("PosteriorMAPEPct = %.2f%%, want <= 2.5%%", rep.PosteriorMAPEPct)
	}

	// Verify PriorSpearmanRho ~ 0.86 and PosteriorSpearmanRho >= 0.99.
	if rep.PriorSpearmanRho < 0.78 || rep.PriorSpearmanRho > 0.92 {
		t.Fatalf("PriorSpearmanRho = %.3f, want ~0.86 (in [0.78, 0.92])", rep.PriorSpearmanRho)
	}
	if rep.PosteriorSpearmanRho < 0.99 {
		t.Fatalf("PosteriorSpearmanRho = %.3f, want >= 0.99", rep.PosteriorSpearmanRho)
	}

	// Verify InversionsResolved >= 3 and specific ranking inversions (#03, #06, #10 vs #13).
	if rep.InversionsResolved < 3 {
		t.Fatalf("InversionsResolved = %d, want >= 3", rep.InversionsResolved)
	}
	if !rep.Residuals[3].RankingInverted {
		t.Fatalf("expected #03 (unroll_16x_reg_spill) to have RankingInverted=true")
	}
	if !rep.Residuals[6].RankingInverted {
		t.Fatalf("expected #06 (vec4_strided_uncoalesced) to have RankingInverted=true")
	}
	if !rep.Residuals[10].RankingInverted {
		t.Fatalf("expected #10 (smem_tile_linear_bank_conflict) to have RankingInverted=true")
	}
	// Check #10 vs #13 inversion specifically: prior predicts #10 < #13, posterior predicts #10 > #13.
	r10 := rep.Residuals[10]
	r13 := rep.Residuals[13]
	if !(r10.PriorPredictedUs < r13.PriorPredictedUs && r10.PosteriorPredictedUs > r13.PosteriorPredictedUs) {
		t.Fatalf("expected #10 vs #13 ranking inversion resolved: #10 prior=%.2f post=%.2f vs #13 prior=%.2f post=%.2f",
			r10.PriorPredictedUs, r10.PosteriorPredictedUs, r13.PriorPredictedUs, r13.PosteriorPredictedUs)
	}

	if rep.ExtractedChampionIter != 23 {
		t.Fatalf("ExtractedChampionIter = %d, want 23", rep.ExtractedChampionIter)
	}

	if _, err := CalibrateEGraphCostModel("sm_121a", nil); err == nil {
		t.Fatalf("expected error on empty iterations slice")
	}
}

func TestCostCalibrateHTTPHandler(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	// 1. GET /v1/egraph/calibrate
	reqGet := httptest.NewRequest(http.MethodGet, "/v1/egraph/calibrate", nil)
	recGet := httptest.NewRecorder()
	srv.handleEGraphCalibrate(recGet, reqGet)
	if recGet.Code != http.StatusOK {
		t.Fatalf("GET /v1/egraph/calibrate status = %d, want 200; body=%s", recGet.Code, recGet.Body.String())
	}
	var repGet CostCalibrationReport
	if err := json.NewDecoder(recGet.Body).Decode(&repGet); err != nil {
		t.Fatalf("decode GET CostCalibrationReport: %v", err)
	}
	if repGet.SampleCount != 24 || repGet.ExtractedChampionIter != 23 || repGet.PosteriorMAPEPct > 2.5 || repGet.PosteriorSpearmanRho < 0.99 {
		t.Fatalf("unexpected GET report: %+v", repGet)
	}

	// 2. POST /v1/egraph/calibrate with JSON body
	postPayload := bytes.NewBufferString(`{"arch":"sm_100a","mode":"simple","bucket_m":1}`)
	reqPost := httptest.NewRequest(http.MethodPost, "/v1/egraph/calibrate", postPayload)
	recPost := httptest.NewRecorder()
	srv.handleEGraphCalibrate(recPost, reqPost)
	if recPost.Code != http.StatusOK {
		t.Fatalf("POST /v1/egraph/calibrate status = %d, want 200; body=%s", recPost.Code, recPost.Body.String())
	}
	var repPost CostCalibrationReport
	if err := json.NewDecoder(recPost.Body).Decode(&repPost); err != nil {
		t.Fatalf("decode POST CostCalibrationReport: %v", err)
	}
	if repPost.Arch != "sm_100a" || repPost.SampleCount != 24 || repPost.InversionsResolved < 3 {
		t.Fatalf("unexpected POST report: %+v", repPost)
	}

	// 3. GET /v1/egraph/calibrate?mode=cross_layer&bucket_m=2
	reqCross := httptest.NewRequest(http.MethodGet, "/v1/egraph/calibrate?mode=cross_layer&bucket_m=2", nil)
	recCross := httptest.NewRecorder()
	srv.handleEGraphCalibrate(recCross, reqCross)
	if recCross.Code != http.StatusOK {
		t.Fatalf("GET /v1/egraph/calibrate?mode=cross_layer status = %d, want 200; body=%s", recCross.Code, recCross.Body.String())
	}
	var repCross CostCalibrationReport
	if err := json.NewDecoder(recCross.Body).Decode(&repCross); err != nil {
		t.Fatalf("decode cross_layer CostCalibrationReport: %v", err)
	}
	if repCross.SampleCount != 4 || repCross.PosteriorMAPEPct >= repCross.PriorMAPEPct {
		t.Fatalf("unexpected cross_layer report: %+v", repCross)
	}

	// 4. DELETE /v1/egraph/calibrate returns 405 Method Not Allowed
	reqDel := httptest.NewRequest(http.MethodDelete, "/v1/egraph/calibrate", nil)
	recDel := httptest.NewRecorder()
	srv.handleEGraphCalibrate(recDel, reqDel)
	if recDel.Code != http.StatusMethodNotAllowed {
		t.Fatalf("DELETE /v1/egraph/calibrate status = %d, want 405", recDel.Code)
	}
}
