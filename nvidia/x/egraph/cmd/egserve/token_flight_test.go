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

func TestTokenFlightTrajectorySimulation(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:            "sm_121a",
		HiddenDim:       256,
		IntermDim:       512,
		Layers:          4,
		PageSize:        4,
		TotalKVPages:    128,
		HBMPageCapacity: 16,
		C2CPageCapacity: 32,
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

	prompt := "Trace live CUgraphExec hot-swaps across speculative buckets M=1..8."
	rep, err := SimulateTokenFlightAcrossTrajectory(context.Background(), srv, prompt, iters)
	if err != nil {
		t.Fatalf("SimulateTokenFlightAcrossTrajectory: %v", err)
	}

	if rep.Arch != "sm_121a" {
		t.Errorf("rep.Arch = %q, want sm_121a", rep.Arch)
	}
	if rep.Prompt != prompt {
		t.Errorf("rep.Prompt = %q, want %q", rep.Prompt, prompt)
	}
	if rep.TotalSteps != 24 || len(rep.Steps) != 24 {
		t.Fatalf("TotalSteps=%d len(Steps)=%d, want 24", rep.TotalSteps, len(rep.Steps))
	}
	if rep.TotalHotSwaps != 14 {
		t.Fatalf("TotalHotSwaps = %d, want 14", rep.TotalHotSwaps)
	}
	if rep.FinalTokensPerSec <= 4.0*rep.InitialTokensPerSec {
		t.Fatalf("FinalTokensPerSec (%.1f) <= 4 * InitialTokensPerSec (%.1f)",
			rep.FinalTokensPerSec, rep.InitialTokensPerSec)
	}
	if rep.Steps[0].StepTPOTMs != 31.4 {
		t.Errorf("step #00 StepTPOTMs = %.2f, want 31.40", rep.Steps[0].StepTPOTMs)
	}
	if rep.Steps[23].StepTPOTMs != 5.19 {
		t.Errorf("step #23 StepTPOTMs = %.2f, want 5.19", rep.Steps[23].StepTPOTMs)
	}
	if rep.Steps[0].InstantTokensPerSec != 31.8 {
		t.Errorf("step #00 InstantTokensPerSec = %.1f, want 31.8", rep.Steps[0].InstantTokensPerSec)
	}
	if rep.Steps[23].InstantTokensPerSec != 192.6 {
		t.Errorf("step #23 InstantTokensPerSec = %.1f, want 192.6", rep.Steps[23].InstantTokensPerSec)
	}

	var observedBucketProgression []int
	var countedHotSwaps int
	prevSpeedup := 0.0
	prevEpoch := 1

	for i, st := range rep.Steps {
		if st.StepIndex != i {
			t.Errorf("step %d: StepIndex = %d, want %d", i, st.StepIndex, i)
		}
		if st.CandidateIteration != i {
			t.Errorf("step %d: CandidateIteration = %d, want %d", i, st.CandidateIteration, i)
		}
		if st.ChampionSpeedup < prevSpeedup {
			t.Fatalf("step %d: non-monotonic ChampionSpeedup %.2f < previous %.2f",
				i, st.ChampionSpeedup, prevSpeedup)
		}
		prevSpeedup = st.ChampionSpeedup

		if st.GraphExecEpoch < prevEpoch {
			t.Fatalf("step %d: non-monotonic GraphExecEpoch %d < previous %d",
				i, st.GraphExecEpoch, prevEpoch)
		}
		prevEpoch = st.GraphExecEpoch

		if st.HotSwapped {
			countedHotSwaps++
		}

		wantBucket := 1
		switch {
		case i >= 18:
			wantBucket = 8
		case i >= 9:
			wantBucket = 4
		case i >= 4:
			wantBucket = 2
		}
		if st.ActiveBucketM != wantBucket {
			t.Fatalf("step %d: ActiveBucketM = %d, want %d", i, st.ActiveBucketM, wantBucket)
		}
		if len(observedBucketProgression) == 0 || observedBucketProgression[len(observedBucketProgression)-1] != st.ActiveBucketM {
			observedBucketProgression = append(observedBucketProgression, st.ActiveBucketM)
		}

		if i > 0 && st.RadixPrefixHitTokens <= 0 {
			t.Errorf("step %d: RadixPrefixHitTokens = %d, want > 0 on shared prefix", i, st.RadixPrefixHitTokens)
		}
		if strings.TrimSpace(st.EmittedTokenText) == "" {
			t.Errorf("step %d: empty EmittedTokenText", i)
		}
		if !strings.HasSuffix(st.CumulativeText, st.EmittedTokenText) {
			t.Errorf("step %d: CumulativeText does not end with EmittedTokenText %q", i, st.EmittedTokenText)
		}
	}

	if countedHotSwaps != 14 {
		t.Fatalf("counted hot swaps in steps = %d, want 14", countedHotSwaps)
	}
	wantProgression := []int{1, 2, 4, 8}
	if !slices.Equal(observedBucketProgression, wantProgression) {
		t.Fatalf("ActiveBucketM progression = %v, want %v", observedBucketProgression, wantProgression)
	}
	if rep.Steps[23].GraphExecEpoch != 15 {
		t.Errorf("final GraphExecEpoch = %d, want 15", rep.Steps[23].GraphExecEpoch)
	}
}

func TestTokenFlightHTTPEndpoint(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	// 1. GET /v1/flight/tokens?prompt=...
	getReq := httptest.NewRequest(http.MethodGet, "/v1/flight/tokens?prompt=Custom+GET+flight+prompt", nil)
	getRec := httptest.NewRecorder()
	srv.handleTokenFlight(getRec, getReq)
	if getRec.Code != http.StatusOK {
		t.Fatalf("GET /v1/flight/tokens status = %d, want 200 (body=%s)", getRec.Code, getRec.Body.String())
	}
	var getRep TokenFlightReport
	if err := json.NewDecoder(getRec.Body).Decode(&getRep); err != nil {
		t.Fatalf("decode GET TokenFlightReport: %v", err)
	}
	if getRep.Prompt != "Custom GET flight prompt" {
		t.Errorf("GET report Prompt = %q, want %q", getRep.Prompt, "Custom GET flight prompt")
	}
	if getRep.TotalSteps != 24 || len(getRep.Steps) != 24 {
		t.Fatalf("GET TotalSteps=%d len(Steps)=%d, want 24", getRep.TotalSteps, len(getRep.Steps))
	}
	if getRep.TotalHotSwaps != 14 {
		t.Fatalf("GET TotalHotSwaps = %d, want 14", getRep.TotalHotSwaps)
	}

	// 2. POST /v1/flight/tokens with JSON body
	postBody, _ := json.Marshal(map[string]string{
		"prompt": "Custom POST JSON flight prompt",
	})
	postReq := httptest.NewRequest(http.MethodPost, "/v1/flight/tokens", bytes.NewReader(postBody))
	postRec := httptest.NewRecorder()
	srv.handleTokenFlight(postRec, postReq)
	if postRec.Code != http.StatusOK {
		t.Fatalf("POST /v1/flight/tokens status = %d, want 200", postRec.Code)
	}
	var postRep TokenFlightReport
	if err := json.NewDecoder(postRec.Body).Decode(&postRep); err != nil {
		t.Fatalf("decode POST TokenFlightReport: %v", err)
	}
	if postRep.Prompt != "Custom POST JSON flight prompt" {
		t.Errorf("POST report Prompt = %q, want %q", postRep.Prompt, "Custom POST JSON flight prompt")
	}
	if postRep.FinalTokensPerSec <= 4.0*postRep.InitialTokensPerSec {
		t.Errorf("POST FinalTokensPerSec (%.1f) <= 4 * InitialTokensPerSec (%.1f)",
			postRep.FinalTokensPerSec, postRep.InitialTokensPerSec)
	}

	// 3. DELETE /v1/flight/tokens -> 405 Method Not Allowed
	delReq := httptest.NewRequest(http.MethodDelete, "/v1/flight/tokens", nil)
	delRec := httptest.NewRecorder()
	srv.handleTokenFlight(delRec, delReq)
	if delRec.Code != http.StatusMethodNotAllowed {
		t.Errorf("DELETE /v1/flight/tokens status = %d, want 405", delRec.Code)
	}
}
