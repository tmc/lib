package main

import (
	"math"
	"slices"
	"testing"
	"time"
)

func TestSummarizeStepDurations(t *testing.T) {
	ms := func(n int) time.Duration { return time.Duration(n) * time.Millisecond }
	for _, tt := range []struct {
		name              string
		steps             []time.Duration
		wantMin, wantMed  time.Duration
		wantMean, wantMax time.Duration
		wantErr           bool
	}{
		{
			name:     "drops step 0 warmup/graph-build spike (odd steady-state count)",
			steps:    []time.Duration{ms(500), ms(14), ms(10), ms(12), ms(16), ms(13)},
			wantMin:  ms(10),
			wantMed:  ms(13),
			wantMean: ms(13),
			wantMax:  ms(16),
		},
		{
			name:     "drops step 0 warmup/graph-build spike (even steady-state count)",
			steps:    []time.Duration{ms(400), ms(20), ms(10), ms(16), ms(14)},
			wantMin:  ms(10),
			wantMed:  ms(15),
			wantMean: ms(15),
			wantMax:  ms(20),
		},
		{
			name:     "single step kept when len == 1",
			steps:    []time.Duration{ms(25)},
			wantMin:  ms(25),
			wantMed:  ms(25),
			wantMean: ms(25),
			wantMax:  ms(25),
		},
		{
			name:    "empty slice rejected",
			steps:   nil,
			wantErr: true,
		},
		{
			name:    "zero duration at step 0 rejected",
			steps:   []time.Duration{0, ms(10), ms(12)},
			wantErr: true,
		},
		{
			name:    "negative duration at steady-state step rejected",
			steps:   []time.Duration{ms(50), ms(10), -ms(1)},
			wantErr: true,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			orig := slices.Clone(tt.steps)
			gotMin, gotMed, gotMean, gotMax, err := SummarizeStepDurations(tt.steps)
			if !slices.Equal(tt.steps, orig) {
				t.Errorf("SummarizeStepDurations mutated input slice: got %v, want %v", tt.steps, orig)
			}
			if tt.wantErr {
				if err == nil {
					t.Fatalf("SummarizeStepDurations(%v) succeeded, want error", tt.steps)
				}
				return
			}
			if err != nil {
				t.Fatalf("SummarizeStepDurations(%v): %v", tt.steps, err)
			}
			if gotMin != tt.wantMin || gotMed != tt.wantMed || gotMean != tt.wantMean || gotMax != tt.wantMax {
				t.Errorf("SummarizeStepDurations(%v) = (min=%v, med=%v, mean=%v, max=%v), want (%v, %v, %v, %v)",
					tt.steps, gotMin, gotMed, gotMean, gotMax, tt.wantMin, tt.wantMed, tt.wantMean, tt.wantMax)
			}
		})
	}
}

func makeTenPairs(wins int, candTPOT, baseTPOT time.Duration) []ABBAPair {
	orders := []string{"AB", "BA", "AB", "BA", "AB", "BA", "AB", "BA", "AB", "BA"}
	pairs := make([]ABBAPair, 10)
	for i := range pairs {
		c, b := candTPOT, baseTPOT
		if i >= wins {
			c, b = baseTPOT+time.Millisecond, baseTPOT
		}
		pairs[i] = ABBAPair{
			Order:         orders[i],
			CandidateTPOT: c,
			BaselineTPOT:  b,
			TokensMatch:   true,
		}
	}
	return pairs
}

func TestEvaluateABBAPairs(t *testing.T) {
	ms := func(n int) time.Duration { return time.Duration(n) * time.Millisecond }
	us := func(n int) time.Duration { return time.Duration(n) * time.Microsecond }

	for _, tt := range []struct {
		name           string
		pairs          []ABBAPair
		wantWins       int
		wantBaseWins   int
		wantTies       int
		wantMedCand    time.Duration
		wantMedBase    time.Duration
		wantImprove    float64
		wantPValue     float64
		wantNoFallback bool
		wantTokMatch   bool
		wantPass       bool
		wantErr        bool
	}{
		{
			name:           "pass: 9/10 wins with 10% median TPOT improvement",
			pairs:          makeTenPairs(9, ms(90), ms(100)),
			wantWins:       9,
			wantBaseWins:   1,
			wantTies:       0,
			wantMedCand:    ms(90),
			wantMedBase:    ms(100),
			wantImprove:    0.10,
			wantPValue:     11.0 / 1024.0, // 0.0107421875
			wantNoFallback: true,
			wantTokMatch:   true,
			wantPass:       true,
		},
		{
			name:           "pass: 10/10 wins with 20% median TPOT improvement",
			pairs:          makeTenPairs(10, ms(80), ms(100)),
			wantWins:       10,
			wantBaseWins:   0,
			wantTies:       0,
			wantMedCand:    ms(80),
			wantMedBase:    ms(100),
			wantImprove:    0.20,
			wantPValue:     1.0 / 1024.0, // 0.0009765625
			wantNoFallback: true,
			wantTokMatch:   true,
			wantPass:       true,
		},
		{
			name:           "must-fail control: 8/10 wins fails sign test",
			pairs:          makeTenPairs(8, ms(85), ms(100)),
			wantWins:       8,
			wantBaseWins:   2,
			wantTies:       0,
			wantMedCand:    ms(85),
			wantMedBase:    ms(100),
			wantImprove:    0.15,
			wantPValue:     56.0 / 1024.0, // 0.0546875
			wantNoFallback: true,
			wantTokMatch:   true,
			wantPass:       false,
		},
		{
			name:           "must-fail control: 9/10 wins with <5% median TPOT improvement (4%)",
			pairs:          makeTenPairs(9, ms(96), ms(100)),
			wantWins:       9,
			wantBaseWins:   1,
			wantTies:       0,
			wantMedCand:    ms(96),
			wantMedBase:    ms(100),
			wantImprove:    0.04,
			wantPValue:     11.0 / 1024.0,
			wantNoFallback: true,
			wantTokMatch:   true,
			wantPass:       false,
		},
		{
			name: "must-fail control: 9/10 wins with candidate fallback > 0",
			pairs: func() []ABBAPair {
				p := makeTenPairs(9, ms(90), ms(100))
				p[4].CandidateFallback = 1
				return p
			}(),
			wantWins:       9,
			wantBaseWins:   1,
			wantTies:       0,
			wantMedCand:    ms(90),
			wantMedBase:    ms(100),
			wantImprove:    0.10,
			wantPValue:     11.0 / 1024.0,
			wantNoFallback: false,
			wantTokMatch:   true,
			wantPass:       false,
		},
		{
			name: "must-fail control: 9/10 wins with baseline fallback > 0",
			pairs: func() []ABBAPair {
				p := makeTenPairs(9, ms(90), ms(100))
				p[7].BaselineFallback = 2
				return p
			}(),
			wantWins:       9,
			wantBaseWins:   1,
			wantTies:       0,
			wantMedCand:    ms(90),
			wantMedBase:    ms(100),
			wantImprove:    0.10,
			wantPValue:     11.0 / 1024.0,
			wantNoFallback: false,
			wantTokMatch:   true,
			wantPass:       false,
		},
		{
			name: "must-fail control: 9/10 wins with token mismatch",
			pairs: func() []ABBAPair {
				p := makeTenPairs(9, ms(90), ms(100))
				p[2].TokensMatch = false
				return p
			}(),
			wantWins:       9,
			wantBaseWins:   1,
			wantTies:       0,
			wantMedCand:    ms(90),
			wantMedBase:    ms(100),
			wantImprove:    0.10,
			wantPValue:     11.0 / 1024.0,
			wantNoFallback: true,
			wantTokMatch:   false,
			wantPass:       false,
		},
		{
			name:           "must-fail control: fewer than 10 pairs (9/9 wins)",
			pairs:          makeTenPairs(10, ms(85), ms(100))[:9],
			wantWins:       9,
			wantBaseWins:   0,
			wantTies:       0,
			wantMedCand:    ms(85),
			wantMedBase:    ms(100),
			wantImprove:    0.15,
			wantPValue:     1.0 / 512.0,
			wantNoFallback: true,
			wantTokMatch:   true,
			wantPass:       false,
		},
		{
			name: "must-fail control: tie is not counted as candidate win",
			pairs: func() []ABBAPair {
				p := makeTenPairs(8, us(9000), us(10000))
				p[8].CandidateTPOT = us(10000)
				p[8].BaselineTPOT = us(10000)
				return p
			}(),
			wantWins:       8,
			wantBaseWins:   1,
			wantTies:       1,
			wantMedCand:    us(9000),
			wantMedBase:    us(10000),
			wantImprove:    0.10,
			wantPValue:     56.0 / 1024.0,
			wantNoFallback: true,
			wantTokMatch:   true,
			wantPass:       false,
		},
		{
			name:    "error: empty pairs slice",
			pairs:   nil,
			wantErr: true,
		},
		{
			name: "error: invalid order string",
			pairs: []ABBAPair{
				{Order: "AA", CandidateTPOT: ms(90), BaselineTPOT: ms(100), TokensMatch: true},
			},
			wantErr: true,
		},
		{
			name: "error: non-positive candidate TPOT",
			pairs: []ABBAPair{
				{Order: "AB", CandidateTPOT: 0, BaselineTPOT: ms(100), TokensMatch: true},
			},
			wantErr: true,
		},
		{
			name: "error: non-positive baseline TPOT",
			pairs: []ABBAPair{
				{Order: "BA", CandidateTPOT: ms(90), BaselineTPOT: -ms(1), TokensMatch: true},
			},
			wantErr: true,
		},
		{
			name: "error: negative fallback count",
			pairs: []ABBAPair{
				{Order: "AB", CandidateTPOT: ms(90), BaselineTPOT: ms(100), CandidateFallback: -1, TokensMatch: true},
			},
			wantErr: true,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := EvaluateABBAPairs(tt.pairs)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("EvaluateABBAPairs succeeded (%+v), want error", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("EvaluateABBAPairs: %v", err)
			}
			if got.Pairs != len(tt.pairs) {
				t.Errorf("Pairs = %d, want %d", got.Pairs, len(tt.pairs))
			}
			if got.CandidateWins != tt.wantWins || got.BaselineWins != tt.wantBaseWins || got.Ties != tt.wantTies {
				t.Errorf("wins/losses/ties = (%d, %d, %d), want (%d, %d, %d)",
					got.CandidateWins, got.BaselineWins, got.Ties, tt.wantWins, tt.wantBaseWins, tt.wantTies)
			}
			if got.MedianCandidateTPOT != tt.wantMedCand || got.MedianBaselineTPOT != tt.wantMedBase {
				t.Errorf("medians = (%v, %v), want (%v, %v)",
					got.MedianCandidateTPOT, got.MedianBaselineTPOT, tt.wantMedCand, tt.wantMedBase)
			}
			if math.Abs(got.MedianImprovement-tt.wantImprove) > 1e-12 {
				t.Errorf("MedianImprovement = %.15f, want %.15f", got.MedianImprovement, tt.wantImprove)
			}
			if got.OneSidedPValue != tt.wantPValue {
				t.Errorf("OneSidedPValue = %.15f, want exact %.15f", got.OneSidedPValue, tt.wantPValue)
			}
			if got.ZeroFallback != tt.wantNoFallback {
				t.Errorf("ZeroFallback = %v, want %v", got.ZeroFallback, tt.wantNoFallback)
			}
			if got.AllTokensMatch != tt.wantTokMatch {
				t.Errorf("AllTokensMatch = %v, want %v", got.AllTokensMatch, tt.wantTokMatch)
			}
			if got.Pass != tt.wantPass {
				t.Errorf("Pass = %v, want %v (summary: %+v)", got.Pass, tt.wantPass, got)
			}
		})
	}
}
