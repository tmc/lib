package specdecode

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestSpeculativeDecodeController(t *testing.T) {
	if _, err := cudahostemu.Compiler(); errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}

	ctrl := SpeculativeDecodeController{
		Name:        "specdecode_cond_ctrl",
		DraftTokens: 4,
		MaxSteps:    8,
	}
	src, err := ctrl.Source()
	if err != nil {
		t.Fatalf("Source(): %v", err)
	}
	for _, want := range []string{
		"cudaGraphConditionalHandle cond_handle",
		"cudaGraphSetConditional(cond_handle, keep_going)",
		"specdecode_cond_ctrl_launch",
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("Source() missing %q", want)
		}
	}

	tests := []struct {
		name         string
		ctrl         SpeculativeDecodeController
		in           SpeculativeDecodeInput
		wantTokens   []int32
		wantAccepted []int32
		wantSteps    int
		wantTrace    []uint32
	}{
		{
			name: "mixed_100pct_partial_0pct_and_eos_termination",
			ctrl: SpeculativeDecodeController{
				Name:        "specdecode_mixed_eos",
				DraftTokens: 4,
				MaxSteps:    8,
			},
			in: SpeculativeDecodeInput{
				InitialTokens: []int32{101, 102},
				EOSTokenID:    999,
				MaxSeqLen:     32,
				Steps: []SpeculativeDecodeStep{
					{
						// Step 0: 100% acceptance (a=4) + bonus token 205 -> 5 tokens appended.
						Draft:        []int32{201, 202, 203, 204},
						TargetGreedy: []int32{201, 202, 203, 204, 205},
					},
					{
						// Step 1: partial acceptance (a=2) + bonus token 399 -> 3 tokens appended.
						Draft:        []int32{301, 302, 303, 304},
						TargetGreedy: []int32{301, 302, 399, 400, 401},
					},
					{
						// Step 2: 0% acceptance (a=0) + bonus token 500 -> 1 token appended.
						Draft:        []int32{410, 411, 412, 413},
						TargetGreedy: []int32{500, 501, 502, 503, 504},
					},
					{
						// Step 3: partial acceptance (a=1) where bonus token is EOS (999) -> stops loop early!
						Draft:        []int32{601, 602, 603, 604},
						TargetGreedy: []int32{601, 999, 603, 604, 605},
					},
					{
						// Step 4: never reached because step 3 encountered EOS.
						Draft:        []int32{701, 702, 703, 704},
						TargetGreedy: []int32{701, 702, 703, 704, 705},
					},
				},
			},
			wantTokens:   []int32{101, 102, 201, 202, 203, 204, 205, 301, 302, 399, 500, 601, 999},
			wantAccepted: []int32{4, 2, 0, 1},
			wantSteps:    4,
			wantTrace:    []uint32{1, 1, 1, 0},
		},
		{
			name: "max_seq_len_termination_mid_step",
			ctrl: SpeculativeDecodeController{
				Name:        "specdecode_max_seq",
				DraftTokens: 4,
				MaxSteps:    8,
			},
			in: SpeculativeDecodeInput{
				InitialTokens: []int32{10, 11, 12},
				EOSTokenID:    999,
				MaxSeqLen:     9, // room for 6 more tokens total
				Steps: []SpeculativeDecodeStep{
					{
						// Step 0: a=3 -> appends 4 tokens (20, 21, 22, 23), seq becomes 7.
						Draft:        []int32{20, 21, 22, 90},
						TargetGreedy: []int32{20, 21, 22, 23, 24},
					},
					{
						// Step 1: all 4 draft tokens match, but MaxSeqLen=9 allows only 2 more tokens (30, 31).
						Draft:        []int32{30, 31, 32, 33},
						TargetGreedy: []int32{30, 31, 32, 33, 34},
					},
					{
						// Step 2: never reached because MaxSeqLen was reached in step 1.
						Draft:        []int32{40, 41, 42, 43},
						TargetGreedy: []int32{40, 41, 42, 43, 44},
					},
				},
			},
			wantTokens:   []int32{10, 11, 12, 20, 21, 22, 23, 30, 31},
			wantAccepted: []int32{3, 2},
			wantSteps:    2,
			wantTrace:    []uint32{1, 0},
		},
		{
			name: "eos_on_accepted_draft_token",
			ctrl: SpeculativeDecodeController{
				Name:        "specdecode_draft_eos",
				DraftTokens: 3,
				MaxSteps:    4,
			},
			in: SpeculativeDecodeInput{
				InitialTokens: []int32{1},
				EOSTokenID:    77,
				MaxSeqLen:     16,
				Steps: []SpeculativeDecodeStep{
					{
						// Draft[1] == TargetGreedy[1] == 77 (EOS): stops immediately after appending 77.
						Draft:        []int32{50, 77, 60},
						TargetGreedy: []int32{50, 77, 60, 70},
					},
					{
						Draft:        []int32{80, 81, 82},
						TargetGreedy: []int32{80, 81, 82, 83},
					},
				},
			},
			wantTokens:   []int32{1, 50, 77},
			wantAccepted: []int32{2},
			wantSteps:    1,
			wantTrace:    []uint32{0},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ref, err := tc.ctrl.Reference(tc.in)
			if err != nil {
				t.Fatalf("Reference: %v", err)
			}
			got, err := tc.ctrl.RunEmulatedLoop(t.Context(), tc.in)
			if err != nil {
				t.Fatalf("RunEmulatedLoop: %v", err)
			}

			if !slices.Equal(got.OutputTokens, tc.wantTokens) {
				t.Fatalf("OutputTokens = %v, want %v", got.OutputTokens, tc.wantTokens)
			}
			if !slices.Equal(got.OutputTokens, ref.OutputTokens) {
				t.Fatalf("OutputTokens %v != Reference %v", got.OutputTokens, ref.OutputTokens)
			}
			if !slices.Equal(got.AcceptedPerStep, tc.wantAccepted) {
				t.Fatalf("AcceptedPerStep = %v, want %v", got.AcceptedPerStep, tc.wantAccepted)
			}
			if !slices.Equal(got.AcceptedPerStep, ref.AcceptedPerStep) {
				t.Fatalf("AcceptedPerStep %v != Reference %v", got.AcceptedPerStep, ref.AcceptedPerStep)
			}
			if got.StepsExecuted != tc.wantSteps || got.StepsExecuted != ref.StepsExecuted {
				t.Fatalf("StepsExecuted = %d (ref %d), want %d", got.StepsExecuted, ref.StepsExecuted, tc.wantSteps)
			}
			if got.FinalCondition != 0 {
				t.Fatalf("FinalCondition = %d, want 0", got.FinalCondition)
			}
			if !slices.Equal(got.IterationTrace, tc.wantTrace) || !slices.Equal(got.IterationTrace, ref.IterationTrace) {
				t.Fatalf("IterationTrace = %v (ref %v), want %v", got.IterationTrace, ref.IterationTrace, tc.wantTrace)
			}
		})
	}

	t.Run("validation_errors", func(t *testing.T) {
		badCtrls := []SpeculativeDecodeController{
			{Name: "1bad", DraftTokens: 4},
			{Name: "bad_k_zero", DraftTokens: 0},
			{Name: "bad_k_large", DraftTokens: 65},
			{Name: "bad_steps", DraftTokens: 4, MaxSteps: -1},
		}
		for _, bc := range badCtrls {
			if _, err := bc.Source(); err == nil {
				t.Errorf("expected error for %+v, got nil", bc)
			}
		}

		valid := SpeculativeDecodeController{DraftTokens: 2}
		if _, err := valid.Reference(SpeculativeDecodeInput{MaxSeqLen: 8}); err == nil {
			t.Errorf("expected error for empty Steps, got nil")
		}
		if _, err := valid.Reference(SpeculativeDecodeInput{
			MaxSeqLen: 8,
			Steps:     []SpeculativeDecodeStep{{Draft: []int32{1}, TargetGreedy: []int32{1, 2, 3}}},
		}); err == nil {
			t.Errorf("expected error for wrong Draft length, got nil")
		}
	})
}
