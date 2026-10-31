package qualify

import (
	"crypto/sha256"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// buildTransformerFFNGraph constructs a transformer pre-norm + two-projection
// FFN + residual term graph:
//
//	norm = RMSNorm(x, rms_w, K)
//	h1   = Map(relu_bias, MatMul(Cast(BF16, norm), w1), b1)
//	h2   = Map(residual_add, MatMul(Cast(BF16, h1), w2), x)
func buildTransformerFFNGraph(m, k, n int) (*hopper.Builder, egraph.Id) {
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.F32, m, k)
	rmsW := b.Input("rms_w", hopper.F32, k)
	w1 := b.Input("w1", hopper.BF16, n, k)
	b1 := b.Input("b1", hopper.F32, n)
	w2 := b.Input("w2", hopper.BF16, k, n)

	norm := b.RMSNorm(x, rmsW, k, 1e-5)
	mm1 := b.MatMul(b.Cast(hopper.BF16, norm), w1)
	act := b.Map(hopper.MustLambda("(max (add $0 $1) 0)"), mm1, b1)
	mm2 := b.MatMul(b.Cast(hopper.BF16, act), w2)
	out := b.Map(hopper.MustLambda("(add $0 $1)"), mm2, x)
	return b, out
}

func TestStructuralBeamExtraction(t *testing.T) {
	for _, tc := range []struct {
		name string
		m    int
		k    int
		n    int
	}{
		{name: "decode_m4", m: 4, k: 256, n: 256},
		{name: "prefill_m64", m: 64, k: 256, n: 256},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, root := buildTransformerFFNGraph(tc.m, tc.k, tc.n)
			cfg := hopper.CostConfig{Device: "NVIDIA H200"}

			beam, err := ExtractStructuralPlanBeam(b.G, root, "ffn_"+tc.name, hopper.F32, cfg, 8)
			if err != nil {
				t.Fatalf("ExtractStructuralPlanBeam: %v", err)
			}
			if len(beam) < 3 {
				t.Fatalf("got %d structural beam candidates, want >= 3", len(beam))
			}

			seenSigs := make(map[string]bool)
			var (
				hasUnfusedMultiKernel bool
				hasEpilogueFused      bool
				hasMegakernelOrTail   bool
			)

			for i, cand := range beam {
				if cand.Plan == nil || len(cand.Plan.Kernels) == 0 {
					t.Fatalf("candidate %d (%s) has empty Plan", i, cand.Name)
				}
				if cand.Expr == nil {
					t.Fatalf("candidate %d (%s) has nil Expr", i, cand.Name)
				}
				if cand.PlanSignature == "" {
					t.Fatalf("candidate %d (%s) has empty PlanSignature", i, cand.Name)
				}
				if seenSigs[cand.PlanSignature] {
					t.Fatalf("duplicate PlanSignature %s at index %d", cand.PlanSignature, i)
				}
				seenSigs[cand.PlanSignature] = true

				if i > 0 && cand.EstimatedNs < beam[i-1].EstimatedNs {
					t.Fatalf("beam not sorted by EstimatedNs: beam[%d]=%f < beam[%d]=%f",
						i, cand.EstimatedNs, i-1, beam[i-1].EstimatedNs)
				}

				src, digest, err := cand.Source()
				if err != nil {
					t.Fatalf("candidate %d (%s) Source(): %v", i, cand.Name, err)
				}
				if err := VerifyDigest([]byte(src), digest); err != nil {
					t.Fatalf("candidate %d (%s) VerifyDigest: %v", i, cand.Name, err)
				}

				// Inspect structural diversity across extracted plans.
				rowCount := 0
				epiExtras := 0
				hasTail := false
				for _, k := range cand.Plan.Kernels {
					if k.Row != nil {
						rowCount++
					}
					if k.GEMM != nil {
						epiExtras += len(k.GEMM.Extras)
						if k.GEMM.Tail != nil {
							hasTail = true
						}
					}
					if k.Tile != nil {
						epiExtras += len(k.Tile.Extras)
					}
				}
				if rowCount >= 2 {
					hasUnfusedMultiKernel = true
				}
				if epiExtras >= 1 {
					hasEpilogueFused = true
				}
				if cand.Megakernel != nil || hasTail {
					hasMegakernelOrTail = true
				}
			}

			if !hasUnfusedMultiKernel {
				t.Errorf("expected at least one unfused multi-row-kernel candidate in beam")
			}
			if !hasEpilogueFused {
				t.Errorf("expected at least one epilogue-fused candidate in beam")
			}
			if !hasMegakernelOrTail {
				t.Errorf("expected at least one persistent megakernel or tail-fused candidate in beam")
			}
		})
	}
}

func TestStructuralBeamProfileExpansionAndPTXASPruning(t *testing.T) {
	b, root := buildTransformerFFNGraph(64, 128, 128)
	cfg := hopper.CostConfig{Device: "NVIDIA H200"}

	beam, err := ExtractStructuralPlanBeam(b.G, root, "ffn_pgo", hopper.F32, cfg, 4)
	if err != nil {
		t.Fatalf("ExtractStructuralPlanBeam: %v", err)
	}
	if len(beam) == 0 {
		t.Fatal("empty initial beam")
	}

	// Provide causal bottlenecks and stall samples for tile_gemm and row_kernel steps.
	causal := map[string]KernelObservation{
		"tile_gemm": {
			SharedBankConflict: 0.28,
			WGMMASerialized:    true,
			DeviceSMs:          132,
		},
		"row_kernel": {
			DurationNs:  12000,
			LaunchGapNs: 2500,
			DeviceSMs:   132,
		},
	}
	stalls := map[string]StallSample{
		"tile_gemm": {
			TotalSamples:   100,
			MIOThrottle:    35,
			LongScoreboard: 25,
		},
	}

	expanded := ExpandBeamWithProfile(beam, causal, stalls, 10)
	if len(expanded) <= len(beam) {
		t.Fatalf("ExpandBeamWithProfile returned %d candidates, want > %d (initial beam size)", len(expanded), len(beam))
	}
	if len(expanded) > 10 {
		t.Fatalf("ExpandBeamWithProfile returned %d candidates, want <= 10", len(expanded))
	}

	seenSigs := make(map[string]bool)
	hasProposals := false
	for _, cand := range expanded {
		if seenSigs[cand.PlanSignature] {
			t.Fatalf("ExpandBeamWithProfile produced duplicate signature %s", cand.PlanSignature)
		}
		seenSigs[cand.PlanSignature] = true
		if len(cand.StepProposals) > 0 {
			hasProposals = true
		}
	}
	if !hasProposals {
		t.Fatal("expected expanded candidates to carry StepProposals from ProposeStepCandidates")
	}

	// Construct PTXAS reports:
	// - expanded[0] spills registers (16B store, 16B load) -> must be rejected
	// - expanded[1] triggers WGMMA serialization C7519 -> must be rejected
	// - expanded[2] exceeds MaxRegisters (240 > 192) -> must be rejected
	// - remaining candidates are clean (128 regs, 0 spills) -> must be kept
	reports := make(map[string]ptxasinfo.Report)
	for i, cand := range expanded {
		vs := cand.Variants()
		if len(vs) == 0 {
			t.Fatalf("candidate %s has no variants", cand.Name)
		}
		targetKernel := vs[len(vs)-1].Name
		var logText string
		switch i {
		case 0:
			logText = "ptxas info    : 0 bytes gmem\n" +
				"ptxas info    : Compiling entry function '" + targetKernel + "' for 'sm_90a'\n" +
				"ptxas info    : Function properties for " + targetKernel + "\n" +
				"    32 bytes stack frame, 16 bytes spill stores, 16 bytes spill loads\n" +
				"ptxas info    : Used 128 registers, 16384 bytes smem, 384 bytes cmem[0]\n"
		case 1:
			logText = "ptxas info    : (C7519) Potential Performance Loss: wgmma.mma_async instructions are serialized due to program dependence on compiler-inserted WG.AR in divergent path in the function '" + targetKernel + "' in line 95\n" +
				"ptxas info    : Compiling entry function '" + targetKernel + "' for 'sm_90a'\n" +
				"ptxas info    : Function properties for " + targetKernel + "\n" +
				"    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n" +
				"ptxas info    : Used 128 registers, 16384 bytes smem, 384 bytes cmem[0]\n"
		case 2:
			logText = "ptxas info    : 0 bytes gmem\n" +
				"ptxas info    : Compiling entry function '" + targetKernel + "' for 'sm_90a'\n" +
				"ptxas info    : Function properties for " + targetKernel + "\n" +
				"    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n" +
				"ptxas info    : Used 240 registers, 16384 bytes smem, 384 bytes cmem[0]\n"
		default:
			logText = "ptxas info    : 0 bytes gmem\n" +
				"ptxas info    : Compiling entry function '" + targetKernel + "' for 'sm_90a'\n" +
				"ptxas info    : Function properties for " + targetKernel + "\n" +
				"    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n" +
				"ptxas info    : Used 128 registers, 16384 bytes smem, 384 bytes cmem[0]\n"
		}
		rep, err := ptxasinfo.Parse(strings.NewReader(logText))
		if err != nil {
			t.Fatalf("ptxasinfo.Parse: %v", err)
		}
		reports[cand.Name] = *rep
	}

	gate := StaticGate{
		NoSpills:     true,
		MaxRegisters: 192,
		MinCTAsPerSM: 1,
	}
	kept, rejected := FilterStructuralBeamByPTXAS(expanded, reports, "sm_90a", gate)
	if len(rejected) != 3 {
		t.Fatalf("got %d rejected candidates, want 3", len(rejected))
	}
	if len(kept) != len(expanded)-3 {
		t.Fatalf("got %d kept candidates, want %d", len(kept), len(expanded)-3)
	}

	if !strings.Contains(rejected[0].Reason(), "spills registers") {
		t.Errorf("rejected[0] reason = %q, want register spill reason", rejected[0].Reason())
	}
	if !strings.Contains(rejected[1].Reason(), "C7519") {
		t.Errorf("rejected[1] reason = %q, want C7519 serialization reason", rejected[1].Reason())
	}
	if !strings.Contains(rejected[2].Reason(), "registers 240 > max 192") {
		t.Errorf("rejected[2] reason = %q, want max registers reason", rejected[2].Reason())
	}
}

func TestStructuralBeamHoldoutTournamentSelection(t *testing.T) {
	b, root := buildTransformerFFNGraph(4, 256, 256)
	cfg := hopper.CostConfig{Device: "NVIDIA H200"}

	beam, err := ExtractStructuralPlanBeam(b.G, root, "ffn_holdout", hopper.F32, cfg, 5)
	if err != nil {
		t.Fatalf("ExtractStructuralPlanBeam: %v", err)
	}
	if len(beam) < 4 {
		t.Fatalf("need at least 4 candidates in beam, got %d", len(beam))
	}

	baseline := beam[0]
	candFastOverfit := beam[1]
	candTokenFlip := beam[2]
	candVerifiedWinner := beam[3]

	// Verify source digests of baseline and candidates using VerifyDigest.
	digests := make(map[string][32]byte)
	sources := make(map[string][]byte)
	for _, c := range []StructuralBeamCandidate{baseline, candFastOverfit, candTokenFlip, candVerifiedWinner} {
		src, dig, err := c.Source()
		if err != nil {
			t.Fatalf("%s.Source(): %v", c.Handle(), err)
		}
		if err := VerifyDigest([]byte(src), dig); err != nil {
			t.Fatalf("VerifyDigest(%s): %v", c.Handle(), err)
		}
		digests[c.Handle()] = dig
		sources[c.Handle()] = []byte(src)
	}

	query := MeasuredIdentity{
		Context:   sha256.Sum256([]byte("sm_90a:ffn_holdout_tournament_context")),
		Admission: sha256.Sum256([]byte("HOST_TOURNAMENT_ADMISSION_POLICY")),
	}
	refTokens := []int{17, 42, 9, 108, 3, 64, 12, 88}
	matchingTokens := []int{17, 42, 9, 108, 3, 64, 12, 88}
	flippedTokens := []int{17, 42, 99, 108, 3, 64, 12, 88}

	if err := VerifyTokenSequence(refTokens, matchingTokens); err != nil {
		t.Fatalf("VerifyTokenSequence(matching): %v", err)
	}
	if err := VerifyTokenSequence(refTokens, flippedTokens); err == nil {
		t.Fatal("VerifyTokenSequence accepted flipped greedy token sequence")
	}

	nullBlock := pairedtiming.Block{A1: 10.0, B1: 10.0, B2: 10.0, A2: 10.0}
	screenStrongWin := pairedtiming.Block{A1: 11.8, B1: 10.0, B2: 10.0, A2: 11.8}
	holdoutLoss := pairedtiming.Block{A1: 9.8, B1: 10.2, B2: 10.1, A2: 9.9}
	screenModestWin := pairedtiming.Block{A1: 11.0, B1: 10.0, B2: 10.0, A2: 11.0}
	holdoutVerifiedWin := pairedtiming.Block{A1: 11.2, B1: 10.0, B2: 10.0, A2: 11.2}

	evidence := map[string]HoldoutCandidateEvidence{
		// 1. Wins 3-block screen (1.18x) but loses on 6-block held-out ABBA -> REJECTED_HOLDOUT.
		candFastOverfit.Handle(): {
			Screen: MeasuredPair{
				Identity:  query,
				Reference: baseline.Handle(),
				Candidate: candFastOverfit.Handle(),
				Blocks:    []pairedtiming.Block{screenStrongWin, screenStrongWin, screenStrongWin},
			},
			HoldoutBlocks: []pairedtiming.Block{
				holdoutLoss, holdoutLoss, holdoutLoss, holdoutLoss, holdoutLoss, holdoutLoss,
			},
			AANullBlocks:    []pairedtiming.Block{nullBlock, nullBlock, nullBlock},
			ReferenceTokens: refTokens,
			CandidateTokens: matchingTokens,
		},
		// 2. Wins both screen and holdout timing (1.18x) but flips greedy token ID at index 2 -> REJECTED_TOKEN_MISMATCH.
		candTokenFlip.Handle(): {
			Screen: MeasuredPair{
				Identity:  query,
				Reference: baseline.Handle(),
				Candidate: candTokenFlip.Handle(),
				Blocks:    []pairedtiming.Block{screenStrongWin, screenStrongWin, screenStrongWin},
			},
			HoldoutBlocks: []pairedtiming.Block{
				screenStrongWin, screenStrongWin, screenStrongWin, screenStrongWin, screenStrongWin, screenStrongWin,
			},
			AANullBlocks:    []pairedtiming.Block{nullBlock, nullBlock, nullBlock},
			ReferenceTokens: refTokens,
			CandidateTokens: flippedTokens,
			LogitMaxAbs:     1.2e-4,
		},
		// 3. Passes 3-block screen (1.10x), 6-block holdout (1.12x, p=0.015625 <= 0.05), A/A null, and exact tokens -> PROMOTED.
		candVerifiedWinner.Handle(): {
			Screen: MeasuredPair{
				Identity:  query,
				Reference: baseline.Handle(),
				Candidate: candVerifiedWinner.Handle(),
				Blocks:    []pairedtiming.Block{screenModestWin, screenModestWin, screenModestWin},
			},
			HoldoutBlocks: []pairedtiming.Block{
				holdoutVerifiedWin, holdoutVerifiedWin, holdoutVerifiedWin,
				holdoutVerifiedWin, holdoutVerifiedWin, holdoutVerifiedWin,
			},
			AANullBlocks:    []pairedtiming.Block{nullBlock, nullBlock, nullBlock},
			ReferenceTokens: refTokens,
			CandidateTokens: matchingTokens,
			LogitMaxAbs:     0,
		},
	}

	verify := func(p MeasuredPair) error {
		if p.Identity != query {
			return errors.New("identity mismatch")
		}
		if err := VerifyDigest(sources[p.Reference], digests[p.Reference]); err != nil {
			return err
		}
		if err := VerifyDigest(sources[p.Candidate], digests[p.Candidate]); err != nil {
			return err
		}
		return nil
	}

	candidates := []StructuralBeamCandidate{candFastOverfit, candTokenFlip, candVerifiedWinner}
	winner, err := SelectStructuralBeamWithHoldout(query, baseline, candidates, evidence, verify)
	if err != nil {
		t.Fatalf("SelectStructuralBeamWithHoldout: %v", err)
	}
	if !winner.Promoted || winner.FallbackToBaseline {
		t.Fatalf("expected promoted winner, got %+v", winner)
	}
	if winner.Winner.PlanSignature != candVerifiedWinner.PlanSignature {
		t.Fatalf("Winner = %s (%s), want %s (%s)",
			winner.Winner.Handle(), winner.Winner.PlanSignature,
			candVerifiedWinner.Handle(), candVerifiedWinner.PlanSignature)
	}
	if winner.Status != "EXACT_HOLDOUT_PROMOTED" || !winner.TokensMatched {
		t.Fatalf("unexpected winner status/tokens: status=%q tokens=%v", winner.Status, winner.TokensMatched)
	}
	if winner.Holdout.Ratio < 1.11 || winner.OneSidedSignP > 0.05 {
		t.Fatalf("unexpected holdout metrics: ratio=%.4f p=%.4f", winner.Holdout.Ratio, winner.OneSidedSignP)
	}
	if len(winner.Evaluations) != 3 {
		t.Fatalf("got %d evaluations, want 3", len(winner.Evaluations))
	}
	if winner.Evaluations[0].Result.Status != "REJECTED_HOLDOUT" {
		t.Errorf("eval[0] (fast overfit) status = %q, want REJECTED_HOLDOUT", winner.Evaluations[0].Result.Status)
	}
	if winner.Evaluations[1].Result.Status != "REJECTED_TOKEN_MISMATCH" {
		t.Errorf("eval[1] (token flip) status = %q, want REJECTED_TOKEN_MISMATCH", winner.Evaluations[1].Result.Status)
	}
	if winner.Evaluations[2].Result.Status != "EXACT_HOLDOUT_PROMOTED" {
		t.Errorf("eval[2] (verified winner) status = %q, want EXACT_HOLDOUT_PROMOTED", winner.Evaluations[2].Result.Status)
	}

	// Verify fallback to baseline when no candidate passes holdout.
	onlyFailing := []StructuralBeamCandidate{candFastOverfit, candTokenFlip}
	fb, err := SelectStructuralBeamWithHoldout(query, baseline, onlyFailing, evidence, verify)
	if err != nil {
		t.Fatalf("SelectStructuralBeamWithHoldout(onlyFailing): %v", err)
	}
	if fb.Promoted || !fb.FallbackToBaseline {
		t.Fatalf("expected fallback to baseline, got Promoted=%v FallbackToBaseline=%v", fb.Promoted, fb.FallbackToBaseline)
	}
	if fb.Winner.PlanSignature != baseline.PlanSignature {
		t.Fatalf("fallback Winner = %s, want baseline %s", fb.Winner.Handle(), baseline.Handle())
	}
}
