package qualify

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

// MeasuredIdentity identifies one complete measured context and its admission.
// Context is the digest of an explicit source/recipe/case/device/image/boundary
// descriptor, not a variant name or an e-graph operation. Admission identifies
// trusted caller policy; these digests do not authenticate themselves.
type MeasuredIdentity struct{ Context, Admission [32]byte }

// MeasuredPair describes opaque whole-variant handles and observed ABBA blocks.
// It is not a device entry, node cost, or native qualification. A trusted caller
// must verify both the complete descriptor and the exact data-to-block binding.
type MeasuredPair struct {
	Identity             MeasuredIdentity
	Reference, Candidate string
	Blocks               []pairedtiming.Block
}

// MeasuredSelection reports a local exact-context decision. An unevaluated
// result has no Selected or Observed value, rather than synthetic zero latency
// or a default ratio of one. Observed.Candidate is the paired estimator's derived
// latency, not an independently measured candidate median or a confidence bound.
type MeasuredSelection struct {
	Status, Reason, CandidateState string
	Selected                       *Variant
	Observed                       *pairedtiming.Summary
}

// SelectMeasured filters by exact context/admission, then applies the existing
// SelectByPairedTiming screen with intercept zero and DefaultMinGain. It leaves
// analytical extraction and RankVariants unchanged.
//
// verify is mandatory trusted caller policy, not a proof supplied by this API.
// It must authenticate the complete source/recipe/input/runtime identity, root
// admission, and exact observed blocks; a raw PASS or copied admission digest is
// insufficient. A callback that merely returns nil grants no native authority.
// The methods consumer provides a source-bound verifier; host tests may use
// explicitly synthetic callbacks. No selector is called for an invalid,
// ambiguous, missing, or rejected pair. This API never exports the selector's
// synthetic reference Summary as a measurement.
func SelectMeasured(query MeasuredIdentity, pairs []MeasuredPair, verify func(MeasuredPair) error) (MeasuredSelection, error) {
	return selectMeasured(query, pairs, verify, SelectByPairedTiming)
}

type measuredSelector func([]Variant, map[string][]pairedtiming.Block, float64) (Variant, pairedtiming.Summary, error)

func selectMeasured(query MeasuredIdentity, pairs []MeasuredPair, verify func(MeasuredPair) error, selectPair measuredSelector) (MeasuredSelection, error) {
	no := func(reason string) (MeasuredSelection, error) {
		return MeasuredSelection{Status: "NOT_EVALUATED", Reason: reason}, nil
	}
	if query.Context == ([32]byte{}) || query.Admission == ([32]byte{}) {
		return no("complete context and trusted admission identities required")
	}
	if verify == nil {
		return no("trusted caller admission verifier absent")
	}
	var pair MeasuredPair
	count := 0
	for _, p := range pairs {
		if p.Identity == query {
			pair = p
			count++
		}
	}
	if count == 0 {
		return no("no exact admitted pair")
	}
	if count != 1 {
		return no("ambiguous exact pair")
	}
	if pair.Reference == "" || pair.Candidate == "" || pair.Reference == pair.Candidate {
		return no("distinct opaque reference/candidate handles required")
	}
	pair.Blocks = append([]pairedtiming.Block(nil), pair.Blocks...)
	if len(pair.Blocks) != 3 {
		return no("three observed ABBA blocks required")
	}
	verification := pair
	verification.Blocks = append([]pairedtiming.Block(nil), pair.Blocks...)
	if err := verify(verification); err != nil {
		return no("admission: " + err.Error())
	}
	observed, err := pairedtiming.Analyze(pair.Blocks, 0)
	if err != nil {
		return no("paired timing: " + err.Error())
	}
	for _, v := range append(append([]float64(nil), observed.Ratios...), observed.Ratio, observed.MinRatio, observed.Baseline, observed.Candidate) {
		if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
			return no("nonfinite or nonpositive paired observation")
		}
	}
	variants := []Variant{{Name: pair.Reference}, {Name: pair.Candidate}}
	selected, _, err := selectPair(variants, map[string][]pairedtiming.Block{pair.Candidate: pair.Blocks}, 0)
	if err != nil {
		return MeasuredSelection{Status: "NOT_EVALUATED", Reason: "selector rejected admitted observation"}, fmt.Errorf("select measured: %w", err)
	}
	state := "MEASURED_NO_PROMOTION"
	if observed.Ratio < 1 {
		state = "MEASURED_LOSS"
	}
	if selected.Name == pair.Candidate {
		state = "PROMOTED_LOCAL_EXACT_CONTEXT"
	} else if selected.Name != pair.Reference {
		return no("selector returned an unknown opaque handle")
	}
	return MeasuredSelection{Status: "EXACT_MEASURED_COMPARISON", CandidateState: state, Selected: &selected, Observed: &observed}, nil
}

// HoldoutPromotionQuery is an alias for [MeasuredIdentity] used by
// [SelectHoldoutPromoted].
type HoldoutPromotionQuery = MeasuredIdentity

// HoldoutCandidateEvidence bundles the 3-block screen pair, the independent
// held-out ABBA blocks (>= 6 blocks), optional A/A null-control blocks, and
// frozen-prompt greedy token-ID trajectories for final promotion gating.
type HoldoutCandidateEvidence struct {
	Screen          MeasuredPair
	HoldoutBlocks   []pairedtiming.Block // must have len >= 6
	AANullBlocks    []pairedtiming.Block // optional or required A/A null blocks (baseline vs baseline)
	ReferenceTokens []int                // frozen-prompt greedy token IDs from reference
	CandidateTokens []int                // frozen-prompt greedy token IDs from candidate
	LogitMaxAbs     float64              // diagnostic only; never substitutes for exact token IDs
}

// HoldoutPromotionResult reports the outcome of the two-stage screen +
// held-out ABBA + A/A null + exact token-ID promotion gate.
type HoldoutPromotionResult struct {
	Status, Reason, CandidateState string
	Selected                       *Variant
	ScreenObserved                 *pairedtiming.Summary
	HoldoutObserved                *pairedtiming.Summary
	AANullObserved                 *pairedtiming.Summary
	OneSidedSignP                  float64
	TokensMatched                  bool
}

// SelectHoldoutPromoted runs the two-stage promotion gate:
//  1. 3-block fast screen via [SelectMeasured].
//  2. Exact frozen-prompt greedy token-ID agreement (never substituted by
//     LogitMaxAbs).
//  3. Optional A/A null-control drift check (rejecting when baseline vs.
//     baseline blocks falsely promote under [SelectByPairedTiming]).
//  4. Held-out ABBA gate requiring >= 6 authenticated blocks, >= 6 non-tied
//     blocks, exact one-sided binomial sign-test p <= 0.05, median ratio >=
//     [DefaultMinGain], and MinRatio >= 1.0.
func SelectHoldoutPromoted(query MeasuredIdentity, ev HoldoutCandidateEvidence, verify func(MeasuredPair) error) (HoldoutPromotionResult, error) {
	screen, err := SelectMeasured(query, []MeasuredPair{ev.Screen}, verify)
	if err != nil {
		return HoldoutPromotionResult{
			Status:         screen.Status,
			Reason:         screen.Reason,
			CandidateState: screen.CandidateState,
			Selected:       screen.Selected,
			ScreenObserved: screen.Observed,
		}, err
	}
	if screen.Status != "EXACT_MEASURED_COMPARISON" || screen.CandidateState != "PROMOTED_LOCAL_EXACT_CONTEXT" {
		reason := screen.Reason
		if reason == "" {
			reason = fmt.Sprintf("three-block screen did not promote candidate (state=%s)", screen.CandidateState)
		}
		return HoldoutPromotionResult{
			Status:         screen.Status,
			Reason:         reason,
			CandidateState: screen.CandidateState,
			Selected:       screen.Selected,
			ScreenObserved: screen.Observed,
		}, nil
	}

	refVariant := &Variant{Name: ev.Screen.Reference}
	if len(ev.ReferenceTokens) == 0 || len(ev.CandidateTokens) == 0 {
		return HoldoutPromotionResult{
			Status:         "REJECTED_TOKEN_MISMATCH",
			Reason:         "non-empty frozen-prompt greedy token-ID trajectories required",
			CandidateState: "REJECTED_TOKEN_MISMATCH",
			Selected:       refVariant,
			ScreenObserved: screen.Observed,
			TokensMatched:  false,
		}, nil
	}
	minTok := min(len(ev.ReferenceTokens), len(ev.CandidateTokens))
	for i := range minTok {
		if ev.ReferenceTokens[i] != ev.CandidateTokens[i] {
			return HoldoutPromotionResult{
				Status:         "REJECTED_TOKEN_MISMATCH",
				Reason:         fmt.Sprintf("greedy token ID mismatch at index %d: reference=%d candidate=%d (logit_max_abs=%g)", i, ev.ReferenceTokens[i], ev.CandidateTokens[i], ev.LogitMaxAbs),
				CandidateState: "REJECTED_TOKEN_MISMATCH",
				Selected:       refVariant,
				ScreenObserved: screen.Observed,
				TokensMatched:  false,
			}, nil
		}
	}
	if len(ev.ReferenceTokens) != len(ev.CandidateTokens) {
		return HoldoutPromotionResult{
			Status:         "REJECTED_TOKEN_MISMATCH",
			Reason:         fmt.Sprintf("greedy token trajectory length mismatch at index %d: reference=%d candidate=%d", minTok, len(ev.ReferenceTokens), len(ev.CandidateTokens)),
			CandidateState: "REJECTED_TOKEN_MISMATCH",
			Selected:       refVariant,
			ScreenObserved: screen.Observed,
			TokensMatched:  false,
		}, nil
	}

	var aaObserved *pairedtiming.Summary
	if len(ev.AANullBlocks) > 0 {
		aaBlocks := append([]pairedtiming.Block(nil), ev.AANullBlocks...)
		aaSum, err := pairedtiming.Analyze(aaBlocks, 0)
		if err != nil || !summaryFinitePositive(aaSum) {
			return HoldoutPromotionResult{
				Status:         "REJECTED_AA_NULL_DRIFT",
				Reason:         "invalid A/A null-control timing blocks",
				CandidateState: "MEASURED_NO_PROMOTION",
				Selected:       refVariant,
				ScreenObserved: screen.Observed,
				TokensMatched:  true,
			}, nil
		}
		aaObserved = &aaSum
		variants := []Variant{{Name: ev.Screen.Reference}, {Name: ev.Screen.Candidate}}
		aaSel, _, aaErr := SelectByPairedTiming(variants, map[string][]pairedtiming.Block{ev.Screen.Candidate: aaBlocks}, 0)
		if aaErr != nil || aaSel.Name == ev.Screen.Candidate {
			return HoldoutPromotionResult{
				Status:         "REJECTED_AA_NULL_DRIFT",
				Reason:         fmt.Sprintf("A/A null control falsely promoted with ratio=%.4f min_ratio=%.4f", aaSum.Ratio, aaSum.MinRatio),
				CandidateState: "MEASURED_NO_PROMOTION",
				Selected:       refVariant,
				ScreenObserved: screen.Observed,
				AANullObserved: aaObserved,
				TokensMatched:  true,
			}, nil
		}
	}

	holdoutBlocks := append([]pairedtiming.Block(nil), ev.HoldoutBlocks...)
	if len(holdoutBlocks) < 6 {
		return HoldoutPromotionResult{
			Status:         "REJECTED_HOLDOUT",
			Reason:         fmt.Sprintf("at least 6 held-out ABBA blocks required, got %d", len(holdoutBlocks)),
			CandidateState: "MEASURED_NO_PROMOTION",
			Selected:       refVariant,
			ScreenObserved: screen.Observed,
			AANullObserved: aaObserved,
			TokensMatched:  true,
		}, nil
	}

	verifyHoldout := MeasuredPair{
		Identity:  query,
		Reference: ev.Screen.Reference,
		Candidate: ev.Screen.Candidate,
		Blocks:    append([]pairedtiming.Block(nil), holdoutBlocks...),
	}
	if err := verify(verifyHoldout); err != nil {
		return HoldoutPromotionResult{
			Status:         "NOT_EVALUATED",
			Reason:         "holdout admission: " + err.Error(),
			CandidateState: "MEASURED_NO_PROMOTION",
			ScreenObserved: screen.Observed,
			AANullObserved: aaObserved,
			TokensMatched:  true,
		}, nil
	}

	holdoutSum, err := pairedtiming.Analyze(holdoutBlocks, 0)
	if err != nil || !summaryFinitePositive(holdoutSum) {
		return HoldoutPromotionResult{
			Status:         "NOT_EVALUATED",
			Reason:         "nonfinite or nonpositive held-out paired observation",
			CandidateState: "MEASURED_NO_PROMOTION",
			ScreenObserved: screen.Observed,
			AANullObserved: aaObserved,
			TokensMatched:  true,
		}, nil
	}

	nonTied := holdoutSum.Wins + holdoutSum.Losses
	oneSidedP := oneSidedSignTest(holdoutSum.Wins, holdoutSum.Losses)
	state := "MEASURED_NO_PROMOTION"
	if holdoutSum.Ratio < 1.0 {
		state = "MEASURED_LOSS"
	}
	if nonTied < 6 || oneSidedP > 0.05 || holdoutSum.Ratio < DefaultMinGain || holdoutSum.MinRatio < 1.0 {
		return HoldoutPromotionResult{
			Status:          "REJECTED_HOLDOUT",
			Reason:          fmt.Sprintf("held-out ABBA gate failed: n=%d wins=%d p=%.4f ratio=%.4f min_ratio=%.4f", nonTied, holdoutSum.Wins, oneSidedP, holdoutSum.Ratio, holdoutSum.MinRatio),
			CandidateState:  state,
			Selected:        refVariant,
			ScreenObserved:  screen.Observed,
			HoldoutObserved: &holdoutSum,
			AANullObserved:  aaObserved,
			OneSidedSignP:   oneSidedP,
			TokensMatched:   true,
		}, nil
	}

	candVariant := &Variant{Name: ev.Screen.Candidate}
	return HoldoutPromotionResult{
		Status:          "EXACT_HOLDOUT_PROMOTED",
		CandidateState:  "PROMOTED_HOLDOUT_VERIFIED",
		Selected:        candVariant,
		ScreenObserved:  screen.Observed,
		HoldoutObserved: &holdoutSum,
		AANullObserved:  aaObserved,
		OneSidedSignP:   oneSidedP,
		TokensMatched:   true,
	}, nil
}

func summaryFinitePositive(s pairedtiming.Summary) bool {
	for _, v := range append(append([]float64(nil), s.Ratios...), s.Ratio, s.MinRatio, s.Baseline, s.Candidate) {
		if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
			return false
		}
	}
	return true
}

func oneSidedSignTest(wins, losses int) float64 {
	n := wins + losses
	if n <= 0 || wins <= 0 {
		return 1
	}
	lgn, _ := math.Lgamma(float64(n + 1))
	var p float64
	for k := wins; k <= n; k++ {
		a, _ := math.Lgamma(float64(k + 1))
		b, _ := math.Lgamma(float64(n - k + 1))
		p += math.Exp(lgn - a - b - float64(n)*math.Ln2)
	}
	return min(1, p)
}
