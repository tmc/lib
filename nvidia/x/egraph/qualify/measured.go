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
