package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func promotionFixture(t *testing.T) (*SearchResult, *CandidatePlan) {
	t.Helper()
	e, _, _, _ := executableFixture(t)
	e.Generation = 7
	e.Workload.InputIdentities = map[string]string{"x": "screen"}
	r, err := NewBeamSearch(BeamConfig{EvaluatorID: "host-fixture-v1"}).Search(context.Background(), executableState(), nil, e)
	if err != nil {
		t.Fatal(err)
	}
	e.Workload.InputIdentities["x"] = "held"
	c, err := e.Confirm(context.Background(), r.Winner)
	if err != nil {
		t.Fatal(err)
	}
	return r, c
}

func TestPromotionDetachedEvidence(t *testing.T) {
	r, c := promotionFixture(t)
	p, err := r.ProposePromotion(c, 7)
	if err != nil {
		t.Fatal(err)
	}
	if p.Generation != 7 || p.CandidateID != r.Winner.ID || p.ConfirmationID != c.ID || len(p.HeldOutTrials) != 4 || len(p.ScreeningTrials) != 4 || p.WorkloadKey == p.HeldOutWorkloadKey {
		t.Fatalf("proposal=%+v", p)
	}
	r.PolicyID.Options["evaluator_identity"] = "changed"
	r.Winner.Trials[0].Provenance["executable_identity"] = "changed"
	c.Trials[1].Provenance["executable_identity"] = "changed"
	c.Receipt.Provenance["executable_identity"] = "changed"
	c.Comparison["out"] = OutputComparison{MaxAbsolute: 999}
	if p.PolicyID.Options["evaluator_identity"] == "changed" || p.ScreeningTrials[0].Provenance["executable_identity"] == "changed" || p.HeldOutTrials[1].Provenance["executable_identity"] == "changed" || p.HeldOutReceipt.Provenance["executable_identity"] == "changed" || p.HeldOutComparison["out"].MaxAbsolute == 999 {
		t.Fatal("proposal aliases mutable evidence")
	}
}

func TestStaleGenerationRejected(t *testing.T) {
	r, c := promotionFixture(t)
	for _, current := range []int{-1, 0, 6, 8} {
		if p, err := r.ProposePromotion(c, current); p != nil || !errors.Is(err, ErrStaleGeneration) {
			t.Fatalf("generation=%d proposal=%v err=%v", current, p, err)
		}
	}
	p, err := r.ProposePromotion(c, 7)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.ValidateGeneration(7); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(p.ValidateGeneration(8), ErrStaleGeneration) {
		t.Fatal("owner boundary accepted stale proposal")
	}
	c.Generation = 8
	if _, err := r.ProposePromotion(c, 7); !errors.Is(err, ErrStaleGeneration) {
		t.Fatal("accepted mismatched confirmation generation")
	}
}

func TestPromotionRefusesIncompleteEvidence(t *testing.T) {
	for _, reason := range []string{"policy", "unconfirmed", "same_inputs", "estimated", "unvalidated", "comparison", "trials", "failed_trial", "wrong_arm", "wrong_identity", "selected_identity", "round", "timing", "tolerance"} {
		t.Run(reason, func(t *testing.T) {
			r, c := promotionFixture(t)
			switch reason {
			case "policy":
				r.PolicyComplete = false
			case "unconfirmed":
				c.ConfirmationOf = "other"
			case "same_inputs":
				c.InputKey = r.Winner.InputKey
			case "estimated":
				c.CostSource = CostEstimated
			case "unvalidated":
				c.Validated = false
			case "comparison":
				c.Comparison = nil
			case "trials":
				c.Trials = nil
			case "failed_trial":
				c.Trials[0].Complete = false
			case "wrong_arm":
				c.Trials[1].Name = "baseline"
			case "wrong_identity":
				c.Trials[1].Provenance["executable_identity"] = "other"
			case "selected_identity":
				c.Receipt.Provenance["executable_identity"] = "other"
			case "round":
				c.Trials[1].Round++
			case "timing":
				c.Receipt.TimingKind = cudatrial.TimingGPUBusy
			case "tolerance":
				c.VerificationPolicy.Absolute = 1
			}
			if p, err := r.ProposePromotion(c, 7); err == nil || p != nil {
				t.Fatalf("proposal=%v err=%v", p, err)
			}
		})
	}
}

func ExampleSearchResult_ProposePromotion() {
	order := []string{}
	x := &testExecutable{id: "image", value: 1, kind: cudatrial.TimingSummedKernel, duration: 1, order: &order}
	e := &ExecutableEvaluator{Generation: 7, Workload: WorkloadBindings{Name: "host-fixture", InputNames: []string{"x"}, InputIdentities: map[string]string{"x": "screen"}, NumericalPolicy: NumericalPolicyExact},
		Baseline: x, References: map[string][]float32{"out": {1}}, Rounds: 2, TimingKind: cudatrial.TimingSummedKernel, HardwareLockChecker: func() bool { return true },
		Compile: func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) { y := *x; return &y, nil },
	}
	result, _ := NewBeamSearch(BeamConfig{EvaluatorID: "synthetic-v1"}).Search(context.Background(), executableState(), nil, e)
	e.Workload.InputIdentities["x"] = "held"
	confirmed, _ := e.Confirm(context.Background(), result.Winner)
	proposal, err := result.ProposePromotion(confirmed, 7)
	fmt.Println(err, proposal.CandidateID, proposal.Generation)
	// Output: <nil> image 7
}

func ExamplePromotionProposal_ValidateGeneration() {
	proposal := &PromotionProposal{Generation: 7}
	fmt.Println(proposal.ValidateGeneration(7))
	fmt.Println(errors.Is(proposal.ValidateGeneration(8), ErrStaleGeneration))
	// Output:
	// <nil>
	// true
}
