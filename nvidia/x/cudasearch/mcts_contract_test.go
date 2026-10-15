package cudasearch

import (
	"context"
	"errors"
	"github.com/tmc/lib/nvidia/x/egraph"
	"math"
	"testing"
)

func TestMCTSAdmissionBeforeEvaluation(t *testing.T) {
	for _, kind := range []string{"canceled", "memory", "root"} {
		t.Run(kind, func(t *testing.T) {
			s, rules := beamContractState()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			cfg := MCTSConfig{Budget: DefaultBudget()}
			switch kind {
			case "canceled":
				cancel()
			case "memory":
				cfg.Budget.MaxMemoryBytes = 1
			case "root":
				cfg.RootName = "absent"
			}
			_, err := NewMCTSSearch(cfg).Search(ctx, s, rules, beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
				t.Fatal("evaluated inadmissible search")
				return nil, nil
			}))
			if err == nil {
				t.Fatal("accepted search")
			}
			if kind == "canceled" && (!errors.Is(err, context.Canceled) || !errors.Is(err, ErrSearchCancelled)) {
				t.Fatalf("cancellation identity lost: %v", err)
			}
		})
	}
}

func TestMCTSRetainsBestAfterStop(t *testing.T) {
	for _, kind := range []string{"canceled", "memory", "states", "evaluations"} {
		t.Run(kind, func(t *testing.T) {
			s, rules := beamContractState()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			cfg := MCTSConfig{Budget: SearchBudget{MaxStates: 10, MaxDepth: 2, MaxEvaluations: 10}}
			switch kind {
			case "memory":
				cfg.Budget.CallbackMemoryBytes = cloneCallbackBytes()
				cfg.Budget.MaxMemoryBytes = cloneLaterRefusalBytes(t, s, rules, &CandidatePlan{ID: "baseline", Cost: 10, Validated: true})
			case "states":
				cfg.Budget.MaxStates = 1
			case "evaluations":
				cfg.Budget.MaxEvaluations = 1
			}
			calls := 0
			base := &CandidatePlan{ID: "baseline", Cost: 10, Validated: true}
			res, err := NewMCTSSearch(cfg).Search(ctx, s, rules, beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
				calls++
				if calls == 1 {
					return base, nil
				}
				cancel()
				return nil, context.Canceled
			}))
			if res == nil || res.Winner != base || res.StopReason == nil {
				t.Fatalf("lost best plan: %+v %v", res, err)
			}
			switch kind {
			case "canceled":
				if !errors.Is(err, ErrSearchCancelled) || len(res.Rejected) != 1 {
					t.Fatalf("canceled: %+v %v", res, err)
				}
			case "memory":
				if !errors.Is(err, ErrMemoryBudgetExceeded) || calls != 1 {
					t.Fatalf("memory: %+v %v", res, err)
				}
			default:
				if err != nil || calls != 1 || res.StatesVisited != 1 || !errors.Is(res.StopReason, ErrBudgetExhausted) {
					t.Fatalf("budget: %+v %v", res, err)
				}
			}
		})
	}
}

func TestMCTSRejectsMalformedEvaluatorResults(t *testing.T) {
	for _, bad := range []*CandidatePlan{nil, {Cost: 1}, {Cost: math.NaN(), Validated: true}, {Cost: math.Inf(-1), Validated: true}} {
		s, rules := beamContractState()
		calls := 0
		base := &CandidatePlan{ID: "baseline", Cost: 10, Validated: true}
		res, err := NewMCTSSearch(MCTSConfig{Budget: SearchBudget{MaxStates: 2, MaxDepth: 1, MaxEvaluations: 2}}).Search(context.Background(), s, rules, beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
			calls++
			if calls == 1 {
				return base, nil
			}
			return bad, nil
		}))
		if err != nil || res.Winner != base || len(res.Rejected) != 1 || len(res.Rejected[0].Derivation) != 1 || !errors.Is(res.Rejected[0].Err, ErrNoAdmittedPlan) {
			t.Fatalf("rejection lost: %+v %v", res, err)
		}
	}
}
