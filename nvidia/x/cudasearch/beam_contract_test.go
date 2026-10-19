package cudasearch

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

type beamEvaluatorFunc func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error)

func (f beamEvaluatorFunc) Evaluate(ctx context.Context, s *egraph.SearchState, id egraph.Id) (*CandidatePlan, error) {
	return f(ctx, s, id)
}

func beamContractState() (*egraph.SearchState, []*egraph.Rewrite) {
	g := egraph.New()
	root := g.Add(egraph.ENode{Op: "base"})
	return egraph.NewSearchState(g, map[string]egraph.Id{"root": root}), []*egraph.Rewrite{egraph.MustRewrite("alternative", "base", "alternative")}
}

func TestBeamAdmissionBeforeEvaluation(t *testing.T) {
	for _, kind := range []string{"canceled", "memory", "root"} {
		t.Run(kind, func(t *testing.T) {
			s, rules := beamContractState()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			cfg := BeamConfig{Budget: DefaultBudget()}
			switch kind {
			case "canceled":
				cancel()
			case "memory":
				cfg.Budget.MaxMemoryBytes = 1
			case "root":
				cfg.RootName = "absent"
			}
			_, err := NewBeamSearch(cfg).Search(ctx, s, rules, beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
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

func TestBeamRetainsBestAfterStop(t *testing.T) {
	for _, kind := range []string{"canceled", "memory", "states", "evaluations"} {
		t.Run(kind, func(t *testing.T) {
			s, rules := beamContractState()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			cfg := BeamConfig{Budget: SearchBudget{MaxStates: 10, MaxDepth: 2, MaxEvaluations: 10}}
			switch kind {
			case "memory":
				cfg.Budget.CallbackMemoryBytes = cloneCallbackBytes()
				cfg.Budget.MaxMemoryBytes = transitionIncumbentBudget(t, cfg.Budget, 1, 10, func(budget SearchBudget, eval Evaluator) (*SearchResult, error) {
					probe := cfg
					probe.Budget = budget
					return NewBeamSearch(probe).Search(ctx, s, rules, eval)
				})
			case "states":
				cfg.Budget.MaxStates = 1
			case "evaluations":
				cfg.Budget.MaxEvaluations = 1
			}
			calls := 0
			base := &CandidatePlan{ID: "baseline", Cost: 10, Validated: true}
			var eval Evaluator = beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
				calls++
				if calls == 1 {
					return base, nil
				}
				cancel()
				return nil, context.Canceled
			})
			if kind == "memory" {
				cfg.Budget, eval = transitionSetupBudget(t, cfg.Budget, eval)
			}
			res, err := NewBeamSearch(cfg).Search(ctx, s, rules, eval)
			if kind == "memory" && res != nil {
				defer res.Close()
			}
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

func TestBeamRejectsMalformedEvaluatorResults(t *testing.T) {
	for _, bad := range []*CandidatePlan{nil, {Cost: 1}, {Cost: math.NaN(), Validated: true}, {Cost: math.Inf(-1), Validated: true}} {
		s, rules := beamContractState()
		calls := 0
		base := &CandidatePlan{ID: "baseline", Cost: 10, Validated: true}
		res, err := NewBeamSearch(BeamConfig{Budget: SearchBudget{MaxStates: 2, MaxDepth: 1, MaxEvaluations: 2}}).Search(context.Background(), s, rules, beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
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
