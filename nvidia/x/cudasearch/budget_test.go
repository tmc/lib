package cudasearch

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestSearchWorkBudget(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		for _, phase := range []string{"extract", "compile", "unsupported"} {
			t.Run(strategy+"/"+phase, func(t *testing.T) {
				s, rules := beamContractState()
				budget := DefaultBudget()
				// The synthetic compiler returns borrowed fixtures; output/receipt maps
				// are bounded by four ABBA calls over one float32 output.
				budget.CallbackMemoryBytes = fixtureCallbackMemoryBytes(budget.MaxDepth)
				e, _, compiled, _ := executableFixture(t)
				var eval Evaluator = e
				if phase == "compile" {
					budget.MaxCompilations = 1
				} else {
					budget.MaxExtractions = 1
				}
				if phase == "unsupported" {
					eval = beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
						t.Fatal("called unsupported evaluator")
						return nil, nil
					})
				}
				var result *SearchResult
				var err error
				if strategy == "beam" {
					result, err = NewBeamSearch(BeamConfig{Budget: budget}).Search(context.Background(), s, rules, eval)
				} else {
					result, err = NewMCTSSearch(MCTSConfig{Budget: budget}).Search(context.Background(), s, rules, eval)
				}
				if phase == "unsupported" {
					if err == nil || result != nil {
						t.Fatal("accepted unsupported work accounting")
					}
					return
				}
				if err != nil || result == nil || result.Winner == nil || !result.WorkAccounted || result.Compilations != 1 || compiled.closed != 1 || !errors.Is(result.StopReason, ErrBudgetExhausted) {
					t.Fatalf("result=%+v err=%v closed=%d", result, err, compiled.closed)
				}
				wantExtractions := 1
				if phase == "compile" {
					wantExtractions = 2
				}
				if result.Extractions != wantExtractions || len(result.Rejected) != 1 {
					t.Fatalf("result=%+v", result)
				}
			})
		}
	}
}

func TestFailedCompileConsumesBudget(t *testing.T) {
	e, _, compiled, _ := executableFixture(t)
	failure := errors.New("compiler failed")
	calls := 0
	e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) {
		calls++
		return compiled, failure
	}
	ctx, work, err := searchWorkBudget(context.Background(), SearchBudget{MaxCompilations: 1}, e)
	if err != nil {
		t.Fatal(err)
	}
	s := executableState()
	if _, err := e.Evaluate(ctx, s, s.Roots["root"]); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	if _, err := e.Evaluate(ctx, s, s.Roots["root"]); !errors.Is(err, ErrBudgetExhausted) {
		t.Fatal(err)
	}
	if calls != 1 || compiled.closed != 1 || work.compilations != 1 || work.extractions != 2 {
		t.Fatalf("calls=%d closed=%d work=%+v", calls, compiled.closed, work)
	}
}
