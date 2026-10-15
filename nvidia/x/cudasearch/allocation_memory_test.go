package cudasearch

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestAllocationSearchEnumerationSharedLedger(t *testing.T) {
	state, _ := beamContractState()
	rules := []*egraph.Rewrite{{Name: "replace", LHS: egraph.Var("x"), RHS: egraph.Pat("replacement")}}
	ledger := &memoryBudget{limit: 1 << 20}
	ctx := context.WithValue(context.Background(), memoryBudgetKey{}, ledger)
	owner, err := enumerateSearchActions(ctx, state, rules, 20)
	if err != nil {
		t.Fatal(err)
	}
	if len(owner.Actions) == 0 || ledger.live == 0 {
		t.Fatal("enumeration bypassed ledger")
	}
	held := ledger.live
	workspace, err := ledger.reserve(ledger.limit - held)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ledger.reserve(1); !errors.Is(err, ErrMemoryBudgetExceeded) {
		t.Fatal("workspace ignored matching storage", err)
	}
	owner.Close()
	owner.Close()
	if ledger.live != ledger.limit-held {
		t.Fatal("owner did not release exactly once", ledger.live, held)
	}
	workspace.release()
	if ledger.live != 0 {
		t.Fatal(ledger.live)
	}
	ledger.limit = 1
	owner, err = enumerateSearchActions(ctx, state, rules, 20)
	if owner != nil || !errors.Is(err, ErrMemoryBudgetExceeded) || ledger.live != 0 {
		t.Fatal("partial allocation on refusal", owner, err, ledger.live)
	}
}

func TestAllocationPublicSearchMatchingLimit(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			state, _ := beamContractState()
			rules := []*egraph.Rewrite{{Name: "wide", LHS: egraph.Var("x"), RHS: egraph.Pat("replacement")}}
			// The evaluator fills the remaining allowance after the initial
			// guarded clone is admitted. Leave only the returned candidate's
			// measured storage, so the next matching owner must refuse.
			calls := 0
			eval := beamEvaluatorFunc(func(ctx context.Context, current *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
				calls++
				ledger := searchMemoryBudget(ctx)
				candidate := &CandidatePlan{Cost: 1, Validated: true}
				candidateBytes, err := storageBytes(candidate)
				if err != nil {
					return nil, err
				}
				_, err = ledger.reserve(ledger.limit - ledger.live - candidateBytes)
				return candidate, err
			})
			budget := SearchBudget{MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: 1, MaxStates: 3, MaxDepth: 1, MaxActions: 10, MaxEvaluations: 2}
			result, err := memoryStrategy(strategy, budget, state, rules, eval)
			if !errors.Is(err, ErrMemoryBudgetExceeded) || calls != 1 || result == nil || result.Winner == nil || result.ActionsAttempted != 0 {
				t.Fatalf("result=%+v calls=%d err=%v", result, calls, err)
			}
		})
	}
}
