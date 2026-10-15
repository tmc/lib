package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func bookkeepingSearch(ctx context.Context, strategy string, budget SearchBudget, state *egraph.SearchState, rules []*egraph.Rewrite, eval Evaluator) (*SearchResult, error) {
	if strategy == "beam" {
		return NewBeamSearch(BeamConfig{BeamWidth: 1, Budget: budget}).Search(ctx, state, rules, eval)
	}
	return NewMCTSSearch(MCTSConfig{Iterations: 4, RolloutDepth: 1, Seed: 42, Budget: budget}).Search(ctx, state, rules, eval)
}

func TestBookkeepingPublicOwners(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		for _, mode := range []string{"normal", "rejected", "canceled"} {
			t.Run(strategy+"/"+mode, func(t *testing.T) {
				state, rules := beamContractState()
				before := stateFingerprint(state)
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				const limit = 1 << 20
				var callback context.Context
				var book *searchBookkeeping
				var release func()
				calls := 0
				failure := errors.New("candidate refused")
				eval := beamEvaluatorFunc(func(ctx context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
					calls++
					callback = ctx
					book = searchMemoryBudget(ctx).book
					if release == nil {
						var err error
						release, err = ReserveSearchMemory(ctx, 137)
						if err != nil {
							return nil, err
						}
					}
					if calls > 1 && mode != "normal" {
						if mode == "canceled" {
							cancel()
						}
						return nil, failure
					}
					return &CandidatePlan{Cost: float64(10 - calls), Validated: true, Derivation: s.Derivation}, nil
				})
				budget := SearchBudget{MaxMemoryBytes: limit, CallbackMemoryBytes: 4096, MaxStates: 5, MaxDepth: 2, MaxEvaluations: 4}
				result, err := bookkeepingSearch(ctx, strategy, budget, state, rules, eval)
				if result == nil || result.Winner == nil || calls < 2 {
					t.Fatalf("result=%+v calls=%d error=%v", result, calls, err)
				}
				defer result.Close()
				defer release()
				if mode == "canceled" && !errors.Is(err, ErrSearchCancelled) || mode != "canceled" && err != nil {
					t.Fatal(err)
				}
				if mode != "normal" && (len(result.Rejected) == 0 || !errors.Is(result.Rejected[0].Err, failure)) {
					t.Fatal("lost rejected candidate identity")
				}
				if result.bookkeeping == nil || result.bookkeeping.Bytes() <= int64(reflect.TypeFor[SearchResult]().Size()) {
					t.Fatal("returned collections have no retained owner")
				}
				if book == nil || searchMemoryBudget(callback).book != nil || book.scratch.Bytes() != 0 {
					t.Fatal("search scratch owner survived Search return")
				}
				held := result.bookkeeping.Bytes()
				for owner := result.clones; owner != nil; owner = owner.next {
					held += owner.memory.Bytes()
				}
				available := cloneAvailable(callback, limit)
				alias := *result
				result.Close()
				if after := cloneAvailable(callback, limit); after != available+held {
					t.Fatalf("owned bookkeeping not released: before=%d after=%d held=%d", available, after, held)
				}
				after := cloneAvailable(callback, limit)
				alias.Close()
				if cloneAvailable(callback, limit) != after {
					t.Fatal("result alias released storage twice")
				}
				release()
				if cloneAvailable(callback, limit) != after+137 {
					t.Fatal("Close released unrelated caller workspace")
				}
				if stateFingerprint(state) != before {
					t.Fatal("search mutated caller state")
				}
			})
		}
	}
}

func TestBookkeepingEarlyRefusalAndRetry(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			state, _ := beamContractState()
			calls := 0
			eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
				calls++
				return &CandidatePlan{Cost: 1, Validated: true}, nil
			})
			budget := SearchBudget{MaxStates: 1, MaxMemoryBytes: 1, CallbackMemoryBytes: 4096}
			result, err := bookkeepingSearch(t.Context(), strategy, budget, state, nil, eval)
			if result != nil || calls != 0 || !errors.Is(err, ErrMemoryBudgetExceeded) {
				t.Fatalf("early refusal result=%+v calls=%d error=%v", result, calls, err)
			}
			budget.MaxMemoryBytes = 64 << 10
			result, err = bookkeepingSearch(t.Context(), strategy, budget, state, nil, eval)
			if err != nil || result == nil || result.Winner == nil || calls != 1 {
				t.Fatalf("retry result=%+v calls=%d error=%v", result, calls, err)
			}
			result.Close()
		})
	}
}

func TestBookkeepingPublicMatchingLimit(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			g := egraph.New()
			for i := 0; i < 32; i++ {
				g.Add(egraph.NewLeaf(fmt.Sprint("leaf", i)))
			}
			state := egraph.NewSearchState(g, map[string]egraph.Id{"root": 0})
			candidate := &CandidatePlan{Cost: 1, Validated: true}
			var book *searchBookkeeping
			calls := 0
			eval := beamEvaluatorFunc(func(ctx context.Context, _ *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
				book = searchMemoryBudget(ctx).book
				calls++
				return candidate, nil
			})
			budget := SearchBudget{MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: 1, MaxStates: 3, MaxDepth: 1, MaxActions: 32}
			// Measure the actual no-rule prefix, including all policy containers
			// and empty enumeration. This is a bounded logical-byte experiment.
			baseline, err := bookkeepingSearch(t.Context(), strategy, budget, state, nil, eval)
			if err != nil || baseline == nil || calls != 1 || baseline.AccountedBytes <= 0 {
				t.Fatal(baseline, calls, err)
			}
			budget.MaxMemoryBytes = baseline.AccountedBytes
			baseline.Close()
			calls = 0
			rules := []*egraph.Rewrite{{Name: "wide", LHS: egraph.Var("x"), RHS: egraph.Pat("replacement")}}
			result, err := bookkeepingSearch(t.Context(), strategy, budget, state, rules, eval)
			if result != nil {
				defer result.Close()
			}
			if !errors.Is(err, ErrMemoryBudgetExceeded) || result == nil || result.Winner != candidate || calls != 1 || result.ActionsAttempted != 0 {
				t.Fatalf("matching did not stop before transition: result=%+v calls=%d error=%v", result, calls, err)
			}
			if book == nil || book.scratch.Err() != nil || book.result.Err() != nil {
				t.Fatal("fixture exhausted policy containers instead of matching")
			}
		})
	}
}

func TestBookkeepingOwnerAdmissionUnwind(t *testing.T) {
	scopeBytes := int64(reflect.TypeFor[egraphcore.MemoryScope]().Size())
	bookBytes := int64(reflect.TypeFor[searchBookkeeping]().Size())
	resultBytes := int64(reflect.TypeFor[SearchResult]().Size())
	for _, limit := range []int64{1, scopeBytes, scopeBytes + bookBytes, 2*scopeBytes + bookBytes + resultBytes - 1, 4096} {
		b := &memoryBudget{limit: limit}
		ctx := context.WithValue(t.Context(), memoryBudgetKey{}, b)
		book, err := newSearchBookkeeping(ctx)
		if limit == 4096 {
			if err != nil || book == nil {
				t.Fatal(book, err)
			}
			book.close()
		} else if book != nil || !errors.Is(err, ErrMemoryBudgetExceeded) {
			t.Fatal(book, err)
		}
		if b.live != 0 || b.book != nil {
			t.Fatalf("refused setup leaked storage at limit %d: live=%d", limit, b.live)
		}
	}
}

// Borrow already constructed candidates. The callback itself allocates no
// owned value; its positive one-byte envelope is intentionally sufficient.
// Logical ReserveSearchMemory probes allocate no requested payload.
func TestBookkeepingPoolRefusalKeepsNewBest(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			state, _ := beamContractState()
			rules := []*egraph.Rewrite{egraph.MustRewrite("a", "base", "A"), egraph.MustRewrite("b", "base", "B")}
			candidates := [3]CandidatePlan{{Cost: 3, Validated: true}, {Cost: 2, Validated: true}, {Cost: 1, Validated: true}}
			var workspace func()
			calls := 0
			const limit = 1 << 20
			eval := beamEvaluatorFunc(func(ctx context.Context, _ *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
				candidate := &candidates[calls]
				calls++
				if calls == len(candidates) {
					bytes, err := storageBytes(candidate)
					if err != nil {
						return nil, err
					}
					workspace, err = ReserveSearchMemory(ctx, cloneAvailable(ctx, limit)-bytes)
					if err != nil {
						return nil, err
					}
				}
				return candidate, nil
			})
			budget := SearchBudget{MaxMemoryBytes: limit, CallbackMemoryBytes: 1, MaxEvaluations: 3, MaxStates: 5, MaxDepth: 1}
			result, err := bookkeepingSearch(t.Context(), strategy, budget, state, rules, eval)
			if workspace != nil {
				defer workspace()
			}
			if result != nil {
				defer result.Close()
			}
			if !errors.Is(err, ErrMemoryBudgetExceeded) || result == nil || calls != 3 || result.Winner != &candidates[2] || len(result.AllCandidates) != 2 {
				t.Fatalf("lost best or prior pool after guarded growth: calls=%d result=%+v error=%v", calls, result, err)
			}
		})
	}
}

func TestBookkeepingFingerprintAdmission(t *testing.T) {
	state, _ := beamContractState()
	state.Roots["a"] = state.Roots["root"]
	state.Derivation = []egraph.Action{{RuleName: strings.Repeat("r:", 256), RuleID: "id", Source: "s", RHSNode: "x", StepIndex: -1, Subst: egraph.Subst{"z": 2, "a": 1}}}
	// Independent old serializer, including separators and lexical key order.
	want := fmt.Sprintf("2;1:a%d;4:root%d;1;%d:%s2:id1:s1:x0;-1;2;1:a1;1:z2;", state.Roots["a"], state.Roots["root"], len(state.Derivation[0].RuleName), state.Derivation[0].RuleName)
	for _, limit := range []int64{128, 4096} {
		b := &memoryBudget{limit: limit}
		scope, err := egraphcore.NewMemoryScope(b)
		if err != nil {
			t.Fatal(err)
		}
		got, err := stateFingerprintWithMemory(state, scope)
		if limit == 128 {
			if got != "" || !errors.Is(err, ErrMemoryBudgetExceeded) {
				t.Fatalf("fingerprint allocated before byte admission: got=%q error=%v", got, err)
			}
		} else if err != nil || got != want {
			t.Fatalf("fingerprint changed: got=%q want=%q error=%v", got, want, err)
		}
		scope.Close()
		if b.live != 0 {
			t.Fatal("fingerprint scratch leaked")
		}
	}
}

type bookkeepingOpaqueError struct{ work func() }

func (bookkeepingOpaqueError) Error() string { panic("sizing invoked Error") }

func TestBookkeepingRejectionAdmission(t *testing.T) {
	state, _ := beamContractState()
	state.Derivation = []egraph.Action{{Subst: egraph.Subst{"x": 1}, Binding: egraphcore.Binding[egraph.Id]{"x": egraphcore.Class(egraph.Id(1))}}}
	failure := errors.New("original candidate failure")
	eval := &ExecutableEvaluator{Comparison: map[string]OutputComparison{"out": {Elements: 4}}, Receipts: []cudatrial.Receipt{{Timings: map[cudatrial.TimingKind]time.Duration{cudatrial.TimingGPUBusy: 1}, Provenance: map[string]string{"id": "original"}}}}
	for _, limit := range []int64{128, 4096} {
		b := &memoryBudget{limit: limit}
		scope, err := egraphcore.NewMemoryScope(b)
		if err != nil {
			t.Fatal(err)
		}
		r, err := rejectionWithMemory(scope, state, eval, failure)
		if limit == 128 {
			if r.Evidence != nil || r.Derivation != nil || !errors.Is(err, ErrMemoryBudgetExceeded) {
				t.Fatalf("rejection copied before admission: %+v %v", r, err)
			}
		} else {
			if err != nil || !errors.Is(r.Err, failure) || r.Evidence == nil || len(r.Derivation) != 1 {
				t.Fatal(r, err)
			}
			r.Derivation[0].Subst["x"] = 9
			r.Evidence.Trials[0].Provenance["id"] = "changed"
			if state.Derivation[0].Subst["x"] != 1 || eval.Receipts[0].Provenance["id"] != "original" {
				t.Fatal("rejection storage aliases copied maps")
			}
		}
		scope.Close()
		if b.live != 0 {
			t.Fatal("rejection charge leaked")
		}
	}
	b := &memoryBudget{limit: 4096}
	scope, err := egraphcore.NewMemoryScope(b)
	if err != nil {
		t.Fatal(err)
	}
	r, err := rejectionWithMemory(scope, state, eval, bookkeepingOpaqueError{work: func() {}})
	if r.Err != nil || !errors.Is(err, ErrMemoryAccountingUnavailable) {
		t.Fatal("unknown error storage did not refuse")
	}
	scope.Close()
	if b.live != 0 {
		t.Fatal("unknown error refusal leaked storage")
	}
}

func TestBookkeepingSelectionOrder(t *testing.T) {
	actions := []egraph.ActionCandidate{{Rule: &egraph.Rewrite{Name: "a"}}, {Rule: &egraph.Rewrite{Name: "b"}}, {Rule: &egraph.Rewrite{Name: "c"}}}
	for _, profile := range []profileActions{{}, {ranks: map[string]int{"a": 0, "b": 0}}, {weights: map[string]float64{"b": 2, "c": 2}}} {
		a, b := rand.New(rand.NewSource(42)), rand.New(rand.NewSource(42))
		for i := 0; i < 64; i++ {
			if chooseSearchAction(profile, actions, a) != profile.choose(actions, b) {
				t.Fatal("seeded optional-prior order changed")
			}
		}
	}
	paths := [][]egraph.Action{nil, {{RuleName: "r", Subst: egraph.Subst{"": 1, "z": 2}}}, {{RuleName: "r", Subst: egraph.Subst{"": 1, "a": 3}}}, {{RuleName: "s"}}, {{RuleName: "r", StepIndex: 1}}}
	for _, a := range paths {
		for _, b := range paths {
			if compareSearchDerivations(a, b) != compareDerivations(a, b) {
				t.Fatal("replay tie order changed")
			}
		}
	}
}
