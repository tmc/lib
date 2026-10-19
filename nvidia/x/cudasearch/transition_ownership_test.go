package cudasearch

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestTransitionDiagnosticOwner(t *testing.T) {
	t.Run("stale-match", func(t *testing.T) {
		state, _ := beamContractState()
		ledger := &memoryBudget{limit: 1 << 20}
		ctx := context.WithValue(context.Background(), memoryBudgetKey{}, ledger)
		stale := egraph.ActionCandidate{
			Rule:  egraph.MustRewrite("stale", "absent", "other"),
			Match: egraph.MatchMatch{ClassID: state.Roots["root"]},
		}
		for i := 0; i < 3; i++ {
			result, owner, err := applySearchAction(ctx, state, stale)
			if result != nil || owner == nil || owner.closed || owner.state != nil || err == nil || ledger.live == 0 || owner.memory.Bytes() != ledger.live {
				t.Fatalf("iter %d: error lost its live scope: result=%v owner=%+v err=%v live=%d", i, result, owner, err, ledger.live)
			}
			if errors.Is(err, ErrMemoryBudgetExceeded) || errors.Is(err, ErrMemoryAccountingUnavailable) || err.Error() != "egraph: candidate match is stale or forged" {
				t.Fatalf("iter %d: error classification or message changed: %v", i, err)
			}
			if ledger.clones != nil {
				t.Fatalf("iter %d: ordinary failure was linked into retained clones", i)
			}
			owner.close()
			owner.close()
			if ledger.live != 0 {
				t.Fatalf("iter %d: closed error owner leaked %d bytes", i, ledger.live)
			}
		}
	})

	t.Run("mandatory-precondition-diagnostic", func(t *testing.T) {
		state, rule := mandatoryOperandDiagnosticState()
		ledger := &memoryBudget{limit: 1 << 20}
		ctx := context.WithValue(context.Background(), memoryBudgetKey{}, ledger)
		actions := egraph.EnumerateActionsBounded(state.Graph, []*egraph.Rewrite{rule}, 5)
		if len(actions) != 1 {
			t.Fatalf("expected 1 candidate action, got %d", len(actions))
		}
		for i := 0; i < 3; i++ {
			result, owner, err := applySearchAction(ctx, state, actions[0])
			if result != nil || owner == nil || owner.closed || owner.state != nil || err == nil || ledger.live == 0 {
				t.Fatalf("iter %d: missing live owner on mandatory diagnostic: result=%v owner=%+v err=%v live=%d", i, result, owner, err, ledger.live)
			}
			if !errors.Is(err, egraph.ErrMissingTypedMetadata) || errors.Is(err, ErrMemoryBudgetExceeded) || errors.Is(err, ErrMemoryAccountingUnavailable) {
				t.Fatalf("iter %d: unexpected error classification: %v", i, err)
			}
			if ledger.clones != nil {
				t.Fatalf("iter %d: ordinary mandatory rejection linked into retained clones", i)
			}
			owner.close()
			if ledger.live != 0 {
				t.Fatalf("iter %d: mandatory diagnostic owner leaked %d bytes", i, ledger.live)
			}
		}
	})

	t.Run("accounting-unavailable-diagnostic", func(t *testing.T) {
		state, _ := beamContractState()
		ledger := &memoryBudget{limit: 1 << 20}
		ctx := context.WithValue(context.Background(), memoryBudgetKey{}, ledger)
		unsupported := &egraph.Rewrite{
			Name: "unsupported-rhs-data",
			LHS:  egraph.Pat("base"),
			RHS:  &egraph.Pattern{Op: "next", HasData: true, Data: []byte{1}},
		}
		actions := egraph.EnumerateActionsBounded(state.Graph, []*egraph.Rewrite{unsupported}, 5)
		if len(actions) != 1 {
			t.Fatalf("expected 1 action, got %d", len(actions))
		}
		result, owner, err := applySearchAction(ctx, state, actions[0])
		if result != nil || owner == nil || owner.closed || owner.state != nil || err == nil || ledger.live == 0 {
			t.Fatalf("missing live owner on accounting unavailable error: result=%v owner=%+v err=%v live=%d", result, owner, err, ledger.live)
		}
		if !errors.Is(err, ErrMemoryAccountingUnavailable) || !errors.Is(err, egraph.ErrMemoryOperation) {
			t.Fatalf("lost ErrMemoryAccountingUnavailable / ErrMemoryOperation identity: %v", err)
		}
		if ledger.clones != nil {
			t.Fatal("applySearchAction linked failed owner before caller inspection")
		}
		owner.close()
		if ledger.live != 0 {
			t.Fatalf("accounting unavailable owner leaked %d bytes", ledger.live)
		}
	})

	t.Run("nil-owner-safety", func(t *testing.T) {
		state, _ := beamContractState()
		stale := egraph.ActionCandidate{
			Rule:  egraph.MustRewrite("stale", "absent", "other"),
			Match: egraph.MatchMatch{ClassID: state.Roots["root"]},
		}
		if res, owner, err := applySearchAction(context.Background(), state, stale); res != nil || owner != nil || err == nil {
			t.Fatalf("unbudgeted applySearchAction: res=%v owner=%v err=%v", res, owner, err)
		}
		tiny := &memoryBudget{limit: 1}
		ctx := context.WithValue(context.Background(), memoryBudgetKey{}, tiny)
		if res, owner, err := applySearchAction(ctx, state, stale); res != nil || owner != nil || !errors.Is(err, ErrMemoryBudgetExceeded) || tiny.live != 0 {
			t.Fatalf("newStateClone refusal: res=%v owner=%v err=%v live=%d", res, owner, err, tiny.live)
		}
	})
}

func mandatoryOperandDiagnosticState() (*egraph.SearchState, *egraph.Rewrite) {
	g := egraph.New()
	x := g.Add(egraph.NewNode("x"))
	w := g.Add(egraph.NewNode("w"))
	width := g.Add(egraph.NewNode("64"))
	root := g.Add(egraph.NewNode("prefill_group_gemm", x, w, width))
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	shape, err := cudashape.New(16, 256)
	if err != nil {
		panic(err)
	}
	state.SetMetadata(root, egraph.TypedMetadata{
		DType:           cudair.Float16,
		Shape:           shape,
		Layout:          cudashape.LayoutRowMajor,
		Effect:          cudair.EffectReadOnly,
		LoopInvariant:   true,
		NumericalPolicy: "fp16",
	})
	var rule *egraph.Rewrite
	for _, r := range egraph.MandatoryRewriteRules() {
		if r.Name == "prefill_group_expand_w256" {
			rule = r
			break
		}
	}
	return state, rule
}

func wideRewriteRule(op, lhs string, arity int) *egraph.Rewrite {
	children := make([]*egraph.Pattern, arity)
	for i := range children {
		children[i] = egraph.Pat(lhs)
	}
	return &egraph.Rewrite{Name: op, LHS: egraph.Pat(lhs), RHS: egraph.Pat(op, children...)}
}

func runTransitionPathSearch(path string, budget SearchBudget, state *egraph.SearchState, rules []*egraph.Rewrite, eval Evaluator) (*SearchResult, error) {
	if path == "beam-expansion" {
		return NewBeamSearch(BeamConfig{Budget: budget}).Search(context.Background(), state, rules, eval)
	}
	return NewMCTSSearch(MCTSConfig{Budget: budget, Iterations: 2, RolloutDepth: 2}).Search(context.Background(), state, rules, eval)
}

func TestTransitionStopOwnerRetentionPublicSearch(t *testing.T) {
	for _, path := range []string{"beam-expansion", "mcts-expansion", "mcts-rollout"} {
		for _, mode := range []string{"budget-exceeded", "post-return-reserve-exceeded", "accounting-unavailable"} {
			t.Run(path+"/"+mode, func(t *testing.T) {
				state, _ := beamContractState()
				var rules []*egraph.Rewrite
				wantCalls := 1
				wantActions := 1
				wantStates := 2
				limit := int64(64 << 10)

				targetLHS := "base"
				if path == "mcts-rollout" {
					rules = append(rules, egraph.MustRewrite("expand", "base", "child"))
					targetLHS = "child"
					wantCalls = 2
					// In rollout on the merged {base, child} class, MCTS with seed 42
					// first draws the now-no-op "expand" action (closing its owner)
					// and then attempts the rollout target action.
					wantActions = 3
					wantStates = 4
				}

				var wantErr error
				switch mode {
				case "budget-exceeded":
					rules = append(rules, wideRewriteRule("wide", targetLHS, 128))
					wantErr = ErrMemoryBudgetExceeded
				case "post-return-reserve-exceeded":
					rules = append(rules, egraph.MustRewrite("step", targetLHS, "final"))
					wantErr = ErrMemoryBudgetExceeded
				case "accounting-unavailable":
					rules = append(rules, &egraph.Rewrite{
						Name: "unsupported",
						LHS:  egraph.Pat(targetLHS),
						RHS:  &egraph.Pattern{Op: "next", HasData: true, Data: []byte{1}},
					})
					wantErr = ErrMemoryAccountingUnavailable
				}

				baseline := &CandidatePlan{ID: "incumbent", Cost: 10, Validated: true}
				improved := &CandidatePlan{ID: "expanded", Cost: 5, Validated: true}
				wantWinner := baseline
				if path == "mcts-rollout" {
					wantWinner = improved
				}

				var wantTransitionOnlyBytes int64
				if mode == "post-return-reserve-exceeded" {
					probeLedger := &memoryBudget{limit: 1 << 20}
					probeCtx := context.WithValue(context.Background(), memoryBudgetKey{}, probeLedger)
					probeState, err := cloneSearchState(probeCtx, state)
					if err != nil {
						t.Fatal(err)
					}
					if path == "mcts-rollout" {
						firstSet, err := enumerateSearchActions(probeCtx, probeState, []*egraph.Rewrite{rules[0]}, 1)
						if err != nil {
							t.Fatal(err)
						}
						res, _, err := applySearchAction(probeCtx, probeState, firstSet.Actions[0])
						firstSet.Close()
						if err != nil {
							t.Fatal(err)
						}
						probeState = res.State
					}
					targetSet, err := enumerateSearchActions(probeCtx, probeState, []*egraph.Rewrite{rules[len(rules)-1]}, 1)
					if err != nil {
						t.Fatal(err)
					}
					probeOwner, err := newStateClone(probeCtx)
					if err != nil {
						t.Fatal(err)
					}
					probeRes, err := probeState.ApplyActionWithCloneBudget(probeCtx, targetSet.Actions[0], probeOwner.work)
					targetSet.Close()
					if err != nil {
						t.Fatal(err)
					}
					wantTransitionOnlyBytes = probeOwner.memory.Bytes()
					postBytes, err := storageBytes(probeRes.State)
					if err != nil || postBytes <= 1 {
						t.Fatalf("postBytes=%d err=%v", postBytes, err)
					}
					probeOwner.close()
					closeStateClones(probeLedger.takeClones())

					// Binary-search the largest MaxMemoryBytes where the target
					// applySearchAction still refuses (retaining a failed owner
					// with state == nil), which is 1 byte below post-return
					// c.memory.Reserve(bytes) admission.
					lo, hi := int64(1), int64(64<<10)
					for lo < hi {
						mid := lo + (hi-lo+1)/2
						probeCalls := 0
						probeEval := beamEvaluatorFunc(func(_ context.Context, _ *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
							probeCalls++
							if probeCalls == 2 {
								return improved, nil
							}
							return baseline, nil
						})
						probeBudget := SearchBudget{
							MaxMemoryBytes:      mid,
							CallbackMemoryBytes: cloneCallbackBytes(),
							MaxStates:           5,
							MaxDepth:            3,
							MaxActions:          5,
							MaxEvaluations:      5,
						}
						probeResult, probeErr := runTransitionPathSearch(path, probeBudget, state, rules, probeEval)
						nonNilClones := 0
						if probeResult != nil {
							for owner := probeResult.clones; owner != nil; owner = owner.next {
								if owner.state != nil {
									nonNilClones++
								}
							}
							probeResult.Close()
						}
						if errors.Is(probeErr, ErrMemoryBudgetExceeded) && probeCalls <= wantCalls && nonNilClones <= wantCalls {
							lo = mid
						} else {
							hi = mid - 1
						}
					}
					limit = lo
				}

				calls := 0
				var callbackCtx context.Context
				eval := beamEvaluatorFunc(func(ctx context.Context, _ *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
					calls++
					callbackCtx = ctx
					if calls == 2 {
						return improved, nil
					}
					return baseline, nil
				})

				budget := SearchBudget{
					MaxMemoryBytes:      limit,
					CallbackMemoryBytes: cloneCallbackBytes(),
					MaxStates:           5,
					MaxDepth:            3,
					MaxActions:          5,
					MaxEvaluations:      5,
				}
				result, err := runTransitionPathSearch(path, budget, state, rules, eval)
				if result != nil {
					defer result.Close()
				}
				if !errors.Is(err, wantErr) || result == nil || !errors.Is(result.StopReason, wantErr) {
					t.Fatalf("expected error %v: got err=%v result=%+v", wantErr, err, result)
				}
				if mode == "accounting-unavailable" && (!errors.Is(err, egraph.ErrMemoryOperation) || !errors.Is(result.StopReason, egraph.ErrMemoryOperation)) {
					t.Fatalf("lost underlying ErrMemoryOperation identity: err=%v stop=%v", err, result.StopReason)
				}
				if result.Winner != wantWinner || calls != wantCalls || result.ActionsAttempted != wantActions || result.StatesVisited != wantStates {
					t.Fatalf("unexpected search state on refusal: winner=%+v calls=%d actions=%d states=%d", result.Winner, calls, result.ActionsAttempted, result.StatesVisited)
				}

				var failedOwner *stateClone
				failedOwners := 0
				var totalCloneBytes int64
				for owner := result.clones; owner != nil; owner = owner.next {
					if owner.closed {
						t.Fatal("retained owner closed before SearchResult.Close")
					}
					totalCloneBytes += owner.memory.Bytes()
					if owner.state == nil && owner.memory.Bytes() > 0 {
						failedOwner = owner
						failedOwners++
					}
				}
				if failedOwners != 1 || failedOwner == nil {
					t.Fatalf("failed transition scope not transferred to result: failedOwners=%d", failedOwners)
				}
				if mode == "post-return-reserve-exceeded" && failedOwner.memory.Bytes() != wantTransitionOnlyBytes {
					t.Fatalf("post-return refusal did not complete ApplyActionWithCloneBudget first: got=%d want=%d", failedOwner.memory.Bytes(), wantTransitionOnlyBytes)
				}
				if err.Error() == "" || result.StopReason.Error() == "" {
					t.Fatal("empty error message while owner is live")
				}

				beforeClose := cloneAvailable(callbackCtx, limit)
				bookkeepingBytes := result.bookkeeping.Bytes()
				alias := *result
				result.Close()
				afterClose := cloneAvailable(callbackCtx, limit)
				if !failedOwner.closed || failedOwner.memory.Bytes() != 0 {
					t.Fatalf("failed owner survived result.Close: closed=%v bytes=%d", failedOwner.closed, failedOwner.memory.Bytes())
				}
				if totalCloneBytes <= 0 || bookkeepingBytes <= 0 || afterClose-beforeClose != totalCloneBytes+bookkeepingBytes {
					t.Fatalf("result.Close did not release exact owned bytes: before=%d after=%d clones=%d bookkeeping=%d", beforeClose, afterClose, totalCloneBytes, bookkeepingBytes)
				}
				alias.Close()
				result.Close()
				if cloneAvailable(callbackCtx, limit) != afterClose {
					t.Fatal("idempotent Close released bytes twice")
				}
			})
		}
	}
}

func TestTransitionOrdinaryRejectionAndNoOpNoAccumulationPublicSearch(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			state, mandatoryRule := mandatoryOperandDiagnosticState()
			noopRule := egraph.MustRewrite("noop-x", "x", "x")
			productiveRule := egraph.MustRewrite("improve-w", "w", "w_fast")
			rolloutNoopRule := egraph.MustRewrite("noop-fast", "w_fast", "w_fast")
			rules := []*egraph.Rewrite{mandatoryRule, noopRule, productiveRule, rolloutNoopRule}

			const limit = 1 << 20
			var callbackCtx context.Context
			calls := 0
			eval := beamEvaluatorFunc(func(ctx context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
				calls++
				callbackCtx = ctx
				return &CandidatePlan{
					ID:         "plan",
					Cost:       float64(10 - len(s.Derivation)),
					Validated:  true,
					Derivation: s.Derivation,
				}, nil
			})

			budget := SearchBudget{
				MaxMemoryBytes:      limit,
				CallbackMemoryBytes: 4096,
				MaxStates:           12,
				MaxDepth:            3,
				MaxActions:          10,
				MaxEvaluations:      6,
			}
			var result *SearchResult
			var err error
			if strategy == "beam" {
				result, err = NewBeamSearch(BeamConfig{BeamWidth: 2, Budget: budget}).Search(context.Background(), state, rules, eval)
			} else {
				result, err = NewMCTSSearch(MCTSConfig{Iterations: 6, RolloutDepth: 2, Budget: budget}).Search(context.Background(), state, rules, eval)
			}
			if err != nil || result == nil || result.Winner == nil || calls < 2 || result.ActionsAttempted < 3 {
				t.Fatalf("result=%+v calls=%d err=%v", result, calls, err)
			}
			defer result.Close()

			failedOwners := 0
			var totalCloneBytes int64
			for owner := result.clones; owner != nil; owner = owner.next {
				if owner.closed {
					t.Fatal("retained productive owner already closed")
				}
				if owner.state == nil {
					failedOwners++
				}
				totalCloneBytes += owner.memory.Bytes()
			}
			if failedOwners != 0 {
				t.Fatalf("ordinary rejected or no-op transition owners accumulated in result.clones: %d", failedOwners)
			}

			beforeClose := cloneAvailable(callbackCtx, limit)
			bookkeepingBytes := result.bookkeeping.Bytes()
			result.Close()
			afterClose := cloneAvailable(callbackCtx, limit)
			if afterClose-beforeClose != totalCloneBytes+bookkeepingBytes {
				t.Fatalf("unexpected released bytes: before=%d after=%d clones=%d bookkeeping=%d", beforeClose, afterClose, totalCloneBytes, bookkeepingBytes)
			}
		})
	}
}

func TestTransitionNilOwnerRefusalPublicSearch(t *testing.T) {
	for _, path := range []string{"beam-expansion", "mcts-expansion", "mcts-rollout"} {
		t.Run(path, func(t *testing.T) {
			state, rules := beamContractState()
			wantActions := 1
			if path == "mcts-rollout" {
				rules = []*egraph.Rewrite{
					egraph.MustRewrite("expand", "base", "child"),
					egraph.MustRewrite("roll", "child", "leaf"),
				}
				wantActions = 2
			}
			// Find the smallest MaxMemoryBytes that reaches the target
			// applySearchAction call (ActionsAttempted >= wantActions), so that
			// enumerateSearchActions succeeds and newStateClone refuses before
			// returning a non-nil stateClone owner.
			lo, hi := int64(1), int64(64<<10)
			for lo < hi {
				mid := lo + (hi-lo)/2
				probeEval := beamEvaluatorFunc(func(_ context.Context, _ *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
					return &CandidatePlan{Cost: 10, Validated: true}, nil
				})
				probeBudget := SearchBudget{
					MaxMemoryBytes:      mid,
					CallbackMemoryBytes: cloneCallbackBytes(),
					MaxStates:           5,
					MaxDepth:            3,
					MaxActions:          5,
					MaxEvaluations:      5,
				}
				probeResult, _ := runTransitionPathSearch(path, probeBudget, state, rules, probeEval)
				attempts := 0
				if probeResult != nil {
					attempts = probeResult.ActionsAttempted
					probeResult.Close()
				}
				if attempts >= wantActions {
					hi = mid
				} else {
					lo = mid + 1
				}
			}
			limit := lo

			calls := 0
			var callbackCtx context.Context
			eval := beamEvaluatorFunc(func(ctx context.Context, _ *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
				calls++
				callbackCtx = ctx
				return &CandidatePlan{Cost: 10, Validated: true}, nil
			})
			budget := SearchBudget{
				MaxMemoryBytes:      limit,
				CallbackMemoryBytes: cloneCallbackBytes(),
				MaxStates:           5,
				MaxDepth:            3,
				MaxActions:          5,
				MaxEvaluations:      5,
			}
			result, err := runTransitionPathSearch(path, budget, state, rules, eval)
			if result != nil {
				defer result.Close()
			}
			if !errors.Is(err, ErrMemoryBudgetExceeded) || result == nil || result.Winner == nil || result.ActionsAttempted != wantActions {
				t.Fatalf("expected nil-owner transition refusal after enumeration: result=%+v calls=%d err=%v", result, calls, err)
			}
			for owner := result.clones; owner != nil; owner = owner.next {
				if owner.state == nil {
					t.Fatal("unexpected failed owner when newStateClone refused")
				}
			}
			beforeClose := cloneAvailable(callbackCtx, limit)
			held := result.bookkeeping.Bytes()
			for owner := result.clones; owner != nil; owner = owner.next {
				held += owner.memory.Bytes()
			}
			result.Close()
			if afterClose := cloneAvailable(callbackCtx, limit); afterClose-beforeClose != held {
				t.Fatalf("Close mismatch on nil-owner refusal: before=%d after=%d held=%d", beforeClose, afterClose, held)
			}
		})
	}
}
