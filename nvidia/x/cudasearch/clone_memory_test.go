package cudasearch

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func cloneCallbackBytes() int64 { return int64(reflect.TypeFor[CandidatePlan]().Size()) }

// Admit the actual initial copy, baseline storage and live action owner, but
// stop at the next clone's SearchState reservation. No transition runs here.
func cloneLaterRefusalBytes(t *testing.T, state *egraph.SearchState, rules []*egraph.Rewrite, candidate *CandidatePlan) int64 {
	t.Helper()
	b := &memoryBudget{limit: 1 << 20}
	ctx := context.WithValue(context.Background(), memoryBudgetKey{}, b)
	owner, err := newStateClone(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer owner.close()
	copy, err := state.CloneWithBudget(ctx, owner.work)
	if err != nil {
		t.Fatal(err)
	}
	copyBytes := owner.memory.Bytes()
	actions, err := egraph.EnumerateActionsWithMemory(copy.Graph, rules, 50, b)
	if err != nil {
		t.Fatal(err)
	}
	defer actions.Close()
	if len(actions.Actions) == 0 {
		t.Fatal("no later clone")
	}
	matchingBytes := b.live - copyBytes
	if matchingBytes <= 0 {
		t.Fatal("matching storage is not reserved")
	}
	candidateBytes, err := storageBytes(candidate)
	if err != nil {
		t.Fatal(err)
	}
	prefix := int64(reflect.TypeFor[egraphcore.MemoryScope]().Size() + reflect.TypeFor[stateClone]().Size() + reflect.TypeFor[egraphcore.WorkBudget]().Size())
	return estimateStateMemory(state) + copyBytes + candidateBytes + matchingBytes + prefix + int64(reflect.TypeFor[egraph.SearchState]().Size()) - 1
}

func cloneRolloutRefusalBytes(t *testing.T, state *egraph.SearchState, rules []*egraph.Rewrite, actionLimit int) int64 {
	t.Helper()
	b := &memoryBudget{limit: 1 << 20}
	ctx := context.WithValue(context.Background(), memoryBudgetKey{}, b)
	defer func() { closeStateClones(b.takeClones()) }()
	root, err := cloneSearchState(ctx, state)
	if err != nil {
		t.Fatal(err)
	}
	beforeInitial := b.live
	initial, err := egraph.EnumerateActionsWithMemory(root.Graph, rules, actionLimit, b)
	if err != nil {
		t.Fatal(err)
	}
	defer initial.Close()
	initialBytes := b.live - beforeInitial
	if initialBytes <= 0 {
		t.Fatal("initial matching storage is not reserved")
	}
	next, _, err := applySearchAction(ctx, root, initial.Actions[0])
	if err != nil || next.NoOp {
		t.Fatal(next, err)
	}
	before := b.live
	child, err := egraph.EnumerateActionsWithMemory(next.State.Graph, rules, actionLimit, b)
	if err != nil {
		t.Fatal(err)
	}
	defer child.Close()
	childBytes := b.live - before
	if childBytes <= 0 {
		t.Fatal("child matching storage is not reserved")
	}
	candidateBytes, err := storageBytes(&CandidatePlan{Cost: 10, Validated: true})
	if err != nil {
		t.Fatal(err)
	}
	// Expansion and rollout each hold their own child action owner.
	prefix := int64(reflect.TypeFor[egraphcore.MemoryScope]().Size() + reflect.TypeFor[stateClone]().Size() + reflect.TypeFor[egraphcore.WorkBudget]().Size())
	return estimateStateMemory(state) + b.live + childBytes + 2*candidateBytes + prefix + int64(reflect.TypeFor[egraph.SearchState]().Size()) - 1
}

// Probe only logical reservations; this allocates no requested payload.
func cloneAvailable(ctx context.Context, limit int64) int64 {
	lo, hi := int64(0), limit
	for lo < hi {
		mid := lo + (hi-lo+1)/2
		release, err := ReserveSearchMemory(ctx, mid)
		if err != nil {
			hi = mid - 1
		} else {
			release()
			lo = mid
		}
	}
	return lo
}

func TestCloneMemoryPredecessorIntegration(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			g := egraph.New()
			root := g.Add(egraph.NewNode("base"))
			state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
			var callbackCtx context.Context
			var evaluated *egraph.SearchState
			eval := beamEvaluatorFunc(func(ctx context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
				callbackCtx, evaluated = ctx, s
				return &CandidatePlan{Cost: 1, Validated: true}, nil
			})
			const limit = 64 << 10
			budget := SearchBudget{MaxMemoryBytes: limit, CallbackMemoryBytes: 4096, MaxStates: 1}
			result, err := memoryStrategy(strategy, budget, state, nil, eval)
			if err != nil || result == nil {
				t.Fatal(result, err)
			}
			defer result.Close()
			if evaluated == state {
				t.Fatal("bounded entry evaluated caller state before clone admission")
			}
			before := cloneAvailable(callbackCtx, limit)
			result.Close()
			if after := cloneAvailable(callbackCtx, limit); after <= before {
				t.Fatalf("clone has no returned owner: before=%d after=%d", before, after)
			}
		})
	}
}

func TestCloneMemoryPublicSearchOwners(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		for _, ending := range []string{"normal", "rejected", "cancel", "initial-error", "noop"} {
			t.Run(strategy+"/"+ending, func(t *testing.T) {
				g := egraph.New()
				leaf := g.Add(egraph.NewNode("leaf"))
				root := g.Add(egraph.NewNode("base", leaf))
				state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
				rules := []*egraph.Rewrite{egraph.MustRewrite("alternative", "(base ?x)", "(alternative ?x)"), egraph.MustRewrite("deeper", "(alternative ?x)", "(deeper ?x)")}
				if ending == "noop" {
					rules = []*egraph.Rewrite{egraph.MustRewrite("same", "(base ?x)", "(base ?x)")}
				}
				before := state.Clone()
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				var callbackCtx context.Context
				var releaseWorkspace func()
				var borrowed []*egraph.SearchState
				calls := 0
				eval := beamEvaluatorFunc(func(c context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
					calls++
					callbackCtx = c
					borrowed = append(borrowed, s)
					if calls == 1 {
						var err error
						releaseWorkspace, err = ReserveSearchMemory(c, 257)
						if err != nil {
							return nil, err
						}
					}
					if ending == "initial-error" || calls > 1 && ending == "rejected" {
						return nil, errors.New("fixture rejection")
					}
					if calls > 1 && ending == "cancel" {
						cancel()
						return nil, context.Canceled
					}
					return &CandidatePlan{Cost: float64(10 - calls), Validated: true, Derivation: s.Derivation}, nil
				})
				const limit = 1 << 20
				budget := SearchBudget{MaxMemoryBytes: limit, CallbackMemoryBytes: 4096, MaxStates: 4, MaxDepth: 2, MaxEvaluations: 4}
				var result *SearchResult
				var err error
				if strategy == "beam" {
					result, err = NewBeamSearch(BeamConfig{Budget: budget}).Search(ctx, state, rules, eval)
				} else {
					result, err = NewMCTSSearch(MCTSConfig{Budget: budget, Iterations: 3, RolloutDepth: 1}).Search(ctx, state, rules, eval)
				}
				if result == nil || callbackCtx == nil || len(borrowed) == 0 {
					t.Fatalf("missing result/consumer: %v", err)
				}
				if borrowed[0] == state {
					t.Fatal("bounded evaluator borrowed caller state")
				}
				if ending == "normal" && (err != nil || calls < 3 || result.Winner == nil || len(result.Winner.Derivation) != 2 || len(result.Winner.Derivation[0].Binding) == 0) {
					t.Fatalf("calls=%d result=%+v err=%v", calls, result, err)
				}
				if ending == "rejected" && len(result.Rejected) == 0 {
					t.Fatal("missing rejection")
				}
				if ending == "rejected" && len(result.Rejected[0].Derivation[0].Binding) == 0 {
					t.Fatal("missing retained binding alias")
				}
				if ending == "noop" && (calls != 1 || result.ActionsAttempted == 0 || err != nil) {
					t.Fatal("no-op admission", calls, err)
				}
				if ending == "initial-error" && (err == nil || len(result.Rejected) != 1) {
					t.Fatal("missing initial failure", err)
				}
				if ending == "cancel" && !errors.Is(err, ErrSearchCancelled) {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(state.Clone(), before) {
					t.Fatal("source changed")
				}
				var ownedBytes int64
				for _, state := range borrowed {
					found := false
					for owner := result.clones; owner != nil; owner = owner.next {
						if owner.state == state && !owner.closed {
							found = true
						}
					}
					if !found {
						t.Fatal("evaluated branch has no result owner")
					}
				}
				for owner := result.clones; owner != nil; owner = owner.next {
					ownedBytes += owner.memory.Bytes()
				}
				available := cloneAvailable(callbackCtx, limit)
				bookkeepingBytes := result.bookkeeping.Bytes()
				alias := *result
				result.Close()
				after := cloneAvailable(callbackCtx, limit)
				if ownedBytes <= 0 || bookkeepingBytes <= 0 || after-available != ownedBytes+bookkeepingBytes {
					t.Fatalf("owned charges not released: before=%d after=%d clones=%d bookkeeping=%d", available, after, ownedBytes, bookkeepingBytes)
				}
				if ending == "noop" {
					// Initial-state accounting ends with Search, candidate
					// accounting with result.Close. Only this caller's explicit
					// workspace reservation may remain live.
					wantLive := int64(257)
					if limit-after != wantLive {
						t.Fatalf("no-op left unowned charge: live=%d want=%d", limit-after, wantLive)
					}
				}
				alias.Close()
				if cloneAvailable(callbackCtx, limit) != after {
					t.Fatal("copied result released twice")
				}
				releaseWorkspace()
				if cloneAvailable(callbackCtx, limit) != after+257 {
					t.Fatal("result closed unrelated callback owner")
				}
			})
		}
	}
}

func TestCloneMemoryOwnerAdmission(t *testing.T) {
	t.Run("bookkeeping", func(t *testing.T) {
		b := &memoryBudget{limit: 1 << 20}
		ctx := context.WithValue(context.Background(), memoryBudgetKey{}, b)
		owner, err := newStateClone(ctx)
		if err != nil {
			t.Fatal(err)
		}
		want := int64(reflect.TypeFor[egraphcore.MemoryScope]().Size() + reflect.TypeFor[stateClone]().Size() + reflect.TypeFor[egraphcore.WorkBudget]().Size())
		if b.live != want || owner.memory.Bytes() != want {
			t.Fatal("unreserved clone owner", b.live, want)
		}
		owner.close()
		if b.live != 0 {
			t.Fatal("owner leaked", b.live)
		}
		b.limit = want - 1
		if owner, err := newStateClone(ctx); owner != nil || !errors.Is(err, ErrMemoryBudgetExceeded) || b.live != 0 {
			t.Fatal("owner allocated before admission", owner, err, b.live)
		}
	})
	state, _ := beamContractState()
	for _, kind := range []string{"clone", "invalid", "noop", "productive"} {
		t.Run(kind, func(t *testing.T) {
			b := &memoryBudget{limit: 64 << 10}
			ctx := context.WithValue(context.Background(), memoryBudgetKey{}, b)
			if kind == "clone" {
				b.limit = 1
				copy, err := cloneSearchState(ctx, state)
				if copy != nil || !errors.Is(err, ErrMemoryBudgetExceeded) || b.live != 0 || b.clones != nil {
					t.Fatal(copy, err, b.live)
				}
				return
			}
			action := egraph.ActionCandidate{}
			if kind != "invalid" {
				rhs := "base"
				if kind == "productive" {
					rhs = "alternative"
				}
				action = egraph.EnumerateActionsBounded(state.Graph, []*egraph.Rewrite{egraph.MustRewrite("change", "base", rhs)}, 1)[0]
			}
			result, owner, err := applySearchAction(ctx, state, action)
			if kind == "invalid" {
				if err == nil || err.Error() != "egraph: nil rewrite rule in candidate" || result != nil || owner == nil || owner.closed || owner.state != nil || b.live <= 0 || owner.memory.Bytes() != b.live || b.clones != nil {
					t.Fatal(result, owner, err, b.live)
				}
				owner.close()
				owner.close()
				if b.live != 0 {
					t.Fatal("leaked", b.live)
				}
				return
			}
			if err != nil || result == nil || owner == nil || b.live <= 0 {
				t.Fatal(result, owner, err, b.live)
			}
			if kind == "noop" {
				if !result.NoOp || b.clones != nil {
					t.Fatal("no-op retained")
				}
				owner.close()
			} else {
				if result.NoOp || b.clones != owner {
					t.Fatal("productive owner missing")
				}
				closeStateClones(b.takeClones())
			}
			if b.live != 0 {
				t.Fatal("leaked", b.live)
			}
		})
	}
}

func TestCloneMemoryBeamPrunedOwners(t *testing.T) {
	g := egraph.New()
	leaf := g.Add(egraph.NewNode("leaf"))
	root := g.Add(egraph.NewNode("base", leaf))
	source := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	rules := []*egraph.Rewrite{egraph.MustRewrite("one", "(base ?x)", "(one ?x)"), egraph.MustRewrite("two", "(base ?x)", "(two ?x)")}
	var callbackCtx context.Context
	calls := 0
	eval := beamEvaluatorFunc(func(ctx context.Context, state *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
		callbackCtx = ctx
		calls++
		return &CandidatePlan{Cost: float64(10 - calls), Validated: true, Derivation: state.Derivation}, nil
	})
	const limit = 1 << 20
	result, err := NewBeamSearch(BeamConfig{BeamWidth: 1, Budget: SearchBudget{MaxMemoryBytes: limit, CallbackMemoryBytes: 4096, MaxDepth: 1, MaxStates: 4}}).Search(context.Background(), source, rules, eval)
	if err != nil || result == nil || calls != 3 || len(result.AllCandidates) != 3 {
		t.Fatal(result, calls, err)
	}
	var bytes int64
	count := 0
	for owner := result.clones; owner != nil; owner = owner.next {
		bytes += owner.memory.Bytes()
		count++
	}
	if count != 3 {
		t.Fatal("pruned state lost owner", count)
	}
	losers := 0
	for _, candidate := range result.AllCandidates {
		if candidate != result.Winner && len(candidate.Derivation) != 0 {
			losers++
			if len(candidate.Derivation[0].Binding) != 1 {
				t.Fatal("pruned candidate lost binding")
			}
		}
	}
	if losers != 1 {
		t.Fatal("no retained pruned candidate", losers)
	}
	before := cloneAvailable(callbackCtx, limit)
	bookkeepingBytes := result.bookkeeping.Bytes()
	result.Close()
	if bytes <= 0 || bookkeepingBytes <= 0 || cloneAvailable(callbackCtx, limit)-before != bytes+bookkeepingBytes {
		t.Fatal("pruned owner charge not released")
	}
}
