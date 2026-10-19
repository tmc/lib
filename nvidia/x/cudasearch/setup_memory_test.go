package cudasearch

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func setupState() *egraph.SearchState {
	g := egraph.New()
	id := g.Add(egraph.NewLeaf("input"))
	return egraph.NewSearchState(g, map[string]egraph.Id{"root": id})
}

func setupHostOnly(t *testing.T) {
	t.Helper()
	if driver.Loaded() || nvrtc.Loaded() {
		t.Fatal("native driver/compiler loaded")
	}
	t.Cleanup(func() {
		if driver.Loaded() || nvrtc.Loaded() {
			t.Error("native driver/compiler loaded")
		}
	})
}

func setupMock() *MockEvaluator {
	return &MockEvaluator{Scoring: ScoringModel{EvidenceTime: time.Unix(1700000000, 0), Epoch: "setup-v1", BaseLatencyWeight: 1}}
}

func TestSetupScoringPublicBound(t *testing.T) {
	setupHostOnly(t)
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			budget := SearchBudget{MaxStates: 1, MaxEvaluations: 1, MaxMemoryBytes: 1 << 20}
			eval := setupMock()
			base, err := bookkeepingSearch(t.Context(), strategy, budget, setupState(), nil, eval)
			if err != nil || base == nil || base.Winner == nil {
				t.Fatal(base, err)
			}
			limit := base.AccountedBytes
			base.Close()
			eval.Evidence = make(map[string]*ProfileEvidence)
			for i := 0; i < 256; i++ {
				eval.Evidence[fmt.Sprint("unused-", i)] = &ProfileEvidence{Epoch: "stale-epoch", Occupancy: 0.5}
			}
			budget.MaxMemoryBytes = limit
			got, err := bookkeepingSearch(t.Context(), strategy, budget, setupState(), nil, eval)
			if got != nil {
				got.Close()
			}
			if got != nil || !errors.Is(err, ErrMemoryBudgetExceeded) || !strings.Contains(err.Error(), "scoring setup") {
				t.Fatalf("snapshot not refused before policy/evaluation: result=%+v err=%v", got, err)
			}
			budget.MaxMemoryBytes = 1 << 20
			got, err = bookkeepingSearch(t.Context(), strategy, budget, setupState(), nil, eval)
			if err != nil || got == nil || got.Winner == nil || got.Winner.Cost != 1000000 || got.AccountedBytes <= limit {
				t.Fatalf("supported snapshot/objective changed: %+v %v", got, err)
			}
			got.Close()
			if eval.Evidence["unused-0"].Occupancy != 0.5 {
				t.Fatal("borrowed evidence mutated")
			}
		})
	}
}

func TestSetupPolicyPublicBound(t *testing.T) {
	setupHostOnly(t)
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			calls := 0
			eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
				calls++
				return &CandidatePlan{Cost: 1, Validated: true}, nil
			})
			budget := SearchBudget{MaxStates: 1, MaxEvaluations: 1, MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: 1}
			base, err := bookkeepingSearch(t.Context(), strategy, budget, setupState(), nil, eval)
			if err != nil || base == nil {
				t.Fatal(base, err)
			}
			limit := base.AccountedBytes
			base.Close()
			rules := []*egraph.Rewrite{egraph.MustRewrite(strings.Repeat("identity", 128), "absent", "other")}
			budget.MaxMemoryBytes, calls = limit, 0
			got, err := bookkeepingSearch(t.Context(), strategy, budget, setupState(), rules, eval)
			if got != nil {
				got.Close()
			}
			if got != nil || !errors.Is(err, ErrMemoryBudgetExceeded) || !strings.Contains(err.Error(), "policy setup") || calls != 0 {
				t.Fatalf("identity preflight bypassed: result=%+v calls=%d err=%v", got, calls, err)
			}
			budget.MaxMemoryBytes = 1 << 20
			got, err = bookkeepingSearch(t.Context(), strategy, budget, setupState(), rules, eval)
			if err != nil || got == nil || got.Winner == nil || got.Winner.Cost != 1 || calls != 1 {
				t.Fatal(got, calls, err)
			}
			got.Close()
		})
	}
}

func TestSetupSearchLifetime(t *testing.T) {
	setupHostOnly(t)
	for _, strategy := range []string{"beam", "mcts"} {
		for _, mode := range []string{"success", "cancel", "reject"} {
			t.Run(strategy+"/"+mode, func(t *testing.T) {
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				var work *workBudget
				var release func()
				failure := errors.New("independent rejection")
				eval := beamEvaluatorFunc(func(ctx context.Context, _ *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
					work, _ = ctx.Value(workBudgetKey{}).(*workBudget)
					var err error
					release, err = ReserveSearchMemory(ctx, 128)
					if err != nil {
						return nil, err
					}
					if mode == "reject" {
						return nil, failure
					}
					if mode == "cancel" {
						cancel()
					}
					return &CandidatePlan{Cost: 1, Validated: true}, nil
				})
				budget := SearchBudget{MaxStates: 2, MaxEvaluations: 2, MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: 4096}
				state := setupState()
				result, err := bookkeepingSearch(ctx, strategy, budget, state, nil, eval)
				if result == nil || work == nil || release == nil {
					t.Fatal(result, work, err)
				}
				defer release()
				if mode == "reject" && !errors.Is(err, failure) || mode == "success" && err != nil || mode == "cancel" && !errors.Is(err, context.Canceled) {
					t.Fatal(mode, err)
				}
				if work.setup != nil || work.scoring != nil || work.bootstrap != 0 {
					t.Fatal("search scratch owner not closed")
				}
				if len(result.PolicyID.Options) == 0 || result.PolicyID.RuleSet == "" || result.bookkeeping == nil {
					t.Fatal("policy not retained with result")
				}
				held := result.bookkeeping.Bytes()
				for c := result.clones; c != nil; c = c.next {
					held += c.memory.Bytes()
				}
				before := work.memory.live
				alias := *result
				result.Close()
				if before-work.memory.live != held || result.PolicyID.Options != nil || result.PolicyID.RuleSet != "" {
					t.Fatalf("policy release mismatch: before=%d after=%d held=%d", before, work.memory.live, held)
				}
				after := work.memory.live
				alias.Close()
				if work.memory.live != after {
					t.Fatal("copied result released twice")
				}
				release()
				if after-work.memory.live != 128 || state.Graph == nil || len(state.Roots) != 1 {
					t.Fatal("Close released caller workspace or state")
				}
			})
		}
	}
}

func TestSetupScoringAdmissionAndCopy(t *testing.T) {
	setupHostOnly(t)
	eval := setupMock()
	eval.Evidence = map[string]*ProfileEvidence{"one": {Epoch: "setup-v1", Occupancy: 0.75}, "nil": nil}
	b := &memoryBudget{limit: 1 << 20}
	scope, err := egraphcore.NewMemoryScope(b)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := snapshotScoringWithMemory(eval, scope)
	if err != nil || snapshot == nil || snapshot.evidence["one"] == eval.Evidence["one"] {
		t.Fatal(snapshot, err)
	}
	want := int64(reflect.TypeFor[egraphcore.MemoryScope]().Size() + reflect.TypeFor[scoringSnapshot]().Size() + reflect.TypeFor[map[string]*ProfileEvidence]().Size() + reflect.TypeFor[struct {
		key   string
		value *ProfileEvidence
	}]().Size() + reflect.TypeFor[ProfileEvidence]().Size())
	if b.live != want {
		t.Fatalf("snapshot storage: got=%d want=%d", b.live, want)
	}
	eval.Evidence["one"].Occupancy = 0.25
	if snapshot.evidence["one"].Occupancy != 0.75 {
		t.Fatal("snapshot changed with caller")
	}
	scope.Close()
	if b.live != 0 {
		t.Fatal("snapshot owner leak", b.live)
	}
	b = &memoryBudget{limit: want - 1}
	scope, err = egraphcore.NewMemoryScope(b)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err = snapshotScoringWithMemory(eval, scope)
	if snapshot != nil || !errors.Is(err, ErrMemoryBudgetExceeded) {
		t.Fatal("snapshot admitted before reservations", snapshot, err)
	}
	scope.Close()
	if b.live != 0 {
		t.Fatal("refused snapshot owner leak", b.live)
	}
}

func TestSetupEncodingBounds(t *testing.T) {
	setupHostOnly(t)
	for _, value := range []any{map[string]*ProfileEvidence(nil), map[string]*ProfileEvidence{"\x00<\xff\"": {Epoch: strings.Repeat("<&\x00\xff", 8), CollectedAt: time.Unix(1700000000, 123)}}, DefaultBudget(), DefaultScoringModel("epoch"), []string{"\x00<\xff\"", "ordinary"}} {
		n, err := setupJSONBound(value, nil)
		data, encodeErr := json.Marshal(value)
		if err != nil || encodeErr != nil || int64(len(data)) > n {
			t.Fatalf("unsafe bound=%d bytes=%d value=%T: %v %v", n, len(data), value, err, encodeErr)
		}
	}
	rule := egraph.MustRewrite("ordinary", "(f ?x)", "(g ?x)")
	n, err := setupRuleBytes(rule, nil)
	if err != nil || n < int64(len(rule.Identity())) {
		t.Fatal(n, err)
	}
	cycle := &egraph.Pattern{Op: "cycle"}
	cycle.Children = []*egraph.Pattern{cycle}
	rule.LHS = cycle
	if _, err := setupRuleBytes(rule, nil); !errors.Is(err, ErrMemoryAccountingUnavailable) {
		t.Fatal("cyclic identity traversed", err)
	}
	var total int64 = 1
	if !errors.Is(setupAdd(&total, int64(^uint64(0)>>1)), ErrMemoryBudgetExceeded) || total != 1 {
		t.Fatal("size overflow not refused")
	}
}

func TestSetupEarlyCancelAndUnwind(t *testing.T) {
	t.Logf("setup sizes: work=%d ledger=%d scope=%d", reflect.TypeFor[workBudget]().Size(), reflect.TypeFor[memoryBudget]().Size(), reflect.TypeFor[egraphcore.MemoryScope]().Size())
	setupHostOnly(t)
	for _, strategy := range []string{"beam", "mcts"} {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		result, err := bookkeepingSearch(ctx, strategy, DefaultBudget(), setupState(), nil, setupMock())
		if result != nil || !errors.Is(err, ErrSearchCancelled) || !errors.Is(err, context.Canceled) {
			t.Fatal(result, err)
		}
	}
	for _, limit := range []int64{1, int64(reflect.TypeFor[workBudget]().Size() + reflect.TypeFor[memoryBudget]().Size()), 1024} {
		_, work, err := searchWorkBudget(t.Context(), SearchBudget{MaxMemoryBytes: limit}, setupMock())
		if work != nil {
			work.closeSearch()
			if work.memory.live != 0 {
				t.Fatal("work setup leaked", work.memory.live)
			}
		} else if !errors.Is(err, ErrMemoryBudgetExceeded) {
			t.Fatal(work, err)
		}
	}
}

func TestSetupPolicyEncodingCompatibility(t *testing.T) {
	setupHostOnly(t)
	for _, strategy := range []string{"beam", "mcts"} {
		for _, limited := range []bool{false, true} {
			budget := SearchBudget{MaxStates: 1}
			if limited {
				budget.MaxMemoryBytes = 1 << 20
			}
			eval := setupMock()
			eval.Evidence = map[string]*ProfileEvidence{"unused": {Epoch: "old"}}
			ctx, work, err := searchWorkBudget(t.Context(), budget, eval)
			if err != nil {
				t.Fatal(err)
			}
			book, err := newSearchBookkeeping(ctx)
			if err != nil {
				t.Fatal(err)
			}
			rules := []*egraph.Rewrite{egraph.MustRewrite("ordinary", "(f ?x)", "(g ?x)")}
			config := searchPolicyOptions{beamWidth: 4, iterations: 7, rolloutDepth: 2, exploration: 1.4142135623730951}
			options := map[string]string{"beam_width": "4", "tie_order": "full-replay-path-v1"}
			if strategy == "mcts" {
				options = map[string]string{"iterations": "7", "rollout_depth": "2", "exploration": "1.4142135623730951"}
			}
			want, wantComplete, wantErr := predecessorSetupPolicy(strategy, "root", budget, options, 42, rules, eval, work, profileActions{}, "")
			got, complete, err := makeSearchPolicy(strategy, "root", budget, config, 42, rules, eval, work, profileActions{}, "")
			if wantErr != nil || err != nil || complete != wantComplete || !reflect.DeepEqual(got, want) {
				t.Fatalf("policy encoding changed: got=%+v want=%+v errors=%v/%v", got, want, err, wantErr)
			}
			book.close()
			work.closeSearch()
			if work.memory.live != 0 {
				t.Fatal("policy owner leaked", work.memory.live)
			}
		}
	}
}

func predecessorSetupPolicy(strategy, root string, budget SearchBudget, options map[string]string, seed int64, rules []*egraph.Rewrite, eval Evaluator, work *workBudget, profile profileActions, evaluatorID string) (cudacache.SearchPolicyID, bool, error) {
	id := cudacache.SearchPolicyID{Strategy: strategy, Seed: seed, Options: options}
	id.Options["contract_version"] = "1"
	id.Options["state_budget"] = "initial-plus-transition-attempts-v1"
	id.Options["root"] = root
	data, err := json.Marshal(budget)
	if err != nil {
		return id, false, err
	}
	id.Options["budget"] = string(data)
	identities := make([]string, len(rules))
	for i, rule := range rules {
		if rule == nil || rule.LHS == nil || rule.RHS == nil {
			return id, false, fmt.Errorf("cudasearch: invalid rewrite at index %d", i)
		}
		identities[i] = rule.Identity()
	}
	data, _ = json.Marshal(identities)
	id.RuleSet = fmt.Sprintf("%x", sha256.Sum256(data))
	complete := evaluatorID != ""
	id.Options["evaluator_identity"] = evaluatorID
	switch e := eval.(type) {
	case *MockEvaluator:
		id.Options["evaluator"] = "mock-v1"
		id.Options["timing_delay"] = e.TimingDelay.String()
		id.Options["workload"] = e.Workload.Key()
		model := work.scoring.model
		if model.EvidenceMaxAge == 0 {
			model.EvidenceTime = time.Time{}
		}
		data, err = json.Marshal(model)
		if err != nil {
			return id, false, fmt.Errorf("cudasearch: scoring policy: %w", err)
		}
		id.Options["scoring"] = string(data)
		id.ScoringEpoch = model.Epoch
		data, err = json.Marshal(work.scoring.evidence)
		if err != nil {
			id.Options["scoring_evidence_error"] = err.Error()
		} else {
			id.Options["scoring_evidence"] = fmt.Sprintf("%x", sha256.Sum256(data))
		}
		_, astSize := e.CostModel.(egraph.AstSizeCost)
		if (e.CostModel == nil || astSize) && e.FailCandidate == nil && e.Workload.ResetState == nil {
			complete = true
		}
		if err != nil {
			complete = false
		}
	case *ExecutableEvaluator:
		id.Options["execution_generation"] = fmt.Sprint(e.Generation)
		id.Options["evaluator"] = "executable-v1"
		id.Options["workload"] = e.Workload.Key()
		data, err = json.Marshal(struct {
			Rounds         int
			Timing         string
			Policy         OutputPolicy
			CompileTimeout time.Duration
		}{e.Rounds, string(e.TimingKind), e.Policy, e.CompileTimeout})
		if err != nil {
			return id, false, err
		}
		id.Options["evaluation"] = string(data)
	default:
		id.Options["evaluator"] = "external"
	}
	if profile.decision != nil {
		id.Options["profile_decision"] = profile.decision.ID
		if profile.decision.ID == "" {
			complete = false
		}
	}
	if profile.memory != nil {
		id.Options["memory_decision"] = profile.memory.ID
	}
	return id, complete, nil
}
