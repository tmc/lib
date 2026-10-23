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

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// These catalogs contain synthetic exploration declarations, not hardware
// observations. Their caller-owned construction is outside the search ledger.
func priorBudgetCatalog(t *testing.T, rows, ids int) *SearchMemory {
	t.Helper()
	k := new(cudacache.StrategyKnowledgeBase)
	for i := range rows {
		s := cudacache.Strategy{Family: cudatrial.MemoryBandwidthLimited, TargetArch: "sm_121", Technique: fmt.Sprintf("tiling%d", i), Confidence: 1, PredictedSpeedup: 1,
			Observations: uint64(ids), HistoricalSpeedups: make([]float64, min(ids, 256)), TrajectoryIDs: make([]string, ids)}
		for j := range s.HistoricalSpeedups {
			s.HistoricalSpeedups[j] = 2
		}
		for j := range s.TrajectoryIDs {
			s.TrajectoryIDs[j] = fmt.Sprintf("%064x", j)
		}
		if err := k.Update(s); err != nil {
			t.Fatal(err)
		}
	}
	return &SearchMemory{Family: cudatrial.MemoryBandwidthLimited, TargetArch: "sm_121", Knowledge: k, RuleTechniques: map[string]string{"z": "tiling0"}}
}

func runPriorBudgetSearch(t *testing.T, strategy string, memory *SearchMemory, limit int64, calls *int, ledger **memoryBudget) (*SearchResult, error) {
	t.Helper()
	state, _ := beamContractState()
	rules := []*egraph.Rewrite{egraph.MustRewrite("a", "base", "first"), egraph.MustRewrite("z", "base", "preferred")}
	budget := DefaultBudget()
	budget.MaxMemoryBytes, budget.MaxDepth, budget.MaxStates, budget.MaxEvaluations = limit, 2, 2, 2
	budget.CallbackMemoryBytes = fixtureCallbackMemoryBytes(2)
	eval := beamEvaluatorFunc(func(ctx context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
		(*calls)++
		*ledger = searchMemoryBudget(ctx)
		return &CandidatePlan{Validated: true, Cost: float64(10 - len(s.Derivation)), Derivation: s.Derivation}, nil
	})
	if strategy == "beam" {
		return NewBeamSearch(BeamConfig{Budget: budget, Memory: memory, EvaluatorID: "prior-budget-fixture"}).Search(t.Context(), state, rules, eval)
	}
	return NewMCTSSearch(MCTSConfig{Budget: budget, Memory: memory, Seed: 2, EvaluatorID: "prior-budget-fixture"}).Search(t.Context(), state, rules, eval)
}

func TestSearchMemoryPriorRefusesBeforeEvaluation(t *testing.T) {
	memory := priorBudgetCatalog(t, 16, 1024)
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			calls := 0
			var ledger *memoryBudget
			result, err := runPriorBudgetSearch(t, strategy, memory, 128<<10, &calls, &ledger)
			if result != nil {
				result.Close()
			}
			if !errors.Is(err, ErrMemoryBudgetExceeded) || calls != 0 || result != nil {
				t.Fatalf("result=%+v calls=%d error=%v", result, calls, err)
			}
		})
	}
}

func TestSearchMemoryPriorRetainedAndUnchanged(t *testing.T) {
	memory := priorBudgetCatalog(t, 1, 256)
	want := MemoryDecision{Version: 1,
		State:      BottleneckState{StateName: memory.Family, PendingHardware: true, Reason: "caller-declared exploration category"},
		TargetArch: memory.TargetArch, Strategies: memory.Knowledge.Query(memory.Family, memory.TargetArch),
		RuleTechniques: map[string]string{"z": "tiling0"}, Weights: map[string]float64{"z": 1}}
	data, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	wantID := fmt.Sprintf("%x", sha256.Sum256(data))
	for _, strategy := range []string{"beam", "mcts"} {
		for _, limit := range []int64{0, 4 << 20} {
			t.Run(fmt.Sprintf("%s/%d", strategy, limit), func(t *testing.T) {
				calls := 0
				var ledger *memoryBudget
				r, err := runPriorBudgetSearch(t, strategy, memory, limit, &calls, &ledger)
				if err != nil || r == nil {
					t.Fatalf("search=%+v error=%v", r, err)
				}
				defer r.Close()
				if calls != 2 || r.MemoryDecision.ID != wantID || r.PolicyID.Options["memory_decision"] != wantID || len(r.Winner.Derivation) != 1 || r.Winner.Derivation[0].RuleName != "z" {
					t.Fatalf("prior ranking/lineage changed: calls=%d decision=%+v winner=%+v", calls, r.MemoryDecision, r.Winner)
				}
				if limit > 0 {
					// At least the actual strategy value,256 ID headers/bytes and
					//256 historical floats remain owned after Search returns.
					minimum := int64(reflect.TypeFor[cudacache.Strategy]().Size()) + 256*(64+int64(reflect.TypeFor[string]().Size())+8)
					if r.bookkeeping.Bytes() < minimum || ledger.live < minimum {
						t.Fatalf("decision escaped ledger: result=%d live=%d want>=%d", r.bookkeeping.Bytes(), ledger.live, minimum)
					}
					r.Close()
					if ledger.live != 0 {
						t.Fatalf("result close leaked %d units", ledger.live)
					}
				}
			})
		}
	}
}

func TestSearchMemoryPriorFreezeAdmission(t *testing.T) {
	config := priorBudgetCatalog(t, 1, 1)
	ledger := &memoryBudget{limit: 1 << 20}
	scratch, _ := egraphcore.NewMemoryScope(ledger)
	result, _ := egraphcore.NewMemoryScope(ledger)
	var p profileActions
	if err := freezeBoundedSearchMemory(config, &p, scratch, result); err != nil || p.memory == nil {
		t.Fatalf("freeze: %v", err)
	}
	scratch.Close()
	if ledger.live != result.Bytes() || ledger.live == 0 {
		t.Fatal("frozen decision does not retain a result owner")
	}
	result.Close()
	if ledger.live != 0 {
		t.Fatal("frozen decision leaked units")
	}
	ledger = &memoryBudget{limit: 1024}
	scratch, _ = egraphcore.NewMemoryScope(ledger)
	result, _ = egraphcore.NewMemoryScope(ledger)
	ledger.limit = ledger.live
	var refused profileActions
	if err := freezeBoundedSearchMemory(config, &refused, scratch, result); !errors.Is(err, ErrMemoryBudgetExceeded) || refused.memory != nil {
		t.Fatalf("unadmitted decision returned: %+v, %v", refused, err)
	}
	scratch.Close()
	result.Close()
	if ledger.live != 0 {
		t.Fatal("refused freeze leaked reservations")
	}
}

func TestSearchMemoryPriorPartialRefusal(t *testing.T) {
	config := priorBudgetCatalog(t, 1, 128)
	for _, limit := range []int64{256, 4096, 16384} {
		ledger := &memoryBudget{limit: limit}
		scratch, _ := egraphcore.NewMemoryScope(ledger)
		result, _ := egraphcore.NewMemoryScope(ledger)
		var p profileActions
		if err := freezeBoundedSearchMemory(config, &p, scratch, result); !errors.Is(err, ErrMemoryBudgetExceeded) || p.memory != nil {
			t.Fatalf("limit=%d decision=%+v error=%v", limit, p.memory, err)
		}
		scratch.Close()
		result.Close()
		if ledger.live != 0 {
			t.Fatalf("limit=%d leaked %d units", limit, ledger.live)
		}
	}
}

func TestSearchMemoryPriorProfileBudget(t *testing.T) {
	// As in TestSearchProfileIdentityAndMutation, forge a white-box host seal
	// only to exercise the admitted branch. This is not native evidence.
	p := BottleneckProfile{TargetArch: "sm_121", WorkloadKey: strings.Repeat("a", 64), InputDigest: strings.Repeat("b", 64),
		State: BottleneckState{StateName: cudatrial.MemoryBandwidthLimited, RelativePatterns: make(RelativePattern), SecondaryCharacteristics: make(SecondaryCharacteristics, 1024)}}
	for i := range 1024 {
		p.State.RelativePatterns[fmt.Sprintf("pattern%d", i)] = "observed"
		p.State.SecondaryCharacteristics[i] = "secondary"
	}
	p.ID = p.digest()
	p.seal = p.ID
	config := &SearchMemory{Family: p.State.StateName, TargetArch: p.TargetArch, Profile: &p, WorkloadKey: p.WorkloadKey, InputDigest: p.InputDigest}
	for _, limit := range []int64{32 << 10, 8 << 20} {
		ledger := &memoryBudget{limit: limit}
		scratch, _ := egraphcore.NewMemoryScope(ledger)
		result, _ := egraphcore.NewMemoryScope(ledger)
		var frozen profileActions
		err := freezeBoundedSearchMemory(config, &frozen, scratch, result)
		if limit == 32<<10 {
			if !errors.Is(err, ErrMemoryBudgetExceeded) || frozen.memory != nil {
				t.Fatalf("large profile escaped admission: %+v, %v", frozen.memory, err)
			}
		} else {
			if err != nil || frozen.memory == nil || frozen.memory.ProfileID != p.ID || len(frozen.memory.State.RelativePatterns) != 1024 || result.Bytes() < 1024*32 {
				t.Fatalf("bounded profile snapshot: %+v, %v", frozen.memory, err)
			}
			p.State.RelativePatterns["pattern0"] = "changed"
			if frozen.memory.State.RelativePatterns["pattern0"] != "observed" {
				t.Fatal("profile state aliases its source")
			}
		}
		scratch.Close()
		result.Close()
		if ledger.live != 0 {
			t.Fatal("profile snapshot leaked reservations")
		}
	}
}
