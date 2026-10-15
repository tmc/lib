package cudasearch

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func searchMemoryFixture(t *testing.T) *SearchMemory {
	t.Helper()
	k := new(cudacache.StrategyKnowledgeBase)
	if err := k.Update(cudacache.Strategy{Family: cudatrial.MemoryBandwidthLimited, TargetArch: "sm_121", Technique: "shared_memory_tiling", Confidence: 1, PredictedSpeedup: 1}); err != nil {
		t.Fatal(err)
	}
	return &SearchMemory{Family: cudatrial.MemoryBandwidthLimited, TargetArch: "sm_121", Knowledge: k, RuleTechniques: map[string]string{"z": "shared_memory_tiling"}}
}

func memorySearch(t *testing.T, strategy string, m *SearchMemory, mutate func()) *SearchResult {
	t.Helper()
	s, _ := beamContractState()
	rules := []*egraph.Rewrite{egraph.MustRewrite("a", "base", "first"), egraph.MustRewrite("z", "base", "preferred")}
	b := DefaultBudget()
	b.CallbackMemoryBytes = fixtureCallbackMemoryBytes(b.MaxDepth)
	b.MaxStates = 2
	b.MaxEvaluations = 2
	eval := beamEvaluatorFunc(func(_ context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
		if mutate != nil {
			mutate()
		}
		return &CandidatePlan{Validated: true, Cost: float64(10 - len(s.Derivation)), Derivation: append([]egraph.Action(nil), s.Derivation...)}, nil
	})
	var r *SearchResult
	var err error
	if strategy == "beam" {
		r, err = NewBeamSearch(BeamConfig{Budget: b, Memory: m}).Search(context.Background(), s, rules, eval)
	} else {
		r, err = NewMCTSSearch(MCTSConfig{Budget: b, Memory: m, Seed: 2}).Search(context.Background(), s, rules, eval)
	}
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestBeamSearch_BottleneckGuidedExpansion(t *testing.T) {
	m := searchMemoryFixture(t)
	r := memorySearch(t, "beam", m, func() {
		m.RuleTechniques["z"] = "not-the-frozen-technique"
		if err := m.Knowledge.Update(cudacache.Strategy{Family: m.Family, TargetArch: m.TargetArch, Technique: "shared_memory_tiling", Confidence: 0, PredictedSpeedup: 1}); err != nil {
			t.Fatal(err)
		}
	})
	if len(r.Winner.Derivation) != 1 || r.Winner.Derivation[0].RuleName != "z" || r.Winner.Cost != 9 {
		t.Fatalf("guidance not consumed: %+v", r)
	}
	d := r.FrozenMemory()
	if d == nil || d.ID == "" || d.RuleTechniques["z"] != "shared_memory_tiling" || !d.State.PendingHardware || r.PolicyID.Options["memory_decision"] != d.ID {
		t.Fatalf("frozen policy: %+v", d)
	}
	d.Weights["z"] = -100
	if r.FrozenMemory().Weights["z"] != 1 {
		t.Fatal("detached copy mutated result")
	}
}

func TestMutant_UnweightedSearchIgnoresBottleneck(t *testing.T) {
	guided := memorySearch(t, "beam", searchMemoryFixture(t), nil)
	unguided := memorySearch(t, "beam", nil, nil)
	if guided.Winner.Derivation[0].RuleName != "z" || unguided.Winner.Derivation[0].RuleName != "a" || guided.PolicyID.String() == unguided.PolicyID.String() {
		t.Fatalf("actual exploration did not differ: guided=%+v unguided=%+v", guided.Winner, unguided.Winner)
	}
}

func TestMCTSSearch_CrossTaskMemoryPrioritizesRewrites(t *testing.T) {
	m := searchMemoryFixture(t)
	m.Knowledge = nil
	m.Domain = strings.Repeat("a", 64)
	m.Trajectories, _ = cudatrial.NewTrajectoryMemory(2)
	// Synthetic persisted declaration tests replay mechanics, not qualification.
	r := cudatrial.TrajectoryReceipt{Version: 1, WorkloadKey: strings.Repeat("b", 64), Domain: m.Domain, TargetArch: "sm_121", ArtifactDigest: m.Domain, CompilerKey: m.Domain, BaselineArtifactDigest: m.Domain, BaselineCompilerKey: m.Domain, InputDigest: m.Domain, PolicyID: m.Domain, AppliedRules: []string{"z"}, BottleneckState: m.Family, MeasuredLatencyNs: 50, BaselineLatencyNs: 100, SpeedupRatio: 2, VerifiedCorrect: true, DeviceUUID: "synthetic-integrity-fixture", LockPID: 123}
	b, _ := json.Marshal(r)
	r.SHA256 = fmt.Sprintf("%x", sha256.Sum256(b))
	if err := m.Trajectories.ImportTrusted(r); err != nil {
		t.Fatal(err)
	}
	result := memorySearch(t, "mcts", m, nil)
	if result.Winner.Derivation[0].RuleName != "z" || len(result.MemoryDecision.Rewards) != 1 || result.Winner.Cost != 9 {
		t.Fatalf("cross-task prior not consumed: %+v", result)
	}
	m.Domain = strings.Repeat("c", 64)
	other := memorySearch(t, "mcts", m, nil)
	if len(other.MemoryDecision.Rewards) != 0 || other.PolicyID.String() == result.PolicyID.String() || other.Winner.Derivation[0].RuleName != "a" {
		t.Fatal("cross-domain history admitted or policy not bound")
	}
}
