package cudasearch

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func makeSearchPolicy(strategy, root string, budget SearchBudget, options map[string]string, seed int64, rules []*egraph.Rewrite, eval Evaluator, work *workBudget, profile profileActions, evaluatorID string) (cudacache.SearchPolicyID, bool, error) {
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
			// Stale evidence may be ignored by scoring even if its values
			// cannot be encoded. Provenance must not change that decision.
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
