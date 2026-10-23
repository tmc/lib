package cudasearch

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// SearchMemory supplies exploration priors. RuleTechniques explicitly maps
// legal rewrite names to catalog techniques; no technique authorizes a rewrite.
// Domain is a caller-defined compatibility digest for cross-task replay.
// An unqualified metric classification remains PendingHardware in the decision.
// Memory weights take precedence over ProfileActions ranks, which break ties.
// Do not mutate the configuration while Search is freezing it.
type SearchMemory struct {
	Metrics        ProfileMetrics
	Profile        *BottleneckProfile // Optional in-process hardware observation.
	WorkloadKey    string             // Exact profile compatibility when Profile is set.
	InputDigest    string             // Exact profile input when Profile is set.
	Family         cudatrial.BottleneckFamily
	TargetArch     string
	Domain         string
	Knowledge      *cudacache.StrategyKnowledgeBase
	Trajectories   *cudatrial.TrajectoryMemory
	RuleTechniques map[string]string
}

func validateMemoryEvaluator(memory *SearchMemory, eval Evaluator, scratch *egraphcore.MemoryScope) error {
	if memory == nil || memory.Profile == nil {
		return nil
	}
	e, ok := eval.(*ExecutableEvaluator)
	if !ok || e == nil {
		return fmt.Errorf("cudasearch: bottleneck profile requires its bound executable evaluator")
	}
	if err := reserveWorkloadKey(scratch, e.Workload); err != nil {
		return err
	}
	if e.Workload.Key() != memory.WorkloadKey {
		return fmt.Errorf("cudasearch: bottleneck profile requires its bound executable evaluator")
	}
	return nil
}

// MemoryDecision freezes all consulted priors once at search start. ID is a
// search-policy digest, not a compiler cache key or a performance measurement.
type MemoryDecision struct {
	Version            int
	ID                 string
	State              BottleneckState
	TargetArch, Domain string
	Strategies         []cudacache.Strategy
	Rewards            []cudatrial.RuleReward
	TrajectoryIDs      []string
	ProfileID          string
	RuleTechniques     map[string]string
	Weights            map[string]float64
}

func freezeSearchMemory(config *SearchMemory, p *profileActions) error {
	return freezeBoundedSearchMemory(config, p, nil, nil)
}

func freezeBoundedSearchMemory(config *SearchMemory, p *profileActions, scratch, result *egraphcore.MemoryScope) error {
	if config == nil {
		return nil
	}
	if err := reserveMemoryDecisionBase(config, scratch, result); err != nil {
		return err
	}
	if cudakernel.ValidateTargetArch(config.TargetArch) != nil || len(config.RuleTechniques) > 4096 {
		return fmt.Errorf("cudasearch: invalid search memory target or mapping")
	}
	state := ClassifyBottleneck(config.Metrics)
	if config.Profile != nil {
		p := config.Profile
		// HardwareObserved hashes JSON. Admit that encoding, including map
		// sorting slots, before calling the provenance checker.
		if err := reserveMemoryJSON(p, scratch); err != nil {
			return err
		}
		if err := scratch.Reserve(64); err != nil {
			return err
		}
		if !p.HardwareObserved() || p.TargetArch != config.TargetArch || p.WorkloadKey != config.WorkloadKey || p.InputDigest != config.InputDigest || (config.Family != "" && config.Family != p.State.StateName) {
			return fmt.Errorf("cudasearch: unqualified or incompatible bottleneck profile")
		}
		state = p.State
		var err error
		state, err = copyMemoryState(p.State, result)
		if err != nil {
			return err
		}
	}
	if config.Family != "" {
		if !config.Family.Valid() {
			return fmt.Errorf("cudasearch: invalid search memory family")
		}
		if config.Profile == nil {
			state = BottleneckState{StateName: config.Family, PendingHardware: true, Reason: "caller-declared exploration category"}
		}
	}
	if !state.StateName.Valid() {
		return fmt.Errorf("cudasearch: search memory category unavailable: %s", state.Reason)
	}
	rules, err := bookkeepingMap[string, string](result)
	if err != nil {
		return err
	}
	weights, err := bookkeepingMap[string, float64](result)
	if err != nil {
		return err
	}
	d := &MemoryDecision{Version: 1, State: state, TargetArch: config.TargetArch, Domain: config.Domain, RuleTechniques: rules, Weights: weights}
	if config.Profile != nil {
		d.ProfileID = config.Profile.ID
	}
	if config.Domain != "" || config.Trajectories != nil {
		if len(config.Domain) != 64 {
			return fmt.Errorf("cudasearch: invalid cross-task compatibility digest")
		}
		if err := scratch.Reserve(160); err != nil {
			return err
		}
		b, err := hex.DecodeString(config.Domain)
		if err != nil || len(b) != 32 || hex.EncodeToString(b) != config.Domain {
			return fmt.Errorf("cudasearch: invalid cross-task compatibility digest")
		}
	}
	if config.Trajectories != nil {
		snapshot, err := config.Trajectories.ReplayWithMemory(config.Domain, state.StateName, config.TargetArch, scratch, result)
		if err != nil {
			return err
		}
		d.Rewards = snapshot.Rewards
		d.TrajectoryIDs = snapshot.ReceiptIDs
	}
	if config.Knowledge != nil {
		d.Strategies, err = config.Knowledge.QueryWithMemory(state.StateName, config.TargetArch, scratch, result)
		if err != nil {
			return err
		}
	}
	techniques, err := bookkeepingMap[string, float64](scratch)
	if err != nil {
		return err
	}
	for _, s := range d.Strategies {
		if s.Confidence > techniques[s.Technique] {
			if err := bookkeepingPut(scratch, techniques, s.Technique, s.Confidence); err != nil {
				return err
			}
		}
	}
	for rule, technique := range config.RuleTechniques {
		if rule == "" || technique == "" || len(rule) > 256 || len(technique) > 128 {
			return fmt.Errorf("cudasearch: invalid technique mapping")
		}
		if err := result.Reserve(2*int64(len(rule)) + int64(len(technique))); err != nil {
			return err
		}
		if err := bookkeepingPut(result, d.RuleTechniques, rule, technique); err != nil {
			return err
		}
		if err := bookkeepingPut(result, d.Weights, rule, techniques[technique]); err != nil {
			return err
		}
	}
	for _, r := range d.Rewards {
		if _, exists := d.Weights[r.Rule]; !exists {
			if err := result.Reserve(int64(len(r.Rule))); err != nil {
				return err
			}
		}
		if err := bookkeepingPut(result, d.Weights, r.Rule, d.Weights[r.Rule]+r.Reward); err != nil {
			return err
		}
	}
	if err := reserveMemoryJSON(d, scratch); err != nil {
		return err
	}
	if err := result.Reserve(64); err != nil {
		return err
	}
	b, err := json.Marshal(d)
	if err != nil {
		return err
	}
	if len(b) > 8<<20 {
		return fmt.Errorf("cudasearch: search memory decision byte limit")
	}
	d.ID = fmt.Sprintf("%x", sha256.Sum256(b))
	p.memory = d
	p.weights = d.Weights
	return nil
}

func (p profileActions) memoryWeight(rule string) float64 { return p.weights[rule] }

// FrozenMemory returns a detached copy of the consulted search memory. Nil
// means no memory policy was requested, not an absence of relevant history.
func (r *SearchResult) FrozenMemory() *MemoryDecision {
	if r == nil || r.MemoryDecision == nil {
		return nil
	}
	d := *r.MemoryDecision
	d.State.SecondaryCharacteristics = append(SecondaryCharacteristics(nil), d.State.SecondaryCharacteristics...)
	d.State.RelativePatterns = make(RelativePattern, len(r.MemoryDecision.State.RelativePatterns))
	for k, v := range r.MemoryDecision.State.RelativePatterns {
		d.State.RelativePatterns[k] = v
	}
	d.Strategies = append([]cudacache.Strategy(nil), d.Strategies...)
	for i := range d.Strategies {
		d.Strategies[i].HistoricalSpeedups = append([]float64(nil), d.Strategies[i].HistoricalSpeedups...)
		d.Strategies[i].TrajectoryIDs = append([]string(nil), d.Strategies[i].TrajectoryIDs...)
	}
	d.Rewards = append([]cudatrial.RuleReward(nil), d.Rewards...)
	d.TrajectoryIDs = append([]string(nil), d.TrajectoryIDs...)
	d.RuleTechniques = make(map[string]string, len(r.MemoryDecision.RuleTechniques))
	for k, v := range r.MemoryDecision.RuleTechniques {
		d.RuleTechniques[k] = v
	}
	d.Weights = make(map[string]float64, len(r.MemoryDecision.Weights))
	for k, v := range r.MemoryDecision.Weights {
		d.Weights[k] = v
	}
	return &d
}
