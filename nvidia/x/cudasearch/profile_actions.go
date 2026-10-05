package cudasearch

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// MetricRulePrior gives a legal rule priority when a named native counter meets
// an explicit threshold in its declared unit. Entries are considered in order.
// This is an exploration policy, not a latency estimate or a legality decision.
type MetricRulePrior struct {
	Metric, Unit, Rule string
	Minimum            int64
}

// ProfileActions supplies offline evidence for action ordering. MaxAge must be
// positive; the other quality bounds are inclusive. An unknown clock uncertainty
// is not admitted. Time zero freezes the current time at search start.
// Invalid or unavailable evidence falls back to the unmodified action order.
type ProfileActions struct {
	Snapshot         *cudaprof.ProfileSnapshot
	Target           cudaprof.ProfileTarget
	Time             time.Time
	MaxAge           time.Duration
	MinCoverage      float64
	MaxLosses        uint64
	MaxUncertaintyNs int64
	Priors           []MetricRulePrior
}

// ProfileDecision records the exact profile action policy and its disposition.
// ID hashes this versioned record with ID and IdentityError empty; it is not an
// executable artifact identity. An unencodable record has an empty ID and error.
// MinCoverage uses round-trippable decimal text, including invalid NaN/Inf input.
type ProfileDecision struct {
	Version                       int
	ID, IdentityError             string
	Target                        cudaprof.ProfileTarget
	Time                          time.Time
	MaxAge                        time.Duration
	MinCoverage                   string
	MaxLosses                     uint64
	MaxUncertaintyNs              int64
	Priors                        []MetricRulePrior
	SnapshotSHA256, SnapshotError string
	ActiveRules                   []string
	Fallback                      string
}

type profileActions struct {
	guidance *cudaprof.SearchGuidance
	fallback string
	ranks    map[string]int
	decision *ProfileDecision
	memory   *MemoryDecision
	weights  map[string]float64
}

func freezeProfileActions(config *ProfileActions) (p profileActions) {
	if config == nil {
		return p
	}
	now := config.Time
	if now.IsZero() {
		now = time.Now()
	}
	d := &ProfileDecision{Version: 1, Target: config.Target, Time: now.UTC(), MaxAge: config.MaxAge, MinCoverage: strconv.FormatFloat(config.MinCoverage, 'g', -1, 64), MaxLosses: config.MaxLosses, MaxUncertaintyNs: config.MaxUncertaintyNs, Priors: append([]MetricRulePrior(nil), config.Priors...)}
	if config.Snapshot != nil {
		data, err := json.Marshal(config.Snapshot)
		if err != nil {
			d.SnapshotError = err.Error()
		} else {
			d.SnapshotSHA256 = fmt.Sprintf("%x", sha256.Sum256(data))
		}
	} else {
		d.SnapshotError = "missing snapshot"
	}
	defer func() {
		d.Fallback = p.fallback
		d.ActiveRules = make([]string, len(p.ranks))
		for rule, rank := range p.ranks {
			d.ActiveRules[rank] = rule
		}
		data, err := json.Marshal(d)
		if err != nil {
			d.IdentityError = err.Error()
		} else {
			d.ID = fmt.Sprintf("%x", sha256.Sum256(data))
		}
		p.decision = d
	}()
	fail := func(reason string) profileActions { return profileActions{fallback: reason} }
	if config.MaxAge <= 0 || config.MaxUncertaintyNs < 0 || math.IsNaN(config.MinCoverage) || config.MinCoverage < 0 || config.MinCoverage > 1 {
		return fail("invalid profile quality policy")
	}
	g, err := config.Snapshot.Guidance(config.Target)
	if err != nil {
		return fail(err.Error())
	}
	if g.Timestamp.IsZero() || g.Timestamp.After(now) || now.Sub(g.Timestamp) > config.MaxAge {
		return fail("profile timestamp missing, future or expired")
	}
	if g.Coverage < config.MinCoverage || g.Losses > config.MaxLosses || g.UncertaintyNs < 0 || g.UncertaintyNs > config.MaxUncertaintyNs {
		return fail("profile quality bounds not met")
	}
	p.guidance = g
	p.ranks = make(map[string]int)
	for _, prior := range config.Priors {
		if prior.Metric == "" || prior.Unit == "" || prior.Rule == "" || prior.Minimum < 0 {
			return fail("invalid metric rule priority")
		}
		count, ok := g.StallCounts[prior.Metric]
		if !ok || g.MetricUnits[prior.Metric] != prior.Unit {
			return fail(fmt.Sprintf("profile metric or unit unavailable: %s", prior.Metric))
		}
		if count >= prior.Minimum {
			if _, ok := p.ranks[prior.Rule]; !ok {
				p.ranks[prior.Rule] = len(p.ranks)
			}
		}
	}
	return p
}

func (p profileActions) rank(action egraph.ActionCandidate) int {
	if rank, ok := p.ranks[action.Rule.Name]; ok {
		return rank
	}
	return len(p.ranks)
}

func (p profileActions) order(actions []egraph.ActionCandidate) {
	if len(p.ranks) == 0 && len(p.weights) == 0 {
		return
	}
	sort.SliceStable(actions, func(i, j int) bool {
		wi, wj := p.memoryWeight(actions[i].Rule.Name), p.memoryWeight(actions[j].Rule.Name)
		if wi != wj {
			return wi > wj
		}
		return p.rank(actions[i]) < p.rank(actions[j])
	})
}

func (p profileActions) choose(actions []egraph.ActionCandidate, rng *rand.Rand) int {
	if len(p.ranks) == 0 && len(p.weights) == 0 {
		return rng.Intn(len(actions))
	}
	best := len(p.ranks)
	bestWeight := math.Inf(-1)
	var choices []int
	for i, action := range actions {
		rank := p.rank(action)
		weight := p.memoryWeight(action.Rule.Name)
		if weight > bestWeight || weight == bestWeight && rank < best {
			bestWeight = weight
			best = rank
			choices = choices[:0]
		}
		if weight == bestWeight && rank == best {
			choices = append(choices, i)
		}
	}
	return choices[rng.Intn(len(choices))]
}
