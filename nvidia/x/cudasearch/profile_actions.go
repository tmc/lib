package cudasearch

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"time"

	"github.com/tmc/lib/egraphcore"
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

// freezeBoundedProfile admits config.Snapshot and freezes the decision.
// Rank and encoding storage is charged to scratch, which the caller closes
// when the search returns; the decision and guidance returned in p are charged
// to result, which transfers to SearchResult. Nil scopes are unlimited and
// cannot fail. A storage refusal is an error; an unusable profile is not.
func freezeBoundedProfile(config *ProfileActions, scratch, result *egraphcore.MemoryScope) (p profileActions, err error) {
	if config == nil {
		return p, nil
	}
	now := config.Time
	if now.IsZero() {
		now = time.Now()
	}
	if err := reserveDecision(config, result); err != nil {
		return p, err
	}
	d := &ProfileDecision{Version: 1, Target: config.Target, Time: now.UTC(), MaxAge: config.MaxAge, MinCoverage: strconv.FormatFloat(config.MinCoverage, 'g', -1, 64), MaxLosses: config.MaxLosses, MaxUncertaintyNs: config.MaxUncertaintyNs, Priors: append([]MetricRulePrior(nil), config.Priors...)}
	if config.Snapshot != nil {
		d.SnapshotSHA256, d.SnapshotError, err = snapshotDigest(config.Snapshot, scratch, result)
		if err != nil {
			return p, err
		}
	} else {
		d.SnapshotError = "missing snapshot"
	}
	p, err = admitProfile(config, now, scratch, result)
	if err != nil {
		return profileActions{}, err
	}
	d.Fallback = p.fallback
	if err := result.Reserve(int64(len(d.Fallback))); err != nil {
		return profileActions{}, err
	}
	if err := bookkeepingSlots[string](result, len(p.ranks)); err != nil {
		return profileActions{}, err
	}
	d.ActiveRules = make([]string, len(p.ranks))
	for rule, rank := range p.ranks {
		d.ActiveRules[rank] = rule
	}
	if err := identifyDecision(d, scratch, result); err != nil {
		return profileActions{}, err
	}
	p.decision = d
	return p, nil
}

// admitProfile applies the quality policy and priors to the snapshot. A
// profile that cannot guide exploration yields a fallback reason, not an error.
func admitProfile(config *ProfileActions, now time.Time, scratch, result *egraphcore.MemoryScope) (profileActions, error) {
	fail := func(reason string) (profileActions, error) { return profileActions{fallback: reason}, nil }
	if config.MaxAge <= 0 || config.MaxUncertaintyNs < 0 || math.IsNaN(config.MinCoverage) || config.MinCoverage < 0 || config.MinCoverage > 1 {
		return fail("invalid profile quality policy")
	}
	admit := func(g *cudaprof.SearchGuidance) error {
		if g.Timestamp.IsZero() || g.Timestamp.After(now) || now.Sub(g.Timestamp) > config.MaxAge {
			return profileFallback("profile timestamp missing, future or expired")
		}
		if g.Coverage < config.MinCoverage || g.Losses > config.MaxLosses || g.UncertaintyNs < 0 || g.UncertaintyNs > config.MaxUncertaintyNs {
			return profileFallback("profile quality bounds not met")
		}
		for _, prior := range config.Priors {
			if prior.Metric == "" || prior.Unit == "" || prior.Rule == "" || prior.Minimum < 0 {
				return profileFallback("invalid metric rule priority")
			}
			if _, ok := g.StallCounts[prior.Metric]; !ok || g.MetricUnits[prior.Metric] != prior.Unit {
				return profileFallback(fmt.Sprintf("profile metric or unit unavailable: %s", prior.Metric))
			}
		}
		return nil
	}
	g, err := config.Snapshot.GuidanceBounded(config.Target, admit, scratch, result)
	if err != nil {
		if memory := errors.Join(scratch.Err(), result.Err()); memory != nil {
			return profileActions{}, memory
		}
		if errors.Is(err, egraphcore.ErrMemorySize) {
			return profileActions{}, err
		}
		return fail(err.Error())
	}
	p := profileActions{guidance: g}
	if p.ranks, err = bookkeepingMap[string, int](scratch); err != nil {
		return profileActions{}, err
	}
	for _, prior := range config.Priors {
		if g.StallCounts[prior.Metric] >= prior.Minimum {
			if _, ok := p.ranks[prior.Rule]; !ok {
				if err := bookkeepingPut(scratch, p.ranks, prior.Rule, len(p.ranks)); err != nil {
					return profileActions{}, err
				}
			}
		}
	}
	return p, nil
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
