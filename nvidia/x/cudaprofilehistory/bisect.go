package cudaprofilehistory

import (
	"fmt"
	"reflect"
)

// Observation retains separate supplied correctness and performance verdicts.
// Performance verdicts carry the comparison baseline and policy digest.
type Observation struct {
	Candidate    string  `json:"candidate"`
	Correctness  Verdict `json:"correctness"`
	Performance  Verdict `json:"performance"`
	Baseline     string  `json:"baseline"`
	PolicyDigest string  `json:"policy_digest"`
}

// BisectRequest is an ordered, bounded search over supplied observations.
// Predicate is correctness or performance; AssumeMonotonic must be explicit
// before FirstBad can be returned. No unobserved predecessor is inferred.
type BisectRequest struct {
	Candidates      []string      `json:"candidates"`
	Observations    []Observation `json:"observations"`
	Predicate       string        `json:"predicate"`
	Baseline        string        `json:"baseline"`
	PolicyDigest    string        `json:"policy_digest"`
	AssumeMonotonic bool          `json:"assume_monotonic"`
	MaxCandidates   int           `json:"max_candidates"`
}

// BisectPlan contains only supplied outcomes and a proposed next identifier.
// A conflict, unknown endpoint or missing predecessor prevents FirstBad.
type BisectPlan struct {
	Next     string `json:"next,omitempty"`
	FirstBad string `json:"first_bad,omitempty"`
	Reason   string `json:"reason"`
	Conflict bool   `json:"conflict"`
}

// PlanBisect proposes a deterministic unobserved midpoint without execution.
func PlanBisect(r BisectRequest) (BisectPlan, error) {
	var p BisectPlan
	if err := boundedValue(reflect.ValueOf(r), 64<<20); err != nil {
		return p, err
	}
	if r.MaxCandidates <= 0 || r.MaxCandidates > 100000 || len(r.Candidates) == 0 || len(r.Candidates) > r.MaxCandidates || len(r.Observations) > 2*r.MaxCandidates {
		return p, fmt.Errorf("history: bisect bounds exceeded")
	}
	if r.Predicate != "correctness" && r.Predicate != "performance" {
		return p, fmt.Errorf("history: invalid bisect predicate")
	}
	if r.Predicate == "performance" && (r.Baseline == "" || !validDigest(r.PolicyDigest)) {
		return p, fmt.Errorf("history: performance bisect missing comparison identity")
	}
	indexes := map[string]int{}
	for i, id := range r.Candidates {
		if id == "" {
			return p, fmt.Errorf("history: empty candidate")
		}
		if _, ok := indexes[id]; ok {
			return p, fmt.Errorf("history: duplicate candidate")
		}
		indexes[id] = i
	}
	observed := make([]bool, len(r.Candidates))
	verdicts := make([]Verdict, len(r.Candidates))
	for i := range verdicts {
		verdicts[i] = Unknown
	}
	for _, o := range r.Observations {
		i, ok := indexes[o.Candidate]
		if !ok {
			return p, fmt.Errorf("history: observation outside candidates")
		}
		for _, v := range []Verdict{o.Correctness, o.Performance} {
			if v != Pass && v != Fail && v != Unknown {
				return p, fmt.Errorf("history: invalid supplied verdict")
			}
		}
		v := o.Correctness
		if r.Predicate == "performance" {
			v = o.Performance
			if v != Unknown && (o.Baseline != r.Baseline || o.PolicyDigest != r.PolicyDigest) {
				return p, fmt.Errorf("history: incomparable performance observations")
			}
			if o.Correctness != Pass {
				v = Unknown
			}
		}
		if observed[i] && v != Unknown && verdicts[i] != v {
			p.Conflict = true
			p.Reason = "conflicting supplied observations"
			return p, nil
		}
		if v != Unknown {
			observed[i] = true
			verdicts[i] = v
		}
	}
	lo, hi := -1, len(verdicts)
	for i, v := range verdicts {
		if v == Pass {
			lo = i
		}
		if v == Fail && hi == len(verdicts) {
			hi = i
		}
	}
	if lo >= hi && r.AssumeMonotonic {
		p.Conflict = true
		p.Reason = "observations violate monotonic order"
		return p, nil
	}
	if lo >= 0 && hi < len(verdicts) && hi == lo+1 && r.AssumeMonotonic {
		p.FirstBad = r.Candidates[hi]
		p.Reason = "observed adjacent pass/fail under explicit monotonicity assumption"
		return p, nil
	}
	lower, upper := 0, len(verdicts)-1
	if r.AssumeMonotonic {
		if lo >= 0 {
			lower = lo + 1
		}
		if hi < len(verdicts) {
			upper = hi - 1
		}
	}
	mid := (lower + upper) / 2
	for distance := 0; distance < len(verdicts); distance++ {
		for _, i := range []int{mid - distance, mid + distance} {
			if i >= lower && i <= upper && !observed[i] {
				p.Next = r.Candidates[i]
				p.Reason = "unobserved candidate proposed; no execution or verdict inferred"
				return p, nil
			}
		}
	}
	p.Reason = "no unobserved candidate; insufficient adjacent qualified endpoints"
	return p, nil
}

// DecodeBisectRequest reads bounded strict JSON without accepting observations
// under unknown or case-aliased field names.
func DecodeBisectRequest(b []byte, maxBytes int) (BisectRequest, error) {
	var r BisectRequest
	if maxBytes <= 0 || maxBytes > 64<<20 || len(b) > maxBytes {
		return r, fmt.Errorf("history: bisect byte budget exceeded")
	}
	if err := strictJSON(b, &r); err != nil {
		return r, err
	}
	return r, nil
}
