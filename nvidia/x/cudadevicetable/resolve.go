package cudadevicetable

import (
	"fmt"
	"os"
	"os/exec"
	"sync"
)

// ptxasCandidates are the usual install locations, tried after $PTXAS and
// $PATH. They are a convenience, not a contract: [FindPTXAS] reports what it
// found so a caller can record which assembler answered.
var ptxasCandidates = []string{
	"/usr/local/cuda/bin/ptxas",
	"/usr/local/cuda-13.0/bin/ptxas",
}

// FindPTXAS locates an assembler to resolve unprobed targets with. It returns
// an error rather than a guess when there is none, because a missing
// assembler must degrade to "unmeasured" and not to a fabricated answer.
func FindPTXAS() (string, error) {
	if p := os.Getenv("PTXAS"); p != "" {
		return p, nil
	}
	if p, err := exec.LookPath("ptxas"); err == nil {
		return p, nil
	}
	for _, p := range ptxasCandidates {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("cudadevicetable: no ptxas: set $PTXAS or put it on $PATH")
}

// A Resolver answers feature questions for targets the checked-in table does
// not cover, by assembling the embedded fixtures with a real ptxas.
//
// This is what keeps the table from being an allowlist. devices.json is a fast
// offline path holding what was measured before release; a Resolver measures
// the rest on demand, so a GPU that ships after this code does gets real
// answers rather than the defaults of its nearest recorded ancestor. The
// answers it produces carry the same provenance as the checked-in ones, and
// are cached per target for the life of the Resolver -- probing is a process
// launch per fixture, which is cheap once and wasteful in a loop.
//
// The zero Resolver is not usable; see [NewResolver].
type Resolver struct {
	ptxas string
	host  string

	mu    sync.Mutex
	cache map[string]Target
}

// NewResolver returns a Resolver using the ptxas at path. host is recorded in
// the probe stamp of anything it measures.
func NewResolver(ptxas, host string) *Resolver {
	return &Resolver{ptxas: ptxas, host: host, cache: map[string]Target{}}
}

// Ask returns the table's answer when it has one, and otherwise probes. A
// probe that fails for any reason -- no assembler, a crash, a PTX version too
// old for the target -- returns an unmeasured [Answer], never a negative one.
func (r *Resolver) Ask(target, feature string) Answer {
	if a := Ask(target, feature); a.Measured {
		return a
	}
	t, err := r.Resolve(target)
	if err != nil {
		return Answer{Target: target, Feature: feature}
	}
	f, ok := t.Features[feature]
	if !ok || f.Source != Probed {
		return Answer{Target: target, Feature: feature}
	}
	a := Answer{Target: target, Feature: feature, Supported: f.Supported, Measured: true, Diagnostic: f.Diagnostic}
	if t.Probe != nil {
		a.Source = fmt.Sprintf("%s %s (%s, %s)", t.Probe.Tool, t.Probe.ToolVersion, t.Probe.Host, t.Probe.ProbedAt)
	}
	return a
}

// Resolve probes every embedded fixture for one target and returns the record,
// caching it. The record is built by the same [BuildTargets] path the
// generator uses, so a resolved target is subject to the same provenance rules
// as a checked-in one -- including the positive control, which is why Resolve
// probes the whole fixture set rather than the one feature asked about.
func (r *Resolver) Resolve(target string) (Target, error) {
	if !targetPattern.MatchString(target) {
		return Target{}, fmt.Errorf("cudadevicetable: %q is not an exact target such as \"sm_90a\"", target)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if t, ok := r.cache[target]; ok {
		return t, nil
	}
	ptxVersion, err := FixtureVersion()
	if err != nil {
		return Target{}, err
	}
	probe, err := NewFeatureProbe(r.ptxas, r.host, ptxVersion)
	if err != nil {
		return Target{}, err
	}
	results, err := ProbeTargets(r.ptxas, []string{target})
	if err != nil {
		return Target{}, err
	}
	built, err := BuildTargets(results, probe)
	if err != nil {
		return Target{}, fmt.Errorf("cudadevicetable: resolve %s: %w", target, err)
	}
	t, ok := built[target]
	if !ok {
		return Target{}, fmt.Errorf("cudadevicetable: resolve %s: no conclusive result", target)
	}
	r.cache[target] = t
	return t, nil
}

// Resolved returns the targets this Resolver has measured, which are exactly
// the ones absent from the checked-in table. Feeding them to the generator is
// how a resolved answer becomes a recorded one.
func (r *Resolver) Resolved() map[string]Target {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[string]Target, len(r.cache))
	for k, v := range r.cache {
		out[k] = v
	}
	return out
}
