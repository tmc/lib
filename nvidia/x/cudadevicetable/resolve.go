package cudadevicetable

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// FindPTXAS locates an assembler to resolve unprobed targets with. It returns
// an error rather than a guess when there is none, because a missing
// assembler must degrade to "unmeasured" and not to a fabricated answer.
//
// $PTXAS names an assembler outright. Otherwise the CUDA toolkit installs are
// tried before $PATH, because a distribution package can leave an older ptxas
// earlier on PATH than the toolkit's own -- and an assembler that predates a
// target does not report that it is too old, it reports that the target is
// unsupported. Resolving through PATH first would record that answer as a
// measurement. [cudatool.SearchPaths] lists what is tried.
func FindPTXAS() (string, error) {
	p, err := cudatool.LookTool("ptxas", os.Getenv("PTXAS"))
	if err != nil {
		return "", fmt.Errorf("cudadevicetable: no ptxas: set $PTXAS or put it on $PATH: %w", err)
	}
	return p, nil
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

	mu      sync.Mutex
	cache   map[string]Target
	witness map[string]bool
}

// NewResolver returns a Resolver using the ptxas at path. host is recorded in
// the probe stamp of anything it measures.
func NewResolver(ptxas, host string) *Resolver {
	return &Resolver{ptxas: ptxas, host: host, cache: map[string]Target{}, witness: map[string]bool{}}
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
// caching it. The record is subject to the same provenance rules as a
// checked-in one -- including the positive control, which is why Resolve
// probes the whole fixture set rather than the one feature asked about. Those
// rules are applied to the checked-in table with the new record merged in,
// because the control they contain is a property of a corpus and not of a
// single target.
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
	built, err := buildTargets(results, probe)
	if err != nil {
		return Target{}, fmt.Errorf("cudadevicetable: resolve %s: %w", target, err)
	}
	if err := r.witnessLocked(built[target]); err != nil {
		return Target{}, fmt.Errorf("cudadevicetable: resolve %s: %w", target, err)
	}
	// Validate against the checked-in table plus this target, not against this
	// target alone. The rules are the generator's, and one of them -- that a
	// probed feature must be supported by some target -- is a statement about
	// a population. A single target is a population of one, in which every
	// feature it does not support is a feature supported nowhere, so resolving
	// alone would fail for every target that is not the most capable one the
	// fixtures describe. The union restores the corpus the rule was written
	// for while still subjecting the new record to all of it.
	//
	// The union alone would make the positive control vacuous here, since the
	// checked-in table satisfies it by construction. witnessLocked above is
	// what keeps it live: it re-measures the "somewhere" with this assembler
	// instead of inheriting it from the one that generated the table.
	union := make(map[string]Target, len(table.Targets)+len(built))
	for k, v := range table.Targets {
		union[k] = v
	}
	for k, v := range built {
		union[k] = v
	}
	if err := (Table{Schema: 1, Targets: union}).validateTargets(); err != nil {
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

// witnessLocked re-measures, with this Resolver's assembler, that each feature
// the resolved target rejects is still accepted by some target. It is the
// positive control made current.
//
// The checked-in table already names, for every feature, a target that accepts
// it -- but that evidence was produced by whichever assembler generated the
// table. A fixture that stops assembling under a newer ptxas would otherwise
// turn into a row of confident negatives here, with the stale table vouching
// for it. So for each rejected feature this probes the table's own witness and
// requires it to still accept. Witnesses are cached for the Resolver's life;
// a feature the target supports needs none, since it is its own witness.
//
// r.mu must be held.
func (r *Resolver) witnessLocked(t Target) error {
	fixtures := Fixtures()
	var failed []string
	names := make([]string, 0, len(t.Features))
	for name := range t.Features {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		f := t.Features[name]
		if f.Source != Probed || f.Supported {
			continue
		}
		witnesses := TargetsSupporting(name)
		if len(witnesses) == 0 {
			failed = append(failed, fmt.Sprintf("%s (the table names no target that supports it)", name))
			continue
		}
		w := witnesses[0]
		key := name + " " + w
		ok, cached := r.witness[key]
		if !cached {
			res, err := ProbeFeature(r.ptxas, name, fixtures[name], w)
			if err != nil {
				return fmt.Errorf("witness probe %s at %s: %w", name, w, err)
			}
			ok = res.Outcome == Accepted
			r.witness[key] = ok
		}
		if !ok {
			failed = append(failed, fmt.Sprintf("%s (witness %s rejects it too)", name, w))
		}
	}
	if len(failed) > 0 {
		return fmt.Errorf("fixture no longer assembles under %s: %s; the negatives are not measurements", r.ptxas, strings.Join(failed, ", "))
	}
	return nil
}
