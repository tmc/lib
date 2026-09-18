package cudadevicetable

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Feature is one instruction-set fact: whether the assembler accepts a
// construct for an exact target.
//
// Feature facts are keyed by target rather than by architecture, unlike the
// [Architecture] attributes, because the feature suffix is the whole point.
// setmaxnreg is accepted at sm_90a and rejected at sm_90, which is the same
// silicon; the suffix selects an instruction set, so a table that normalized
// it away would have to pick one of two contradictory answers.
type Feature struct {
	// Supported is what the instrument reported, not what a datasheet says.
	Supported bool `json:"supported"`

	Source Source `json:"source"`

	// Instrument names the assembler invocation that produced the answer,
	// for example "ptxas -arch=sm_90a". Required when Source is [Probed],
	// empty otherwise.
	Instrument string `json:"instrument,omitempty"`

	// Fixture names the file in ptxprobe/ that was assembled. Required when
	// Source is [Probed]: without it the claim cannot be re-run, and a
	// feature fact that cannot be re-run is a datasheet figure wearing a
	// measurement's label.
	Fixture string `json:"fixture,omitempty"`

	// Diagnostic is the assembler's first error line for an unsupported
	// feature. It is what distinguishes a target rejection from a fixture
	// that never assembled anywhere.
	Diagnostic string `json:"diagnostic,omitempty"`

	// Note explains why a fact could not be probed. Required when Source is
	// [Unverified].
	Note string `json:"note,omitempty"`
}

// FeatureProbe identifies the assembler a target's feature facts were read
// with. A feature answer is a property of a toolchain version as much as of
// the hardware: a construct rejected by one ptxas can be accepted by the next.
type FeatureProbe struct {
	Tool        string `json:"tool"`
	ToolVersion string `json:"toolVersion"`
	PTXVersion  string `json:"ptxVersion"`
	Host        string `json:"host"`
	ProbedAt    string `json:"probedAt"`
}

// Target is the set of feature facts for one exact PTX target.
type Target struct {
	Target string `json:"target"`

	// Probe is nil when no feature in Features is [Probed].
	Probe *FeatureProbe `json:"probe,omitempty"`

	Features map[string]Feature `json:"features"`
}

var targetPattern = regexp.MustCompile(`^sm_[0-9]+[a-z]*$`)

// LookupTarget returns the feature record for an exact target such as
// "sm_90a". Unlike [Lookup] it does not drop the feature suffix, and it does
// not fall back to the base target: sm_121a's answers are not sm_121's.
func LookupTarget(target string) (Target, bool) {
	t, ok := table.Targets[strings.ToLower(strings.TrimSpace(target))]
	return t, ok
}

// Targets returns the probed target keys in sorted order.
func Targets() []string {
	keys := make([]string, 0, len(table.Targets))
	for k := range table.Targets {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Supports reports whether the target accepts the named feature. It returns
// an error rather than false when the fact was never probed, so that an
// absent measurement cannot be read as a negative one. "No instrument has
// looked" and "the instrument said no" are different answers, and a bool
// cannot hold both.
func (t Target) Supports(feature string) (bool, error) {
	f, ok := t.Features[feature]
	if !ok {
		return false, fmt.Errorf("cudadevicetable: %s: no feature %q", t.Target, feature)
	}
	if f.Source != Probed {
		return false, fmt.Errorf("cudadevicetable: %s: %q is %s, not probed: %s", t.Target, feature, f.Source, f.Note)
	}
	return f.Supported, nil
}

// FeatureNames returns the feature names recorded for any target, sorted.
func FeatureNames() []string {
	seen := map[string]bool{}
	for _, t := range table.Targets {
		for name := range t.Features {
			seen[name] = true
		}
	}
	names := make([]string, 0, len(seen))
	for name := range seen {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// validateTargets enforces the feature-side provenance rules.
func (t Table) validateTargets() error {
	// supportedSomewhere tracks whether each probed feature was accepted by
	// at least one target. See the positive-control rule below.
	supportedSomewhere := map[string]bool{}
	probedAnywhere := map[string]bool{}

	for key, tgt := range t.Targets {
		if !targetPattern.MatchString(key) {
			return fmt.Errorf("target key %q is not an exact target such as \"sm_90a\"", key)
		}
		if tgt.Target != key {
			return fmt.Errorf("target %q records Target %q", key, tgt.Target)
		}
		if len(tgt.Features) == 0 {
			return fmt.Errorf("target %q has no features", key)
		}
		probed := false
		for name, f := range tgt.Features {
			switch f.Source {
			case Probed:
				probed = true
				probedAnywhere[name] = true
				if f.Supported {
					supportedSomewhere[name] = true
				}
				if f.Instrument == "" {
					return fmt.Errorf("%s: feature %q is probed but names no instrument", key, name)
				}
				if !strings.HasPrefix(f.Instrument, "ptxas ") {
					return fmt.Errorf("%s: feature %q names instrument %q, want a ptxas invocation", key, name, f.Instrument)
				}
				if !strings.Contains(f.Instrument, key) {
					return fmt.Errorf("%s: feature %q was probed with %q, which does not name this target", key, name, f.Instrument)
				}
				if f.Fixture == "" {
					return fmt.Errorf("%s: feature %q is probed but names no fixture", key, name)
				}
				if !f.Supported && f.Diagnostic == "" {
					return fmt.Errorf("%s: feature %q is probed unsupported but records no diagnostic", key, name)
				}
			case Derived:
				return fmt.Errorf("%s: feature %q is derived; feature facts are probed or unverified", key, name)
			case Unverified:
				if f.Instrument != "" || f.Fixture != "" {
					return fmt.Errorf("%s: feature %q is unverified but names an instrument or fixture", key, name)
				}
				if f.Note == "" {
					return fmt.Errorf("%s: feature %q is unverified with no note explaining why", key, name)
				}
			default:
				return fmt.Errorf("%s: feature %q has source %q, want probed or unverified", key, name, f.Source)
			}
		}
		switch {
		case probed && tgt.Probe == nil:
			return fmt.Errorf("%s: has probed features but no probe record", key)
		case !probed && tgt.Probe != nil:
			return fmt.Errorf("%s: has a probe record but no probed features", key)
		}
		if tgt.Probe != nil {
			for field, s := range map[string]string{
				"tool":        tgt.Probe.Tool,
				"toolVersion": tgt.Probe.ToolVersion,
				"ptxVersion":  tgt.Probe.PTXVersion,
				"host":        tgt.Probe.Host,
				"probedAt":    tgt.Probe.ProbedAt,
			} {
				if s == "" {
					return fmt.Errorf("%s: feature probe record has empty %s", key, field)
				}
			}
		}
	}

	// The positive-control rule. A feature rejected by every target is not a
	// hardware fact; it is a fixture that never assembled, and recording it
	// would turn a broken probe into a row of confident negatives. This is
	// not hypothetical: the wgmma fixture first written for this table used
	// a constant-zero matrix descriptor, which ptxas 13.0.88 segfaults on. A
	// classifier that read a non-zero exit as "unsupported" reported wgmma
	// unsupported at sm_90a, the one target that does support it.
	for name := range probedAnywhere {
		if !supportedSomewhere[name] {
			return fmt.Errorf("feature %q is unsupported at every probed target: its fixture assembles nowhere, so the negatives are not measurements", name)
		}
	}
	return nil
}

// TargetsSupporting returns the probed targets that accept the named feature,
// sorted. It is the answer to "then where does this work", which is the
// question a rejection raises and the one a hand-written error message gets
// wrong as soon as the toolchain moves.
func TargetsSupporting(feature string) []string {
	var out []string
	for name, t := range table.Targets {
		if ok, err := t.Supports(feature); err == nil && ok {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}

// Diagnostic returns the assembler's recorded rejection message for a feature,
// or "" when the feature is supported or was never probed.
func (t Target) Diagnostic(feature string) string {
	return t.Features[feature].Diagnostic
}
