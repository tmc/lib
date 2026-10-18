package cudareg

import (
	"encoding/hex"
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// FactStatus describes the evidence for a register count.
type FactStatus string

const (
	FactUnavailable FactStatus = "unavailable" // no value, not a known zero
	FactUnverified  FactStatus = "unverified"  // a value with missing evidence or a caveat
	FactAvailable   FactStatus = "available"   // expected producer evidence, not runtime proof
)

// Fact is a per-thread register count. Count has meaning unless Status is
// FactUnavailable. Reason is required unless Status is FactAvailable.
type Fact struct {
	Count  uint32
	Status FactStatus
	Reason string
}

// Resources holds caller-supplied facts for one compiled entry. PeakLive is
// a whole-entry analysis result, never a per-role value. Cap is a compiler
// condition, never dynamic ownership. These facts do not authenticate a run.
type Resources struct {
	ImageDigest   string
	Entry         string
	Target        string // exact declared target
	DecodedTarget string // numeric image target; empty when unavailable
	Allocated     Fact
	PeakLive      Fact
	Cap           Fact
	Modeled       Fact
}

// Validate checks identity syntax, numeric target agreement and fact states.
// A decoded target cannot carry a suffix: numeric agreement does not establish
// support for the declared target's feature suffix.
func (r Resources) Validate() error {
	if digest, err := hex.DecodeString(r.ImageDigest); err != nil || len(digest) != 32 {
		return errors.New("register resources: invalid image digest")
	}
	if r.Entry == "" || strings.TrimSpace(r.Entry) != r.Entry || strings.ContainsRune(r.Entry, 0) {
		return errors.New("register resources: invalid entry")
	}
	if !targetPattern.MatchString(r.Target) {
		return errors.New("register resources: invalid target")
	}
	if r.DecodedTarget != "" {
		decoded := strings.TrimPrefix(r.DecodedTarget, "sm_")
		declared := strings.TrimRight(strings.TrimPrefix(r.Target, "sm_"), "abcdefghijklmnopqrstuvwxyz")
		if !strings.HasPrefix(r.DecodedTarget, "sm_") || strings.Trim(decoded, "0123456789") != "" || len(strings.TrimLeft(decoded, "0")) < 2 || strings.TrimLeft(decoded, "0") != strings.TrimLeft(declared, "0") {
			return errors.New("register resources: decoded target is not the declared numeric architecture")
		}
	}
	for _, f := range []struct {
		name string
		fact Fact
	}{{"allocated", r.Allocated}, {"peak live", r.PeakLive}, {"cap", r.Cap}, {"modeled", r.Modeled}} {
		switch f.fact.Status {
		case FactAvailable:
			if f.fact.Count == 0 && (f.name == "allocated" || f.name == "cap") {
				return fmt.Errorf("register resources: zero %s count", f.name)
			}
		case FactUnavailable, FactUnverified:
			if strings.TrimSpace(f.fact.Reason) == "" {
				return fmt.Errorf("register resources: %s has no reason", f.name)
			}
			if f.fact.Status == FactUnavailable && f.fact.Count != 0 {
				return fmt.Errorf("register resources: unavailable %s has a count", f.name)
			}
		default:
			return fmt.Errorf("register resources: invalid %s status %q", f.name, f.fact.Status)
		}
	}
	return nil
}

// PublicationStatus distinguishes a checked protocol from unbound resources.
type PublicationStatus string

const (
	PublicationUnverified PublicationStatus = "unverified"
	ProtocolFeasible      PublicationStatus = "protocol feasible"
)

// Candidate is a detached finite register schedule. ProtocolFeasible means
// Schedule.Validate passed and every initial CompilerAllocated and Dynamic
// count matches available allocation evidence. It proves neither per-role
// liveness nor compiler acceptance, suffix support, residency or execution.
type Candidate struct {
	Schedule       Schedule
	Resources      *Resources
	Status         PublicationStatus
	Reason         string
	TargetEvidence cudadevicetable.Answer
	TargetReason   string // numeric image agreement does not prove suffix support
}

// PublishCandidate checks s before binding r to its initial register budgets.
// It preserves the caller's per-role PeakLive and Cap: global analysis and a
// compiler cap cannot establish per-role liveness or dynamic ownership.
// Nil resources or an empty decoded target leave the candidate unverified.
// It returns no partial candidate on error, never changes inputs, and neither
// enumerates alternatives nor ranks them.
func PublishCandidate(r *Resources, s Schedule) (Candidate, error) {
	if err := s.Validate(); err != nil {
		return Candidate{}, fmt.Errorf("register candidate: %w", err)
	}
	c := Candidate{
		Status: PublicationUnverified, Reason: "compiled resources unavailable",
		TargetEvidence: cudadevicetable.Ask(s.Target.Name, "setmaxnreg"),
		TargetReason:   "numeric decoded target does not establish feature-suffix support",
	}
	if r != nil {
		if err := r.Validate(); err != nil {
			return Candidate{}, err
		}
		if r.Target != s.Target.Name {
			return Candidate{}, errors.New("register candidate: target mismatch")
		}
		c.Reason = "allocation unavailable or unverified"
		if r.Allocated.Status == FactAvailable {
			for role, budget := range s.Groups {
				if budget.CompilerAllocated != r.Allocated.Count || budget.Dynamic != r.Allocated.Count {
					return Candidate{}, fmt.Errorf("register candidate: group %q allocation differs from compiled entry", role)
				}
			}
			c.Status, c.Reason = ProtocolFeasible, ""
		}
		if r.DecodedTarget == "" {
			c.Status, c.Reason = PublicationUnverified, "decoded target unavailable"
		}
		copy := *r
		c.Resources = &copy
	}
	c.Schedule = s
	c.Schedule.Target.LegalCounts = append([]uint32(nil), s.Target.LegalCounts...)
	c.Schedule.Groups = make(map[string]Budget, len(s.Groups))
	for role, budget := range s.Groups {
		c.Schedule.Groups[role] = budget
	}
	c.Schedule.Steps = append([]Transition(nil), s.Steps...)
	return c, nil
}
