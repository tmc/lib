package cudadiageval

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
)

// AnalysisResult binds separately produced analysis output to original arm
// evidence. Its canonical analysis-result-v1 bytes establish a relation only;
// independent execution, statistical validity and acquisition require policy
// acceptance. It contains no enclosing scope, pair or acceptance digest.
type AnalysisResult struct {
	Version             int                    `json:"version"`
	Profile             string                 `json:"profile"`
	RegistrationDigest  string                 `json:"registration_digest"`
	ProtocolDigest      string                 `json:"protocol_digest"`
	Arm                 string                 `json:"arm"`
	SelectedCell        SelectedCellIdentity   `json:"selected_cell"`
	OriginalToolReceipt RegistrationArtifact   `json:"original_tool_receipt"`
	MethodArtifact      RegistrationArtifact   `json:"method_artifact"`
	Population          []Population           `json:"population"`
	Scope               AnalysisResultScope    `json:"scope"`
	OutputArtifacts     []RegistrationArtifact `json:"output_artifacts"`
}

// AnalysisResultScope is the semantic scope of an analysis result. It omits
// ScopeID, artifact digests and enclosing identities to keep result and scope
// preimages acyclic. An empty Property denotes the registered property set;
// Properties still binds every exact unit, denominator and timing method.
type AnalysisResultScope struct {
	Role             string                     `json:"role"`
	TargetID         string                     `json:"target_id"`
	SelectedTargets  []string                   `json:"selected_targets"`
	Property         string                     `json:"property"`
	Unit             string                     `json:"unit"`
	Denominator      uint64                     `json:"denominator"`
	PopulationFamily string                     `json:"population_family"`
	TimingFamily     string                     `json:"timing_family"`
	ClockDomain      string                     `json:"clock_domain"`
	ClockEpoch       string                     `json:"clock_epoch"`
	RangeStart       int64                      `json:"range_start"`
	RangeEnd         int64                      `json:"range_end"`
	ExperimentPhase  string                     `json:"experiment_phase"`
	Properties       []RegistrationPropertySpec `json:"properties"`
	ClockDomains     []RegistrationClockDomain  `json:"clock_domains"`
}

func analysisResultScope(r Registration, v RegisteredReceipt, scope EvidenceScope) (AnalysisResultScope, error) {
	// These selector fields must also match a library-derived required role key.
	// Scalar metric and clock facts below come only from the registered plan
	// and checked original arm, never from a declaration's asserted values.
	out := AnalysisResultScope{Role: scope.Role, TargetID: scope.TargetID, Property: scope.Property, PopulationFamily: scope.PopulationFamily, ExperimentPhase: scope.ExperimentPhase}
	if r.Analysis.Body == nil || r.Clock.Body == nil {
		return out, errors.New("analysis result: registered semantic scope unresolved")
	}
	for _, target := range r.TargetRows {
		if target.Applicability == "selected" {
			out.SelectedTargets = append(out.SelectedTargets, target.ID)
		}
	}
	sort.Strings(out.SelectedTargets)
	clocks := map[string]bool{}
	for _, property := range r.Analysis.Body.Properties {
		if scope.Property != "" && property.Property != scope.Property {
			continue
		}
		out.Properties = append(out.Properties, property)
		if scope.Property != "" {
			out.Unit = property.Unit
			out.Denominator = property.Denominator
			out.TimingFamily = property.TimingFamily
			out.ClockDomain = property.ClockDomain
		}
		if property.ClockDomain != "" {
			clocks[property.ClockDomain] = true
		}
	}
	if len(out.Properties) == 0 {
		return out, errors.New("analysis result: registered property scope missing")
	}
	for _, domain := range r.Clock.Body.Domains {
		if clocks[domain.ID] {
			out.ClockDomains = append(out.ClockDomains, domain)
			delete(clocks, domain.ID)
		}
	}
	if len(clocks) != 0 {
		return out, errors.New("analysis result: registered clock scope missing")
	}
	sort.Slice(out.ClockDomains, func(i, j int) bool { return out.ClockDomains[i].ID < out.ClockDomains[j].ID })
	if out.ClockDomain != "" {
		found := false
		for _, domain := range out.ClockDomains {
			if domain.ID != out.ClockDomain {
				continue
			}
			for _, bounds := range v.AcquisitionBounds {
				if bounds.Domain != domain.ID {
					continue
				}
				if found || bounds.Family != domain.Family || bounds.Epoch != domain.Epoch || bounds.Unit != domain.Unit || bounds.Start > bounds.End || bounds.Start < domain.ValidityStart || bounds.End > domain.ValidityEnd {
					return out, errors.New("analysis result: original clock scope mismatch")
				}
				found = true
				out.ClockEpoch = domain.Epoch
				out.RangeStart = bounds.Start
				out.RangeEnd = bounds.End
			}
		}
		if !found {
			return out, errors.New("analysis result: original clock scope missing")
		}
	}
	return out, nil
}

func (s *registeredPair) validateAnalysisResult(v RegisteredReceipt, declaration ExternalAttestation, data map[string][]byte) error {
	check := declaration.Analysis
	if check == nil {
		return errors.New("analysis result: analysis body missing")
	}
	if !requiredRoleKeys(s)[bindingKey(declaration.Scope)] {
		return errors.New("analysis result: unregistered role selector")
	}
	scope, err := analysisResultScope(s.registration, v, declaration.Scope)
	if err != nil {
		return err
	}
	// Aggregate scopes have no scalar metric or clock value. Their explicit
	// Properties and ClockDomains retain registered unknown or known facts.
	actual := scope
	actual.Unit = declaration.Scope.Unit
	actual.Denominator = declaration.Scope.Denominator
	actual.TimingFamily = declaration.Scope.TimingFamily
	actual.ClockDomain = declaration.Scope.ClockDomain
	actual.ClockEpoch = declaration.Scope.ClockEpoch
	actual.RangeStart = declaration.Scope.RangeStart
	actual.RangeEnd = declaration.Scope.RangeEnd
	if !reflect.DeepEqual(actual, scope) {
		return errors.New("analysis result: declared semantic scope mismatch")
	}
	ref := check.ResultArtifact
	if ref == v.ToolReceipt {
		return nil
	}
	raw := data[ref.Path]
	if len(raw) > registrationLimit {
		return errors.New("analysis result: record exceeds 4 MiB limit")
	}
	var result AnalysisResult
	if err := pairedDecode(raw, &result, true); err != nil {
		return fmt.Errorf("analysis result: %w", err)
	}
	if result.Version != 1 || result.Profile != "analysis-result-v1" {
		return errors.New("analysis result: unsupported profile or version")
	}
	if result.RegistrationDigest != s.registration.Digest || result.ProtocolDigest != s.registration.Protocol.FrozenDigest || result.Arm != v.Arm {
		return errors.New("analysis result: original arm or registration mismatch")
	}
	var cell SelectedCellIdentity
	for _, selected := range selectedIdentities(s.registration, s.ledger) {
		if selected.CellID == v.CellID {
			cell = selected
		}
	}
	if cell.CellID == "" || !reflect.DeepEqual(result.SelectedCell, cell) {
		return errors.New("analysis result: selected cell identity mismatch")
	}
	if result.OriginalToolReceipt != v.ToolReceipt || result.MethodArtifact != s.registration.Analysis.Body.Method || result.MethodArtifact != declaration.MethodArtifact {
		return errors.New("analysis result: original receipt or method mismatch")
	}
	if !reflect.DeepEqual(result.Population, v.Population) {
		return errors.New("analysis result: original population mismatch")
	}
	if !reflect.DeepEqual(result.Scope, scope) {
		return errors.New("analysis result: registered semantic scope mismatch")
	}
	if len(result.OutputArtifacts) == 0 {
		return errors.New("analysis result: independent output references missing")
	}
	// Enclosing declarations are separate from all underlying result preimages.
	forbiddenPaths, forbiddenDigests := map[string]bool{ref.Path: true}, map[string]bool{ref.SHA256: true}
	for _, arm := range []RegisteredReceipt{s.candidate, s.comparator} {
		for _, binding := range arm.EvidenceBindings {
			if binding.DecoderProfile == "external-attestation-v1" {
				for _, artifact := range binding.Artifacts {
					forbiddenPaths[artifact.Path] = true
					forbiddenDigests[artifact.SHA256] = true
				}
			}
		}
	}
	refs := append([]RegistrationArtifact{result.OriginalToolReceipt, result.MethodArtifact}, result.OutputArtifacts...)
	seenOutputs := map[string]bool{}
	for i, artifact := range refs {
		if forbiddenPaths[artifact.Path] || forbiddenDigests[artifact.SHA256] {
			return errors.New("analysis result: self or enclosing reference")
		}
		if i >= 2 {
			if seenOutputs[artifact.Path] {
				return errors.New("analysis result: duplicate or conflicting output reference")
			}
			seenOutputs[artifact.Path] = true
		}
		if err := registrationArtifacts([]RegistrationArtifact{artifact}); err != nil {
			return fmt.Errorf("analysis result: %w", err)
		}
		if !checkedReference(artifact, data) || !hasDigest(declaration.Scope.ArtifactDigests, artifact.SHA256) {
			return errors.New("analysis result: unchecked or out-of-scope reference")
		}
	}
	return nil
}
