package cudadiageval

import (
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
)

// MeasurementScope freezes the unit and useful-work denominator for a property.
type MeasurementScope struct {
	Unit        string `json:"unit"`
	Denominator uint64 `json:"denominator"`
}

// ComparisonSpec registers a task and two tool arms before measurements exist.
// Values retain the lower-is-better convention of PropertyMeasurement.
type ComparisonSpec struct {
	Task       string                      `json:"task"`
	Candidate  string                      `json:"candidate"`
	Comparator string                      `json:"comparator"`
	Properties map[string]MeasurementScope `json:"properties"`
}

func measuredVersion(version string) bool {
	switch strings.ToLower(strings.TrimSpace(version)) {
	case "", "unknown", "unverified", "unavailable", "synthetic", "latest", "head":
		return false
	}
	return true
}

func (s ComparisonSpec) validate(p Protocol) error {
	if s.Task == "" || s.Candidate == "" || s.Comparator == "" || s.Candidate == s.Comparator || len(s.Properties) == 0 || len(s.Properties) > 4 {
		return fmt.Errorf("%w: incomplete comparison registration", ErrInvalidProtocol)
	}
	tools := make(map[string]ToolSpec)
	for _, tool := range p.Tools {
		tools[tool.Name] = tool
	}
	candidate, cok := tools[s.Candidate]
	comparator, bok := tools[s.Comparator]
	if !cok || !bok || candidate.Support != "measured" || comparator.Support != "measured" || !measuredVersion(candidate.Version) || !measuredVersion(comparator.Version) || candidate.Lineage == comparator.Lineage {
		return fmt.Errorf("%w: comparison requires two independent measured tool arms", ErrInvalidProtocol)
	}
	for property, scope := range s.Properties {
		if _, ok := p.PropertyLimits[property]; !ok || scope.Unit == "" || scope.Denominator == 0 {
			return fmt.Errorf("%w: incomplete comparison measurement scope", ErrInvalidProtocol)
		}
	}
	return nil
}

// MatchedResult records an observed difference between two independently
// retained artifacts. Effect is comparator minus candidate. Uncertainty is
// the sum of the supplied absolute error bounds, not an inferred confidence
// interval. Unknown inputs remain withheld; overlapping bounds report overlap.
type MatchedResult struct {
	Property         string  `json:"property"`
	Unit             string  `json:"unit"`
	Denominator      uint64  `json:"denominator"`
	CandidateDigest  string  `json:"candidate_digest"`
	ComparatorDigest string  `json:"comparator_digest"`
	Status           string  `json:"status"`
	Effect           float64 `json:"effect"`
	Uncertainty      float64 `json:"uncertainty"`
	Reason           string  `json:"reason,omitempty"`
}

// CompareTools verifies both bounded raw receipts against a frozen task, tool
// matrix, units and denominators. It refuses synthetic arms and missing absolute
// uncertainty bounds. This validates receipt consistency; the caller remains
// responsible for authentic acquisition and the stated uncertainty method.
func CompareTools(p Protocol, candidate, comparator ToolReceipt, candidateRaw, comparatorRaw io.Reader) ([]MatchedResult, error) {
	if p.Comparison == nil {
		return nil, errors.New("cudadiageval: comparison is not registered")
	}
	if _, err := CompareTool(p, candidate, candidateRaw); err != nil {
		return nil, fmt.Errorf("candidate: %w", err)
	}
	if _, err := CompareTool(p, comparator, comparatorRaw); err != nil {
		return nil, fmt.Errorf("comparator: %w", err)
	}
	spec := p.Comparison
	if candidate.Synthetic || comparator.Synthetic || candidate.Tool.Name != spec.Candidate || comparator.Tool.Name != spec.Comparator || candidate.Task != spec.Task || comparator.Task != spec.Task {
		return nil, errors.New("cudadiageval: comparison arm or task mismatch")
	}
	measurements := func(receipt ToolReceipt) map[string]PropertyMeasurement {
		values := make(map[string]PropertyMeasurement)
		for _, value := range receipt.Measurements {
			values[value.Property] = value
		}
		return values
	}
	cvalues, bvalues := measurements(candidate), measurements(comparator)
	var results []MatchedResult
	for _, property := range []string{"attribution", "diagnosis", "cost", "reproducibility"} {
		scope, selected := spec.Properties[property]
		if !selected {
			continue
		}
		result := MatchedResult{Property: property, Unit: scope.Unit, Denominator: scope.Denominator, CandidateDigest: candidate.RawDigest, ComparatorDigest: comparator.RawDigest, Status: "withheld", Reason: "missing measured evidence"}
		c, cok := cvalues[property]
		b, bok := bvalues[property]
		if !cok || !bok || c.Status != "measured" || b.Status != "measured" {
			results = append(results, result)
			continue
		}
		if c.Unit != scope.Unit || b.Unit != scope.Unit || c.Denominator != scope.Denominator || b.Denominator != scope.Denominator {
			return nil, errors.New("cudadiageval: comparison measurement scope mismatch")
		}
		if c.Uncertainty == nil || b.Uncertainty == nil {
			return nil, errors.New("cudadiageval: comparison uncertainty is missing")
		}
		result.Effect, result.Uncertainty = b.Value-c.Value, *b.Uncertainty+*c.Uncertainty
		if math.IsInf(result.Effect, 0) || math.IsInf(result.Uncertainty, 0) {
			return nil, errors.New("cudadiageval: comparison arithmetic overflow")
		}
		result.Status, result.Reason = "overlap", "absolute bounds overlap"
		if result.Effect > result.Uncertainty {
			result.Status, result.Reason = "candidate_lower", ""
		}
		if result.Effect < -result.Uncertainty {
			result.Status, result.Reason = "comparator_lower", ""
		}
		results = append(results, result)
	}
	return results, nil
}
