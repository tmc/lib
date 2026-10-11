package cudasotaeval

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
)

// ToolSpec freezes a tool's identity, configuration and support status.
// Support is measured, unknown or unavailable; lineage identifies shared implementations.
type ToolSpec struct {
	Name          string `json:"name"`
	Version       string `json:"version"`
	Configuration string `json:"configuration"`
	Lineage       string `json:"lineage"`
	Support       string `json:"support"`
}

// PropertyMeasurement retains an observed metric separately from unavailable evidence.
// Property is attribution, diagnosis, cost or reproducibility. Lower is better.
type PropertyMeasurement struct {
	Unit        string   `json:"unit,omitempty"`
	Denominator uint64   `json:"denominator,omitempty"`
	Uncertainty *float64 `json:"uncertainty,omitempty"`
	Property    string   `json:"property"`
	Status      string   `json:"status"`
	Value       float64  `json:"value"`
	Reason      string   `json:"reason,omitempty"`
}

// ToolReceipt identifies an independently retained raw execution artifact.
// Measurements must come from that artifact, never from simulated defaults.
type ToolReceipt struct {
	Task           string                `json:"task,omitempty"`
	Tool           ToolSpec              `json:"tool"`
	ProtocolDigest string                `json:"protocol_digest"`
	RawDigest      string                `json:"raw_digest"`
	Measurements   []PropertyMeasurement `json:"measurements"`
	Synthetic      bool                  `json:"synthetic"`
}

// ComparisonGate is one independent comparison result.
type ComparisonGate struct {
	Property string `json:"property"`
	Status   string `json:"status"`
	Reason   string `json:"reason,omitempty"`
}

// CompareTool verifies a bounded raw artifact before evaluating preregistered
// thresholds. Each property remains unknown unless measured and supported.
// It makes no claim about independent lineage or superiority over other tools.
func CompareTool(p Protocol, receipt ToolReceipt, raw io.Reader) ([]ComparisonGate, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if !p.Frozen || p.Phase != PhaseQualification {
		return nil, ErrProtocolNotFrozen
	}
	if receipt.ProtocolDigest != p.FrozenDigest {
		return nil, errors.New("cudasotaeval: receipt protocol mismatch")
	}
	var tool *ToolSpec
	for i := range p.Tools {
		if p.Tools[i].Name == receipt.Tool.Name {
			tool = &p.Tools[i]
			break
		}
	}
	if tool == nil || *tool != receipt.Tool {
		return nil, errors.New("cudasotaeval: tool differs from frozen matrix")
	}
	if tool.Support == "measured" && !measuredVersion(tool.Version) {
		return nil, errors.New("cudasotaeval: measured tool has placeholder version")
	}
	if raw == nil {
		return nil, errors.New("cudasotaeval: missing raw artifact")
	}
	data, err := io.ReadAll(io.LimitReader(raw, 16<<20+1))
	n := len(data)
	if err != nil {
		return nil, fmt.Errorf("read artifact: %w", err)
	}
	if n > 16<<20 || fmt.Sprintf("%x", sha256.Sum256(data)) != receipt.RawDigest {
		return nil, errors.New("cudasotaeval: raw artifact size or digest mismatch")
	}
	var observed ToolReceipt
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&observed); err != nil {
		return nil, fmt.Errorf("decode artifact: %w", err)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return nil, errors.New("cudasotaeval: trailing artifact data")
	}
	// The digest is external to the raw artifact, avoiding a recursive hash.
	if observed.RawDigest != "" {
		return nil, errors.New("cudasotaeval: raw artifact embeds digest")
	}
	expected := receipt
	expected.RawDigest = ""
	if !reflect.DeepEqual(observed, expected) {
		return nil, errors.New("cudasotaeval: measurements differ from raw artifact")
	}
	values := make(map[string]PropertyMeasurement)
	for _, m := range receipt.Measurements {
		if _, exists := values[m.Property]; exists {
			return nil, errors.New("cudasotaeval: duplicate property")
		}
		if _, exists := p.PropertyLimits[m.Property]; !exists {
			return nil, errors.New("cudasotaeval: undeclared property")
		}
		if m.Status != "measured" && m.Status != "unknown" && m.Status != "unavailable" && m.Status != "failed" {
			return nil, errors.New("cudasotaeval: invalid measurement status")
		}
		if math.IsNaN(m.Value) || math.IsInf(m.Value, 0) || m.Value < 0 {
			return nil, errors.New("cudasotaeval: invalid measurement")
		}
		if m.Uncertainty != nil && (math.IsNaN(*m.Uncertainty) || math.IsInf(*m.Uncertainty, 0) || *m.Uncertainty < 0) {
			return nil, errors.New("cudasotaeval: invalid measurement uncertainty")
		}
		values[m.Property] = m
	}
	gates := make([]ComparisonGate, 0, 4)
	for _, property := range []string{"attribution", "diagnosis", "cost", "reproducibility"} {
		gate := ComparisonGate{Property: property, Status: "unknown", Reason: "missing measured evidence"}
		m, ok := values[property]
		limit, declared := p.PropertyLimits[property]
		if ok && declared && !receipt.Synthetic && tool.Support == "measured" {
			gate.Status, gate.Reason = m.Status, m.Reason
			if m.Status == "measured" {
				gate.Status = "passed"
				if m.Value > limit {
					gate.Status = "failed"
				}
			}
		}
		gates = append(gates, gate)
	}
	return gates, nil
}

// WriteComparison writes machine-readable gates without omitting unknown cells.
func WriteComparison(w io.Writer, gates []ComparisonGate) error {
	return json.NewEncoder(w).Encode(gates)
}
