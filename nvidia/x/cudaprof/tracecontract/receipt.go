package tracecontract

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

// Gate preserves one independent evidence disposition. Disposition is passed,
// failed, unavailable, unchecked or unsupported. An unavailable gate must not
// be promoted by another gate's success.
type Gate struct {
	Disposition string `json:"disposition"`
	Reason      string `json:"reason"`
	Provenance  string `json:"provenance"`
}

// Backend describes a backend actually used. Version and SHA256 remain empty
// when unavailable; native processing does not claim an external importer ran.
type Backend struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"`
	SHA256  string `json:"sha256,omitempty"`
}

// ProbeCoverage records observation presence and coverage ratio for a probe.
// Ratio is bounded in [0.0, 1.0].
type ProbeCoverage struct {
	Present bool    `json:"present"`
	Ratio   float64 `json:"ratio"`
}

// Loss records dropped and total records for a record family. Unknown indicates
// unquantified loss where dropped records cannot be bounded.
type Loss struct {
	Dropped uint64 `json:"dropped"`
	Total   uint64 `json:"total"`
	Unknown bool   `json:"unknown,omitempty"`
}

// CaptureBounds defines the closed time interval [Start, End] in nanoseconds.
type CaptureBounds struct {
	Start int64  `json:"start_ns"`
	End   int64  `json:"end_ns"`
	Clock string `json:"clock,omitempty"`
}

// CorrelationOutcome summarizes correlation between submission calls and device activities.
type CorrelationOutcome struct {
	Matched   uint64 `json:"matched"`
	Unmatched uint64 `json:"unmatched"`
	Ambiguous uint64 `json:"ambiguous"`
	Missing   uint64 `json:"missing"`
	Status    string `json:"status,omitempty"`
}

// Receipt is bounded trace-analysis metadata. Gates must contain collector,
// import, workload, stationarity, shape and correctness independently. InputSHA256
// hashes the exact input bytes parsed; Kind is capture, pftrace or comparison_archive.
// Schema and tool versions are retained separately from evidence dispositions.
// The zero value is not a receipt; Version must be 1 and identities supplied.
type Receipt struct {
	Version                  int                      `json:"version"`
	Command                  string                   `json:"command"`
	Options                  map[string]string        `json:"options"`
	InputPath                string                   `json:"input"`
	InputSHA256              string                   `json:"input_sha256"`
	Kind                     string                   `json:"detected_type"`
	ProducerVersion          string                   `json:"producer_version,omitempty"`
	ConsumerVersion          string                   `json:"consumer_version,omitempty"`
	ProducerSchema           string                   `json:"producer_schema,omitempty"`
	Backends                 []Backend                `json:"backends"`
	Gates                    map[string]Gate          `json:"gates"`
	Warnings                 []string                 `json:"warnings"`
	Limits                   map[string]int64         `json:"limits"`
	Partial                  bool                     `json:"partial"`
	ProjectedRows            map[string]int64         `json:"projected_rows_by_kind,omitempty"`
	RawDeviceRecords         map[string]int64         `json:"raw_device_records_by_kind,omitempty"`
	CollectorActivityRecords *uint64                  `json:"collector_activity_records,omitempty"`
	ProbeCoverage            map[string]ProbeCoverage `json:"probe_coverage,omitempty"`
	LostRecords              map[string]Loss          `json:"lost_records_by_family,omitempty"`
	CaptureBounds            *CaptureBounds           `json:"capture_bounds,omitempty"`
	Correlation              *CorrelationOutcome      `json:"correlation_outcome,omitempty"`
	ClockDomain              string                   `json:"clock_domain,omitempty"`
	ClockProvenance          string                   `json:"clock_provenance,omitempty"`
	ClockUncertaintyNS       *int64                   `json:"clock_uncertainty_ns,omitempty"`
	Uncertainty              *float64                 `json:"uncertainty,omitempty"`
}

// JSON validates the receipt and returns deterministic versioned JSON. It
// refuses metadata exceeding 64 KiB; large evidence belongs in source records.
func (r Receipt) JSON() ([]byte, error) {
	if len(r.Options) > 128 || len(r.Backends) > 64 || len(r.Gates) > 32 || len(r.Warnings) > 256 || len(r.Limits) > 128 || len(r.ProbeCoverage) > 64 || len(r.LostRecords) > 64 {
		return nil, fmt.Errorf("trace receipt: metadata collection limit exceeded")
	}
	if r.Version != 1 || r.Command == "" || r.InputPath == "" || len(r.Backends) == 0 || !validHash(r.InputSHA256) {
		return nil, fmt.Errorf("trace receipt: missing identity or unsupported version")
	}
	switch r.Kind {
	case "capture", "pftrace", "comparison_archive":
	default:
		return nil, fmt.Errorf("trace receipt: unsupported input type %q", r.Kind)
	}
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		_, ok := r.Gates[name]
		if !ok {
			return nil, fmt.Errorf("trace receipt: missing %s evidence disposition", name)
		}
	}
	for name, gate := range r.Gates {
		if name == "" || gate.Reason == "" || gate.Provenance == "" {
			return nil, fmt.Errorf("trace receipt: incomplete evidence disposition")
		}
		switch gate.Disposition {
		case "passed", "failed", "unavailable", "unchecked", "unsupported":
		default:
			return nil, fmt.Errorf("trace receipt: invalid %s disposition", name)
		}
	}
	for _, backend := range r.Backends {
		if backend.Name == "" || (backend.SHA256 != "" && !validHash(backend.SHA256)) {
			return nil, fmt.Errorf("trace receipt: invalid backend identity")
		}
	}
	for _, limit := range r.Limits {
		if limit < 0 {
			return nil, fmt.Errorf("trace receipt: negative resource limit")
		}
	}
	for name, cov := range r.ProbeCoverage {
		if name == "" || cov.Ratio < 0 || cov.Ratio > 1.0 || math.IsNaN(cov.Ratio) || math.IsInf(cov.Ratio, 0) {
			return nil, fmt.Errorf("trace receipt: invalid probe coverage for %q", name)
		}
	}
	for family, loss := range r.LostRecords {
		if family == "" || (!loss.Unknown && loss.Total > 0 && loss.Dropped > loss.Total) {
			return nil, fmt.Errorf("trace receipt: invalid loss record for family %q", family)
		}
	}
	if r.CaptureBounds != nil {
		if r.CaptureBounds.End < r.CaptureBounds.Start {
			return nil, fmt.Errorf("trace receipt: inverted capture bounds")
		}
	}
	if r.ClockUncertaintyNS != nil && *r.ClockUncertaintyNS < 0 {
		return nil, fmt.Errorf("trace receipt: negative clock uncertainty")
	}
	if r.Uncertainty != nil && (*r.Uncertainty < 0 || math.IsNaN(*r.Uncertainty) || math.IsInf(*r.Uncertainty, 0)) {
		return nil, fmt.Errorf("trace receipt: invalid uncertainty")
	}
	data, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	if len(data) > 64<<10 {
		return nil, fmt.Errorf("trace receipt: metadata exceeds 65536 bytes")
	}
	return data, nil
}

// LossFor returns the loss record for the named family. If loss was not
// reported in this receipt, ok is false (unassessed).
func (r Receipt) LossFor(family string) (Loss, bool) {
	if r.LostRecords == nil {
		return Loss{}, false
	}
	l, ok := r.LostRecords[family]
	return l, ok
}

// ProbeCoverageFor returns probe coverage for the named probe. If probe
// coverage was not reported, ok is false (unassessed).
func (r Receipt) ProbeCoverageFor(probe string) (ProbeCoverage, bool) {
	if r.ProbeCoverage == nil {
		return ProbeCoverage{}, false
	}
	c, ok := r.ProbeCoverage[probe]
	return c, ok
}

// Write emits one official TrackEvent instant on a caller-registered receipt
// track. cudaprof.receipt_json retains the complete receipt; individual gate
// dispositions are also directly available through Perfetto's args table.
func (r Receipt) Write(writer *perfetto.Writer, uuid, timestamp uint64) error {
	if writer == nil || uuid == 0 {
		return fmt.Errorf("trace receipt: missing writer or track")
	}
	data, err := r.JSON()
	if err != nil {
		return err
	}
	annotations := []perfetto.DebugAnnotation{
		text("cudaprof_schema", "cudaprof.receipt.v1"),
		number("cudaprof_schema_version", 1),
		text("cudaprof_receipt_json", string(data)),
		text("cudaprof_input_sha256", r.InputSHA256),
		text("cudaprof_command", r.Command),
	}
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		annotations = append(annotations, text("cudaprof_gate_"+name, r.Gates[name].Disposition))
	}
	if r.ClockDomain != "" {
		annotations = append(annotations, text("cudaprof_clock_domain", r.ClockDomain))
	}
	if r.ClockProvenance != "" {
		annotations = append(annotations, text("cudaprof_clock_provenance", r.ClockProvenance))
	}
	annotations = append(annotations, perfetto.DebugAnnotation{Name: "cudaprof_partial", Type: perfetto.AnnotationTypeBool, BoolValue: r.Partial})
	return writer.WriteFlowInstant(uuid, timestamp, "cudaprof.receipt.v1", "cudaprof.semantic", annotations, nil, nil)
}
