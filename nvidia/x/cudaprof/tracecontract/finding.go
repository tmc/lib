package tracecontract

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"golang.org/x/mod/semver"
	"sort"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

// FindingClass classifies how a diagnostic finding was established.
type FindingClass string

const (
	// FindingObserved indicates a conclusion established directly from complete primary observations.
	FindingObserved FindingClass = "observed"
	// FindingAssociated indicates a conclusion linked through temporal correlation, external linkage, or heuristic attribution.
	FindingAssociated FindingClass = "associated"
	// FindingExperimentallyConfirmed indicates a conclusion validated by an active intervention against an explicit control receipt.
	FindingExperimentallyConfirmed FindingClass = "experimentally_confirmed"
	// FindingUnavailable indicates the finding could not be admitted because required evidence was missing, incomplete, or disqualified.
	FindingUnavailable FindingClass = "unavailable"
)

// Reason codes for admission and refusal.
const (
	// ReasonAdmitted indicates all required evidence was satisfied.
	ReasonAdmitted = "admitted"
	// ReasonMissingReturn indicates a call lacks its matching return/exit event.
	ReasonMissingReturn = "missing_return"
	// ReasonClockDomainMismatch indicates mismatched or incompatible clock domains.
	ReasonClockDomainMismatch = "clock_domain_mismatch"
	// ReasonClockUncertaintyExceeded indicates clock uncertainty exceeds the allowable threshold.
	ReasonClockUncertaintyExceeded = "clock_uncertainty_exceeded"
	// ReasonUnknownLoss indicates unquantified record loss in a required family.
	ReasonUnknownLoss = "unknown_loss"
	// ReasonLossExceeded indicates dropped records exceeded the threshold for this requirement.
	ReasonLossExceeded = "loss_exceeded"
	// ReasonStreamTruncated indicates the record stream ended prematurely.
	ReasonStreamTruncated = "stream_truncated"
	// ReasonCaptureBoundsExceeded indicates interval falls outside the qualified capture bounds.
	ReasonCaptureBoundsExceeded = "capture_bounds_exceeded"
	// ReasonIdentityMismatch indicates artifact digest, process, thread, device, or workload mismatch.
	ReasonIdentityMismatch = "identity_mismatch"
	// ReasonBackendUnsupported indicates required backend is not in the admitted backend list.
	ReasonBackendUnsupported = "backend_unsupported"
	// ReasonBackendVersionMismatch indicates backend version does not satisfy the requirement.
	ReasonBackendVersionMismatch = "backend_version_mismatch"
	// ReasonProbeMissing indicates a required probe was not enabled or present.
	ReasonProbeMissing = "probe_missing"
	// ReasonProbeCoverageInsufficient indicates probe observation ratio is below the required threshold.
	ReasonProbeCoverageInsufficient = "probe_coverage_insufficient"
	// ReasonCorrelationUnmatched indicates required correlation between records was not established.
	ReasonCorrelationUnmatched = "correlation_unmatched"
	// ReasonInterventionMissing indicates experimental confirmation lacks an intervention receipt.
	ReasonInterventionMissing = "intervention_missing"
	// ReasonControlMissing indicates experimental confirmation lacks a control receipt.
	ReasonControlMissing = "control_missing"
	// ReasonControlVerificationFailed indicates control/intervention comparison did not confirm the mechanism.
	ReasonControlVerificationFailed = "control_verification_failed"
	// ReasonEvidenceUnavailable indicates required evidence was not collected or is unassessed.
	ReasonEvidenceUnavailable = "evidence_unavailable"
)

// Interval defines a closed time span [Start, End] in nanoseconds within an identified clock domain.
type Interval struct {
	Start int64  `json:"start_ns"`
	End   int64  `json:"end_ns"`
	Clock string `json:"clock,omitempty"`
}

// Duration returns the elapsed interval span in nanoseconds, or 0 if End < Start.
func (i Interval) Duration() int64 {
	if i.End < i.Start {
		return 0
	}
	return i.End - i.Start
}

// InterventionReceipt references evidence from an active intervention and its paired control.
type InterventionReceipt struct {
	// TestedRule and workload/operation identities scope confirmation. Legacy
	// receipts lacking these fields cannot confirm a workload-specific mechanism.
	TestedRule              string `json:"tested_rule,omitempty"`
	WorkloadSHA256          string `json:"workload_sha256,omitempty"`
	OperationID             string `json:"operation_id,omitempty"`
	InterventionOperationID string `json:"intervention_operation_id,omitempty"`
	InterventionSHA256      string `json:"intervention_sha256"`
	ControlSHA256           string `json:"control_sha256"`
	InterventionName        string `json:"intervention_name"`
	ControlName             string `json:"control_name"`
	Outcome                 string `json:"outcome"`
	Verified                bool   `json:"verified"`
}

// Requirement defines one rule-specific evidence prerequisite.
type Requirement struct {
	// ReceiptGate names an independent qualification gate required by this rule.
	ReceiptGate        string  `json:"receipt_gate,omitempty"`
	Name               string  `json:"name"`
	Description        string  `json:"description,omitempty"`
	Family             string  `json:"family,omitempty"`
	Backend            string  `json:"backend,omitempty"`
	MinVersion         string  `json:"min_version,omitempty"`
	MaxLoss            uint64  `json:"max_loss,omitempty"`
	AllowUnknownLoss   bool    `json:"allow_unknown_loss,omitempty"`
	Probe              string  `json:"probe,omitempty"`
	MinProbeCoverage   float64 `json:"min_probe_coverage,omitempty"`
	ClockDomain        string  `json:"clock_domain,omitempty"`
	MaxUncertaintyNS   int64   `json:"max_uncertainty_ns,omitempty"`
	RequireReturn      bool    `json:"require_return,omitempty"`
	RequireCorrelation bool    `json:"require_correlation,omitempty"`
	RequireControl     bool    `json:"require_control,omitempty"`
	ArtifactDigest     string  `json:"artifact_digest,omitempty"`
}

// Admission records the evaluated qualification of evidence for a finding.
type Admission struct {
	Admitted   bool            `json:"admitted"`
	Status     string          `json:"status"` // "admitted" or "withheld"
	ReasonCode string          `json:"reason_code"`
	Reason     string          `json:"reason"`
	Gates      map[string]Gate `json:"gates"`
}

// Finding represents a deterministic diagnostic finding whose validity is
// computed from the evidence it actually requires.
type Finding struct {
	ID           string               `json:"id"`
	Rule         string               `json:"rule"`
	Class        FindingClass         `json:"class"`
	Admission    Admission            `json:"admission"`
	Artifacts    []string             `json:"artifacts"`
	Intervals    []Interval           `json:"intervals,omitempty"`
	Requirements []Requirement        `json:"requirements"`
	Message      string               `json:"message"`
	Value        *float64             `json:"value,omitempty"`
	Unit         string               `json:"unit,omitempty"`
	Intervention *InterventionReceipt `json:"intervention,omitempty"`
}

// EvidenceContext provides the surrounding trace receipt, raw artifacts, and
// runtime observations against which finding requirements are evaluated.
type EvidenceContext struct {
	Receipt      Receipt
	Artifacts    []string
	Intervals    []Interval
	HasReturn    bool
	ClockDomain  string
	Intervention *InterventionReceipt
	Correlation  *CorrelationOutcome
}

// EvaluateRequirements evaluates a set of requirements deterministically against the provided evidence.
func EvaluateRequirements(reqs []Requirement, ev EvidenceContext) Admission {
	sortedReqs := append([]Requirement(nil), reqs...)
	sort.Slice(sortedReqs, func(i, j int) bool { return sortedReqs[i].Name < sortedReqs[j].Name })

	gates := make(map[string]Gate, len(sortedReqs))
	firstFailCode := ""
	firstFailReason := ""

	for _, req := range sortedReqs {
		gate, code, failReason := evaluateOneRequirement(req, ev)
		if _, duplicate := gates[req.Name]; duplicate {
			gate = Gate{Disposition: "failed", Reason: "duplicate requirement name", Provenance: "requirement definition"}
			code, failReason = ReasonEvidenceUnavailable, gate.Reason
		}
		gates[req.Name] = gate
		if gate.Disposition != "passed" && firstFailCode == "" {
			firstFailCode = code
			firstFailReason = failReason
		}
	}

	if firstFailCode != "" {
		return Admission{
			Admitted:   false,
			Status:     "withheld",
			ReasonCode: firstFailCode,
			Reason:     firstFailReason,
			Gates:      gates,
		}
	}

	return Admission{
		Admitted:   true,
		Status:     "admitted",
		ReasonCode: ReasonAdmitted,
		Reason:     "all required evidence satisfied",
		Gates:      gates,
	}
}

func evaluateOneRequirement(req Requirement, ev EvidenceContext) (Gate, string, string) {
	refuse := func(code, reason string) (Gate, string, string) {
		return Gate{Disposition: "failed", Reason: reason, Provenance: "evaluated evidence"}, code, reason
	}
	if req.Name == "" || req.MaxUncertaintyNS < 0 || math.IsNaN(req.MinProbeCoverage) || math.IsInf(req.MinProbeCoverage, 0) || req.MinProbeCoverage < 0 || req.MinProbeCoverage > 1 {
		return refuse(ReasonEvidenceUnavailable, "invalid requirement")
	}
	if ev.Receipt.ClockUncertaintyNS != nil && *ev.Receipt.ClockUncertaintyNS < 0 {
		return refuse(ReasonEvidenceUnavailable, "negative clock uncertainty")
	}
	for _, iv := range ev.Intervals {
		if iv.End < iv.Start {
			return refuse(ReasonCaptureBoundsExceeded, "inverted evidence interval")
		}
		if ev.Receipt.CaptureBounds != nil {
			bounds := ev.Receipt.CaptureBounds
			if bounds.End < bounds.Start {
				return refuse(ReasonCaptureBoundsExceeded, "inverted capture bounds")
			}
			if (req.ClockDomain != "" || req.MaxUncertaintyNS > 0) && bounds.Clock != "" && iv.Clock != bounds.Clock {
				return refuse(ReasonClockDomainMismatch, "interval and capture clocks differ")
			}
		}
		if req.ClockDomain != "" && iv.Clock != req.ClockDomain {
			return refuse(ReasonClockDomainMismatch, "interval clock does not match requirement")
		}
	}

	if req.ReceiptGate != "" {
		gate, ok := ev.Receipt.Gates[req.ReceiptGate]
		if !ok || gate.Disposition != "passed" {
			return refuse(ReasonEvidenceUnavailable, "required receipt gate not passed: "+req.ReceiptGate)
		}
	}

	if req.RequireReturn && !ev.HasReturn {
		return Gate{
			Disposition: "failed",
			Reason:      "call return event is missing",
			Provenance:  "trace records",
		}, ReasonMissingReturn, "call return event is missing"
	}

	if req.Family != "" {
		loss, ok := ev.Receipt.LossFor(req.Family)
		if !ok {
			return Gate{
				Disposition: "unavailable",
				Reason:      fmt.Sprintf("loss unassessed for family %q", req.Family),
				Provenance:  "trace receipt",
			}, ReasonEvidenceUnavailable, fmt.Sprintf("loss unassessed for family %q", req.Family)
		}
		if loss.Unknown && !req.AllowUnknownLoss {
			return Gate{
				Disposition: "failed",
				Reason:      fmt.Sprintf("unknown loss detected in family %q", req.Family),
				Provenance:  "trace receipt",
			}, ReasonUnknownLoss, fmt.Sprintf("unknown loss in family %q", req.Family)
		}
		if loss.Dropped > req.MaxLoss {
			return Gate{
				Disposition: "failed",
				Reason:      fmt.Sprintf("dropped records (%d) exceed max allowed (%d) in family %q", loss.Dropped, req.MaxLoss, req.Family),
				Provenance:  "trace receipt",
			}, ReasonLossExceeded, fmt.Sprintf("loss exceeded in family %q", req.Family)
		}
	}

	if req.Backend != "" {
		found := false
		var backendVer string
		for _, b := range ev.Receipt.Backends {
			if b.Name == req.Backend {
				found = true
				backendVer = b.Version
				break
			}
		}
		if !found {
			return Gate{
				Disposition: "unsupported",
				Reason:      fmt.Sprintf("required backend %q not present in receipt", req.Backend),
				Provenance:  "trace receipt",
			}, ReasonBackendUnsupported, fmt.Sprintf("backend %q unsupported", req.Backend)
		}
		if req.MinVersion != "" && (!semver.IsValid("v"+strings.TrimPrefix(backendVer, "v")) || !semver.IsValid("v"+strings.TrimPrefix(req.MinVersion, "v")) || semver.Compare("v"+strings.TrimPrefix(backendVer, "v"), "v"+strings.TrimPrefix(req.MinVersion, "v")) < 0) {
			return Gate{
				Disposition: "unsupported",
				Reason:      fmt.Sprintf("backend %q version %q below required %q", req.Backend, backendVer, req.MinVersion),
				Provenance:  "trace receipt",
			}, ReasonBackendVersionMismatch, fmt.Sprintf("backend %q version mismatch", req.Backend)
		}
	}

	if req.Probe != "" {
		cov, ok := ev.Receipt.ProbeCoverageFor(req.Probe)
		if !ok || !cov.Present {
			return Gate{
				Disposition: "unavailable",
				Reason:      fmt.Sprintf("required probe %q missing or inactive", req.Probe),
				Provenance:  "trace receipt",
			}, ReasonProbeMissing, fmt.Sprintf("probe %q missing", req.Probe)
		}
		if math.IsNaN(cov.Ratio) || math.IsInf(cov.Ratio, 0) || cov.Ratio < 0 || cov.Ratio > 1 || cov.Ratio < req.MinProbeCoverage {
			return Gate{
				Disposition: "failed",
				Reason:      fmt.Sprintf("probe %q coverage ratio %.2f below required %.2f", req.Probe, cov.Ratio, req.MinProbeCoverage),
				Provenance:  "trace receipt",
			}, ReasonProbeCoverageInsufficient, fmt.Sprintf("probe %q coverage insufficient", req.Probe)
		}
	}

	if req.ClockDomain != "" {
		clock := ev.ClockDomain
		if clock == "" {
			clock = ev.Receipt.ClockDomain
		}
		if clock == "" {
			return Gate{
				Disposition: "unavailable",
				Reason:      "clock domain unstated",
				Provenance:  "trace evidence",
			}, ReasonEvidenceUnavailable, "clock domain unstated"
		}
		if clock != req.ClockDomain {
			return Gate{
				Disposition: "failed",
				Reason:      fmt.Sprintf("clock domain mismatch: got %q, want %q", clock, req.ClockDomain),
				Provenance:  "trace evidence",
			}, ReasonClockDomainMismatch, fmt.Sprintf("clock domain mismatch: got %q, want %q", clock, req.ClockDomain)
		}
	}

	if req.MaxUncertaintyNS > 0 {
		if ev.Receipt.ClockUncertaintyNS == nil {
			return Gate{
				Disposition: "unavailable",
				Reason:      "clock uncertainty unassessed",
				Provenance:  "trace receipt",
			}, ReasonEvidenceUnavailable, "clock uncertainty unassessed"
		}
		if *ev.Receipt.ClockUncertaintyNS > req.MaxUncertaintyNS {
			return Gate{
				Disposition: "failed",
				Reason:      fmt.Sprintf("clock uncertainty %d ns exceeds limit %d ns", *ev.Receipt.ClockUncertaintyNS, req.MaxUncertaintyNS),
				Provenance:  "trace receipt",
			}, ReasonClockUncertaintyExceeded, "clock uncertainty exceeded"
		}
	}

	if req.ArtifactDigest != "" {
		matched := false
		for _, a := range ev.Artifacts {
			if a == req.ArtifactDigest {
				matched = true
				break
			}
		}
		if !matched && ev.Receipt.InputSHA256 == req.ArtifactDigest {
			matched = true
		}
		if !matched {
			return Gate{
				Disposition: "failed",
				Reason:      fmt.Sprintf("artifact digest %q not found in evidence", req.ArtifactDigest),
				Provenance:  "trace evidence",
			}, ReasonIdentityMismatch, fmt.Sprintf("artifact digest %q mismatch", req.ArtifactDigest)
		}
	}

	if req.RequireCorrelation {
		corr := ev.Correlation
		if corr == nil {
			corr = ev.Receipt.Correlation
		}
		if corr == nil {
			return Gate{
				Disposition: "unavailable",
				Reason:      "correlation outcome unassessed",
				Provenance:  "trace receipt",
			}, ReasonEvidenceUnavailable, "correlation unassessed"
		}
		if corr.Unmatched > 0 || corr.Ambiguous > 0 || corr.Missing > 0 || corr.Matched == 0 || corr.Status != "exact" {
			return Gate{
				Disposition: "failed",
				Reason:      fmt.Sprintf("correlation incomplete: %d unmatched, %d ambiguous, %d missing", corr.Unmatched, corr.Ambiguous, corr.Missing),
				Provenance:  "correlation outcome",
			}, ReasonCorrelationUnmatched, "correlation unmatched"
		}
	}

	if req.RequireControl {
		if ev.Intervention == nil {
			return Gate{
				Disposition: "unavailable",
				Reason:      "intervention receipt missing",
				Provenance:  "intervention context",
			}, ReasonInterventionMissing, "intervention receipt missing"
		}
		if !validHash(ev.Intervention.ControlSHA256) || !validHash(ev.Intervention.InterventionSHA256) {
			return Gate{
				Disposition: "failed",
				Reason:      "control receipt missing or unbound",
				Provenance:  "intervention context",
			}, ReasonControlMissing, "control receipt missing"
		}
		if !ev.Intervention.Verified {
			return Gate{
				Disposition: "failed",
				Reason:      "control verification failed",
				Provenance:  "intervention context",
			}, ReasonControlVerificationFailed, "control verification failed"
		}
	}

	if len(ev.Intervals) > 0 && ev.Receipt.CaptureBounds != nil {
		bounds := *ev.Receipt.CaptureBounds
		for _, iv := range ev.Intervals {
			if iv.Start < bounds.Start || iv.End > bounds.End {
				return Gate{
					Disposition: "failed",
					Reason:      fmt.Sprintf("interval [%d, %d] exceeds capture bounds [%d, %d]", iv.Start, iv.End, bounds.Start, bounds.End),
					Provenance:  "capture bounds",
				}, ReasonCaptureBoundsExceeded, "interval exceeds capture bounds"
			}
		}
	}

	return Gate{
		Disposition: "passed",
		Reason:      "requirement satisfied: " + req.Name,
		Provenance:  "evaluated evidence",
	}, "", ""
}

// Admit evaluates candidate requirements against evidence and returns a
// fully attributed Finding. If admission fails, the returned Finding has Class
// FindingUnavailable, Admission.Admitted false, and the stable ReasonCode.
func Admit(id, rule string, targetClass FindingClass, msg string, value *float64, unit string, reqs []Requirement, ev EvidenceContext) Finding {
	if targetClass == FindingExperimentallyConfirmed {
		reqs = append(append([]Requirement(nil), reqs...), Requirement{Name: "experimental_control", RequireControl: true})
	}
	admission := EvaluateRequirements(reqs, ev)
	class := targetClass
	validClass := class == FindingObserved || class == FindingAssociated || class == FindingExperimentallyConfirmed || class == FindingUnavailable
	if !validClass || id == "" || rule == "" || (value != nil && (math.IsNaN(*value) || math.IsInf(*value, 0))) {
		admission = Admission{Status: "withheld", ReasonCode: ReasonEvidenceUnavailable, Reason: "invalid finding identity, class or measurement"}
	}
	if !admission.Admitted {
		class = FindingUnavailable
	}

	sortedReqs := append([]Requirement(nil), reqs...)
	sort.Slice(sortedReqs, func(i, j int) bool { return sortedReqs[i].Name < sortedReqs[j].Name })

	artifacts := append([]string(nil), ev.Artifacts...)
	sort.Strings(artifacts)

	intervals := append([]Interval(nil), ev.Intervals...)
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i].Start != intervals[j].Start {
			return intervals[i].Start < intervals[j].Start
		}
		return intervals[i].End < intervals[j].End
	})

	var valCopy *float64
	if value != nil && !math.IsNaN(*value) && !math.IsInf(*value, 0) {
		v := *value
		valCopy = &v
	}

	var interventionCopy *InterventionReceipt
	if ev.Intervention != nil {
		inv := *ev.Intervention
		interventionCopy = &inv
	}

	return Finding{
		ID:           id,
		Rule:         rule,
		Class:        class,
		Admission:    admission,
		Artifacts:    artifacts,
		Intervals:    intervals,
		Requirements: sortedReqs,
		Message:      msg,
		Value:        valCopy,
		Unit:         unit,
		Intervention: interventionCopy,
	}
}

// CheckClaim verifies that the finding was admitted with the expected class.
// It checks the finding representation, not the authenticity of imported data.
// Imported or modified findings must be recomputed from their source evidence;
// prose summaries are not evidence.
func (f Finding) CheckClaim(expectedClass FindingClass) error {
	if _, err := f.JSON(); err != nil {
		return err
	}
	if !f.Admission.Admitted {
		return fmt.Errorf("finding claim: finding %q is withheld with reason %q", f.ID, f.Admission.ReasonCode)
	}
	if f.Class == FindingUnavailable {
		return fmt.Errorf("finding claim: finding %q is unavailable (%s)", f.ID, f.Admission.ReasonCode)
	}
	if f.Class != expectedClass {
		return fmt.Errorf("finding claim: class mismatch got %q, want %q", f.Class, expectedClass)
	}
	if expectedClass == FindingExperimentallyConfirmed {
		if f.Intervention == nil || !f.Intervention.Verified || f.Intervention.ControlSHA256 == "" {
			return fmt.Errorf("finding claim: experimentally confirmed finding %q lacks verified control receipt", f.ID)
		}
	}
	return nil
}

// JSON validates the finding and returns deterministic JSON.
func (f Finding) JSON() ([]byte, error) {
	if f.ID == "" || f.Rule == "" || f.Class == "" {
		return nil, fmt.Errorf("trace finding: missing id, rule, or class")
	}
	switch f.Class {
	case FindingObserved, FindingAssociated, FindingExperimentallyConfirmed, FindingUnavailable:
	default:
		return nil, fmt.Errorf("trace finding: unsupported class %q", f.Class)
	}
	if !f.Admission.Admitted && f.Class != FindingUnavailable {
		return nil, fmt.Errorf("trace finding: unadmitted finding cannot claim %q", f.Class)
	}
	if f.Class == FindingExperimentallyConfirmed && (f.Intervention == nil || !f.Intervention.Verified) {
		return nil, fmt.Errorf("trace finding: experimentally confirmed finding requires verified intervention receipt")
	}
	return json.Marshal(f)
}

// Write emits one official TrackEvent instant for the finding on the specified track.
// Perfetto args expose debug.cudaprof_finding_* annotations, with complete finding
// JSON in debug.cudaprof_finding_json.
func (f Finding) Write(writer *perfetto.Writer, trackUUID, timestamp uint64) error {
	if writer == nil || trackUUID == 0 {
		return fmt.Errorf("trace finding: missing writer or track")
	}
	data, err := f.JSON()
	if err != nil {
		return err
	}
	annotations := []perfetto.DebugAnnotation{
		text("cudaprof_schema", "cudaprof.finding.v1"),
		number("cudaprof_schema_version", 1),
		text("cudaprof_finding_id", f.ID),
		text("cudaprof_finding_rule", f.Rule),
		text("cudaprof_finding_class", string(f.Class)),
		{Name: "cudaprof_finding_admitted", Type: perfetto.AnnotationTypeBool, BoolValue: f.Admission.Admitted},
		text("cudaprof_finding_reason_code", f.Admission.ReasonCode),
		text("cudaprof_finding_json", string(data)),
	}
	return writer.WriteFlowInstant(trackUUID, timestamp, "cudaprof.finding.v1", "cudaprof.semantic", annotations, nil, nil)
}

// WriteFindings writes a sequence of findings onto a dedicated findings track.
func WriteFindings(writer *perfetto.Writer, trackUUID, timestamp uint64, findings []Finding) error {
	for _, f := range findings {
		if err := f.Write(writer, trackUUID, timestamp); err != nil {
			return err
		}
	}
	return nil
}
