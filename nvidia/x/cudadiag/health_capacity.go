package cudadiag

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

const healthCapacityVersion = 1

// HealthCapacityInput contains bounded source bytes and independently supplied
// workload and rule evidence. The caller must not mutate the input during the
// call. Receipt gates and artifact hashes are structural linkage, not proof of
// producer authenticity or authorization.
type HealthCapacityInput struct {
	HealthJSON     []byte                       `json:"health_json"`
	HealthLimits   nvtelemetry.HealthLimits     `json:"health_limits"`
	TelemetryJSONL []byte                       `json:"telemetry_jsonl"`
	Telemetry      TelemetryOptions             `json:"-"`
	HealthFindings []tracecontract.Finding      `json:"health_findings,omitempty"`
	Work           []nvtelemetry.AccountingWork `json:"work,omitempty"`
	ExternalRefs   []HealthCapacityArtifactRef  `json:"external_refs,omitempty"`
}

// HealthCapacityLimits bounds source bytes and record collections. Every
// field must be positive and within its hard cap.
type HealthCapacityLimits struct {
	MaxBytes        int64
	MaxFindings     int
	MaxWork         int
	MaxExternalRefs int
}

// HealthCapacityArtifactRef identifies an opaque artifact owned by an external
// validation or recovery consumer. It does not import, authenticate, or
// authorize the referenced record.
type HealthCapacityArtifactRef struct {
	Kind   string `json:"kind"` // request, authorization, result, action, post_action
	ID     string `json:"id"`
	SHA256 string `json:"sha256"`
	State  string `json:"state"`
}

// HealthCapacityIncident retains source observations, rule findings and
// workload diagnosis as separate records. Impact is only a temporal association
// between admitted evidence; it never establishes health causality or usable
// capacity.
type HealthCapacityIncident struct {
	Version         int                           `json:"version"`
	HealthSHA256    string                        `json:"health_sha256"`
	TelemetrySHA256 string                        `json:"telemetry_sha256"`
	ReceiptSHA256   string                        `json:"receipt_sha256,omitempty"`
	Receipt         *tracecontract.Receipt        `json:"receipt,omitempty"`
	Health          nvtelemetry.HealthSnapshot    `json:"health"`
	HealthFindings  []tracecontract.Finding       `json:"health_findings,omitempty"`
	Diagnosis       TelemetryDiagnosis            `json:"diagnosis"`
	Assignment      nvtelemetry.EntityAssignment  `json:"assignment"`
	Workload        cudaworkload.WorkloadIdentity `json:"workload"`
	Work            []nvtelemetry.AccountingWork  `json:"work,omitempty"`
	Impact          tracecontract.Admission       `json:"impact"`
	ImpactClass     tracecontract.FindingClass    `json:"impact_class"`
	WithheldReason  string                        `json:"withheld_reason,omitempty"`
	ExternalRefs    []HealthCapacityArtifactRef   `json:"external_refs,omitempty"`
}

// ComposeHealthCapacity reads the existing versioned health and canonical
// telemetry formats, then composes only their admitted evidence. Health JSON
// and telemetry JSONL are hashed as exact input bytes. The supplied trace
// receipt remains independent from the stream manifest. An admitted impact
// requires receipt gates named health_identity_binding and health_freshness,
// both passed, and receipt options binding the exact health and assignment
// digests, health node/topology/clock, assignment node/boot/device/MIG/allocation
// identity, workload run, and impact interval. Those caller-supplied values are
// structural linkage, not producer authentication. No hardware, tests,
// scheduler, or remediation action is invoked.
func ComposeHealthCapacity(ctx context.Context, in HealthCapacityInput, limits HealthCapacityLimits) (HealthCapacityIncident, error) {
	var out HealthCapacityIncident
	if ctx == nil {
		return out, fmt.Errorf("health capacity: nil context")
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	if err := limits.validate(); err != nil {
		return out, err
	}
	if len(in.HealthJSON) == 0 || len(in.TelemetryJSONL) == 0 {
		return out, fmt.Errorf("health capacity: missing source bytes")
	}
	healthSize := int64(len(in.HealthJSON))
	if healthSize > limits.MaxBytes || int64(len(in.TelemetryJSONL)) > limits.MaxBytes-healthSize {
		return out, fmt.Errorf("health capacity: source bytes exceed limit")
	}
	if len(in.HealthFindings) > limits.MaxFindings || len(in.Work) > limits.MaxWork || len(in.ExternalRefs) > limits.MaxExternalRefs {
		return out, fmt.Errorf("health capacity: record limit exceeded")
	}
	if in.HealthLimits.MaxBytes <= 0 || in.HealthLimits.MaxBytes > limits.MaxBytes {
		return out, fmt.Errorf("health capacity: invalid health reader byte limit")
	}
	if in.Telemetry.MaxBytes <= 0 || in.Telemetry.MaxBytes > limits.MaxBytes {
		return out, fmt.Errorf("health capacity: invalid telemetry reader byte limit")
	}

	for _, finding := range in.HealthFindings {
		if err := boundHealthFinding(finding); err != nil {
			return out, err
		}
	}
	for _, work := range in.Work {
		if len(work.ID)+len(work.AllocationID)+len(work.Unit)+len(work.Policy)+len(work.Evidence)+len(work.ClockDomain)+len(work.ClockEpoch) > 16<<10 {
			return out, fmt.Errorf("health capacity: work string limit exceeded")
		}
	}
	refs := append([]HealthCapacityArtifactRef(nil), in.ExternalRefs...)
	if err := validateHealthCapacityRefs(refs); err != nil {
		return out, err
	}

	findingsJSON, err := json.Marshal(in.HealthFindings)
	if err != nil {
		return out, fmt.Errorf("health capacity: encode findings: %w", err)
	}
	workJSON, err := json.Marshal(in.Work)
	if err != nil {
		return out, fmt.Errorf("health capacity: encode work evidence: %w", err)
	}
	refsJSON, err := json.Marshal(refs)
	if err != nil {
		return out, fmt.Errorf("health capacity: encode external references: %w", err)
	}
	remaining := limits.MaxBytes - healthSize - int64(len(in.TelemetryJSONL))
	for _, size := range []int64{int64(len(findingsJSON)), int64(len(workJSON)), int64(len(refsJSON))} {
		if size > remaining {
			return out, fmt.Errorf("health capacity: total input bytes exceed limit")
		}
		remaining -= size
	}
	findings, err := cloneJSON(in.HealthFindings)
	if err != nil {
		return out, fmt.Errorf("health capacity: clone findings: %w", err)
	}
	health, err := nvtelemetry.ReadHealthSnapshot(bytes.NewReader(in.HealthJSON), in.HealthLimits)
	if err != nil {
		return out, fmt.Errorf("health capacity: read health snapshot: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	diagnosis, err := DiagnoseTelemetry(ctx, bytes.NewReader(in.TelemetryJSONL), in.Telemetry)
	if err != nil {
		return out, fmt.Errorf("health capacity: diagnose telemetry: %w", err)
	}
	// DiagnoseTelemetry can retain the caller's workload on its withheld path.
	diagnosis.Diagnosis.Workload = diagnosis.Diagnosis.Workload.Clone()
	if err := ctx.Err(); err != nil {
		return out, err
	}

	healthHash := sha256.Sum256(in.HealthJSON)
	telemetryHash := sha256.Sum256(in.TelemetryJSONL)
	assignment, err := cloneJSON(in.Telemetry.Assignment)
	if err != nil {
		return out, fmt.Errorf("health capacity: clone assignment: %w", err)
	}
	var receipt *tracecontract.Receipt
	if in.Telemetry.Receipt != nil {
		copy, err := cloneJSON(*in.Telemetry.Receipt)
		if err != nil {
			return out, fmt.Errorf("health capacity: clone independent receipt: %w", err)
		}
		receipt = &copy
	}
	out = HealthCapacityIncident{
		Version:         healthCapacityVersion,
		HealthSHA256:    hex.EncodeToString(healthHash[:]),
		TelemetrySHA256: hex.EncodeToString(telemetryHash[:]),
		Receipt:         receipt,
		Health:          *health,
		HealthFindings:  findings,
		Diagnosis:       diagnosis,
		Assignment:      assignment,
		Workload:        in.Telemetry.Workload.Clone(),
		Work:            append([]nvtelemetry.AccountingWork(nil), in.Work...),
		ExternalRefs:    refs,
		Impact: tracecontract.Admission{
			Status:     "withheld",
			ReasonCode: tracecontract.ReasonEvidenceUnavailable,
			Reason:     "required health-to-workload evidence is incomplete or mismatched",
		},
		ImpactClass: tracecontract.FindingUnavailable,
	}
	if in.Telemetry.Receipt != nil {
		receiptJSON, err := in.Telemetry.Receipt.JSON()
		if err != nil {
			return HealthCapacityIncident{}, fmt.Errorf("health capacity: validate independent receipt: %w", err)
		}
		receiptHash := sha256.Sum256(receiptJSON)
		out.ReceiptSHA256 = hex.EncodeToString(receiptHash[:])
	}

	if reason := healthCapacityWithheldReason(*health, diagnosis, findings, in.Telemetry, in.Work, out.HealthSHA256); reason != "" {
		out.WithheldReason = reason
		out.Impact.Reason = reason
		return out, nil
	}
	out.Impact = tracecontract.Admission{
		Admitted:   true,
		Status:     "admitted",
		ReasonCode: tracecontract.ReasonAdmitted,
		Reason:     "independent workload degradation and health evidence overlap in an admitted identity and clock window; association only",
	}
	out.ImpactClass = tracecontract.FindingAssociated
	return out, nil
}

func (l HealthCapacityLimits) validate() error {
	if l.MaxBytes <= 0 || l.MaxBytes > 64<<20 || l.MaxFindings <= 0 || l.MaxFindings > 4096 || l.MaxWork <= 0 || l.MaxWork > 4096 || l.MaxExternalRefs <= 0 || l.MaxExternalRefs > 256 {
		return fmt.Errorf("health capacity: limits must be positive and bounded")
	}
	return nil
}

func healthCapacityWithheldReason(health nvtelemetry.HealthSnapshot, diagnosis TelemetryDiagnosis, findings []tracecontract.Finding, telemetry TelemetryOptions, work []nvtelemetry.AccountingWork, healthSHA256 string) string {
	if !health.Complete || health.LostRecords != 0 {
		return "health snapshot is incomplete or reports record loss"
	}
	if !diagnosis.Diagnosis.Admitted {
		return "workload diagnosis is not admitted"
	}
	switch diagnosis.Diagnosis.Mechanism {
	case MechanismHostSubmissionDelay, MechanismDeviceBottleneck, MechanismAmbiguousSimultaneous:
	default:
		return "workload diagnosis contains no admitted degradation finding"
	}
	intervals, duplicate := admittedHealthFindingIntervals(findings, healthSHA256, health.ClockDomain)
	if duplicate {
		return "health rule finding IDs are ambiguous"
	}
	if len(intervals) == 0 {
		return "no admitted health rule finding references the exact health snapshot"
	}
	assignment := telemetry.Assignment
	if err := assignment.Validate(); err != nil || !assignment.Authoritative || assignment.State != "assigned" {
		return "assignment is not structurally valid and authoritative"
	}
	if assignment.Process.Process != telemetry.Workload.Process || assignment.Process.NodeID != health.NodeID || assignment.Process.BootID == "" || assignment.Device.UUID == "" || assignment.Owner.AllocationID == "" {
		return "health, workload, device, allocation, or boot identity mismatch"
	}
	if assignment.ClockDomain != health.ClockDomain || assignment.ClockEpoch != health.ClockEpoch || assignment.ClockEpoch != telemetry.ClockEpoch || telemetry.WallInterval.Clock != health.ClockDomain {
		return "health, assignment, and workload clock domains or epochs differ"
	}
	if !healthIdentityBindingPassed(health, telemetry, healthSHA256) {
		return "independent identity or freshness binding is missing, unchecked, or mismatched"
	}
	if !healthWindowOverlapsWork(health, intervals, work, assignment, diagnosis.Diagnosis.TimeAccounting.WallInterval) {
		return "no correct useful-work record and admitted health finding overlap the same raw health observation and degradation interval"
	}
	return ""
}

func admittedHealthFindingIntervals(findings []tracecontract.Finding, healthSHA256, clock string) ([]tracecontract.Interval, bool) {
	seen := make(map[string]bool)
	intervals := make([]tracecontract.Interval, 0)
	for _, finding := range findings {
		if finding.ID == "" || finding.Rule == "" || !finding.Admission.Admitted || finding.Admission.Status != "admitted" || !contains(finding.Artifacts, healthSHA256) {
			continue
		}
		switch finding.Class {
		case tracecontract.FindingObserved, tracecontract.FindingAssociated, tracecontract.FindingExperimentallyConfirmed:
		default:
			continue
		}
		if seen[finding.ID] {
			return nil, true
		}
		seen[finding.ID] = true
		for _, interval := range finding.Intervals {
			if interval.Clock == clock && interval.End >= interval.Start {
				intervals = append(intervals, interval)
			}
		}
	}
	return intervals, false
}

func boundHealthFinding(finding tracecontract.Finding) error {
	if len(finding.ID) > 4096 || len(finding.Rule) > 4096 || len(finding.Message) > 16<<10 || len(finding.Unit) > 256 || len(finding.Artifacts) > 256 || len(finding.Intervals) > 256 || len(finding.Requirements) > 256 || len(finding.Admission.Gates) > 32 || len(finding.Admission.Reason) > 4096 {
		return fmt.Errorf("health capacity: finding exceeds record limits")
	}
	for _, value := range finding.Artifacts {
		if len(value) > 4096 {
			return fmt.Errorf("health capacity: finding artifact reference exceeds string limit")
		}
	}
	for _, requirement := range finding.Requirements {
		if len(requirement.Name)+len(requirement.Description)+len(requirement.Backend)+len(requirement.ArtifactDigest)+len(requirement.Family)+len(requirement.Probe)+len(requirement.ReceiptGate) > 16<<10 {
			return fmt.Errorf("health capacity: finding requirement exceeds string limit")
		}
	}
	return nil
}

func healthIdentityBindingPassed(health nvtelemetry.HealthSnapshot, telemetry TelemetryOptions, healthSHA256 string) bool {
	if telemetry.Receipt == nil {
		return false
	}
	for _, name := range []string{"health_identity_binding", "health_freshness"} {
		gate, ok := telemetry.Receipt.Gates[name]
		if !ok || gate.Disposition != "passed" || gate.Reason == "" || gate.Provenance == "" {
			return false
		}
	}
	assignmentJSON, err := json.Marshal(telemetry.Assignment)
	if err != nil {
		return false
	}
	assignmentHash := sha256.Sum256(assignmentJSON)
	a := telemetry.Assignment
	options := telemetry.Receipt.Options
	expect := map[string]string{
		"health_sha256":                      healthSHA256,
		"assignment_sha256":                  hex.EncodeToString(assignmentHash[:]),
		"health_node_id":                     health.NodeID,
		"health_topology_epoch":              health.TopologyEpoch,
		"health_clock_domain":                health.ClockDomain,
		"health_clock_epoch":                 health.ClockEpoch,
		"assignment_node_id":                 a.Process.NodeID,
		"assignment_boot_id":                 a.Process.BootID,
		"assignment_pid":                     fmt.Sprint(a.Process.Process.PID),
		"assignment_process_generation":      fmt.Sprint(a.Process.Process.Generation),
		"assignment_source":                  a.Source,
		"assignment_source_revision":         a.SourceRevision,
		"assignment_start_ns":                fmt.Sprint(a.StartNanos),
		"assignment_end_ns":                  fmt.Sprint(a.EndNanos),
		"device_uuid":                        a.Device.UUID,
		"device_mig_scope":                   a.Device.MIGScope,
		"device_mig_parent_uuid":             a.Device.MIGParentUUID,
		"device_mig_instance_id":             fmt.Sprint(a.Device.MIGInstanceID),
		"device_has_mig_instance_id":         fmt.Sprint(a.Device.HasMIGInstanceID),
		"device_mig_compute_instance_id":     fmt.Sprint(a.Device.MIGComputeInstanceID),
		"device_has_mig_compute_instance_id": fmt.Sprint(a.Device.HasMIGComputeInstanceID),
		"mig_configuration_epoch":            a.MIGConfigurationEpoch,
		"allocation_id":                      a.Owner.AllocationID,
		"allocation_attempt":                 a.Owner.Attempt,
		"workload_run_id":                    telemetry.Workload.RunID,
		"impact_clock_domain":                telemetry.WallInterval.Clock,
		"impact_clock_epoch":                 telemetry.ClockEpoch,
		"impact_start_ns":                    fmt.Sprint(telemetry.WallInterval.Start),
		"impact_end_ns":                      fmt.Sprint(telemetry.WallInterval.End),
	}
	for name, value := range expect {
		if options[name] != value {
			return false
		}
	}
	return true
}

func healthWindowOverlapsWork(health nvtelemetry.HealthSnapshot, findings []tracecontract.Interval, work []nvtelemetry.AccountingWork, assignment nvtelemetry.EntityAssignment, impact tracecontract.Interval) bool {
	if len(findings) == 0 || len(work) == 0 || impact.End <= impact.Start {
		return false
	}
	available := make(map[string]bool, len(health.Sources))
	for _, source := range health.Sources {
		available[source.ID] = source.State == "available"
	}
	for _, item := range work {
		if item.ID == "" || item.AllocationID == "" || item.AllocationID != assignment.Owner.AllocationID || item.Units == 0 || item.Unit == "" || !item.Success || !item.Correct || item.Policy == "" || item.Evidence == "" {
			continue
		}
		if item.ClockDomain != health.ClockDomain || item.ClockEpoch != health.ClockEpoch || item.ClockDomain != assignment.ClockDomain || item.ClockEpoch != assignment.ClockEpoch {
			continue
		}
		if item.StartNanos < assignment.StartNanos || item.EndNanos > assignment.EndNanos || item.EndNanos <= item.StartNanos || item.StartNanos > impact.End || item.EndNanos <= impact.Start {
			continue
		}
		for _, observation := range health.Observations {
			if !available[observation.SourceID] || observation.Class != "raw" || observation.Reading.DeviceUUID != assignment.Device.UUID || !observation.Reading.IsAvailable() {
				continue
			}
			reading := observation.Reading
			start, end := reading.TimestampMonotonic, reading.TimestampMonotonic
			if reading.HasSourceTimestamp {
				if reading.SourceTimestamp < 0 || reading.SourceTimestampDomain != health.ClockDomain {
					continue
				}
				start, end = reading.SourceTimestamp, reading.SourceTimestamp
			}
			if reading.HasQueryBounds {
				if reading.QueryEndNanos < reading.QueryStartNanos || (reading.HasSourceTimestamp && (reading.SourceTimestamp < reading.QueryStartNanos || reading.SourceTimestamp > reading.QueryEndNanos)) {
					continue
				}
				if !reading.HasSourceTimestamp {
					start, end = reading.QueryStartNanos, reading.QueryEndNanos
				}
			}
			for _, finding := range findings {
				if finding.Clock != health.ClockDomain || finding.End < finding.Start {
					continue
				}
				commonStart := maxInt64(start, finding.Start, item.StartNanos, impact.Start)
				commonEnd := minInt64(end, finding.End, item.EndNanos, impact.End)
				// Work is half-open at EndNanos; source spans use closed bounds.
				if commonStart <= commonEnd && commonStart < item.EndNanos && commonEnd >= item.StartNanos {
					return true
				}
			}
		}
	}
	return false
}

func maxInt64(values ...int64) int64 {
	value := values[0]
	for _, next := range values[1:] {
		if next > value {
			value = next
		}
	}
	return value
}

func minInt64(values ...int64) int64 {
	value := values[0]
	for _, next := range values[1:] {
		if next < value {
			value = next
		}
	}
	return value
}

func validateHealthCapacityRefs(refs []HealthCapacityArtifactRef) error {
	seen := make(map[string]bool, len(refs))
	for _, ref := range refs {
		if ref.ID == "" || len(ref.ID) > 4096 || !validHealthCapacityHash(ref.SHA256) || ref.State == "" || len(ref.State) > 128 {
			return fmt.Errorf("health capacity: invalid external artifact reference")
		}
		switch ref.Kind {
		case "request", "authorization", "result", "action", "post_action":
		default:
			return fmt.Errorf("health capacity: invalid external artifact kind %q", ref.Kind)
		}
		key := ref.Kind + "\x00" + ref.ID
		if seen[key] {
			return fmt.Errorf("health capacity: duplicate external artifact reference")
		}
		seen[key] = true
	}
	return nil
}

func validHealthCapacityHash(hash string) bool {
	if len(hash) != 64 || strings.ToLower(hash) != hash {
		return false
	}
	_, err := hex.DecodeString(hash)
	return err == nil
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func cloneJSON[T any](value T) (T, error) {
	var out T
	data, err := json.Marshal(value)
	if err != nil {
		return out, err
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return out, err
	}
	return out, nil
}
