package cudadiag

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// StoppingRules specifies pre-registered termination and validity thresholds.
type StoppingRules struct {
	MaxTrials       int     `json:"max_trials"`
	MinEffectSize   float64 `json:"min_effect_size"`  // Minimum required relative improvement (e.g. 0.20 for 20%)
	OracleTolerance float64 `json:"oracle_tolerance"` // Maximum allowable absolute difference in output tensor elements
}

// InterventionPlan defines a paired intervention to verify a hypothesized bottleneck mechanism.
type InterventionPlan struct {
	HypothesisID   string                        `json:"hypothesis_id"`
	TargetWorkload cudaworkload.WorkloadIdentity `json:"target_workload"`
	// ReferenceOutput contains independently computed expected values for both arms.
	ReferenceOutput   []float64         `json:"reference_output"`
	CorrectnessOracle string            `json:"correctness_oracle"`
	InterventionType  string            `json:"intervention_type"`
	Ordering          string            `json:"ordering"`
	DeviceSettings    map[string]string `json:"device_settings,omitempty"`
	CollectorSettings map[string]string `json:"collector_settings,omitempty"`
	StoppingRules     StoppingRules     `json:"stopping_rules"`
}

// Validate checks the structural completeness of the intervention plan.
func (p InterventionPlan) Validate() error {
	if p.HypothesisID != string(MechanismHostSubmissionDelay) && p.HypothesisID != string(MechanismDeviceBottleneck) {
		return errors.New("intervention plan: missing hypothesis ID")
	}
	if p.TargetWorkload.Process.PID == 0 || p.TargetWorkload.Process.Generation == 0 || p.TargetWorkload.RunID == "" || p.Ordering == "" || p.StoppingRules.MaxTrials < 2 || p.StoppingRules.MaxTrials > 65536 {
		return errors.New("intervention plan: incomplete workload or stopping protocol")
	}
	if len(p.ReferenceOutput) == 0 || len(p.ReferenceOutput) > 1<<20 {
		return errors.New("intervention plan: missing or oversized independent reference output")
	}
	for _, value := range p.ReferenceOutput {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return errors.New("intervention plan: nonfinite reference output")
		}
	}
	if p.CorrectnessOracle == "" {
		return errors.New("intervention plan: missing correctness oracle specification")
	}
	if p.InterventionType == "" {
		return errors.New("intervention plan: missing intervention type")
	}
	if !finiteNonnegative(p.StoppingRules.OracleTolerance) || !finiteNonnegative(p.StoppingRules.MinEffectSize) || p.StoppingRules.MinEffectSize > 1 {
		return errors.New("intervention plan: negative oracle tolerance")
	}
	return nil
}

// CheckNumericalCorrectness compares test outputs against control reference outputs.
// It verifies that numerical correctness was preserved under the intervention.
func CheckNumericalCorrectness(controlOut, interventionOut []float64, tolerance float64) error {
	if !finiteNonnegative(tolerance) {
		return errors.New("invalid oracle tolerance")
	}
	if len(controlOut) == 0 || len(interventionOut) == 0 {
		return errors.New("empty output slice for correctness oracle evaluation")
	}
	if len(controlOut) != len(interventionOut) {
		return fmt.Errorf("output length mismatch: control has %d, intervention has %d",
			len(controlOut), len(interventionOut))
	}

	var maxDiff float64
	for i, cVal := range controlOut {
		iVal := interventionOut[i]
		if math.IsNaN(cVal) || math.IsNaN(iVal) {
			return fmt.Errorf("NaN encountered at index %d (control=%v, intervention=%v)", i, cVal, iVal)
		}
		if math.IsInf(cVal, 0) || math.IsInf(iVal, 0) {
			return fmt.Errorf("Inf encountered at index %d (control=%v, intervention=%v)", i, cVal, iVal)
		}
		diff := math.Abs(cVal - iVal)
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tolerance {
			return fmt.Errorf("oracle tolerance exceeded at index %d: diff %g > tolerance %g", i, diff, tolerance)
		}
	}
	return nil
}

// EvaluateIntervention executes verification against the paired intervention plan.
// It checks one supplied pair and its numerical oracle. It does not execute
// trials or verify that the declared ordering and stopping protocol ran.
func EvaluateIntervention(
	plan InterventionPlan,
	control DiagnosisResult,
	controlOut []float64,
	intervention DiagnosisResult,
	interventionOut []float64,
) (tracecontract.InterventionReceipt, error) {
	receipt := tracecontract.InterventionReceipt{
		InterventionName: plan.InterventionType,
		ControlName:      "control_baseline",
		Verified:         false,
	}

	if err := plan.Validate(); err != nil {
		receipt.Outcome = fmt.Sprintf("invalid_plan: %v", err)
		return receipt, err
	}

	if !control.Admitted || !intervention.Admitted || !control.PrimaryFinding.Admission.Admitted || !intervention.PrimaryFinding.Admission.Admitted || workloadDigest(control.Workload) != workloadDigest(plan.TargetWorkload) || workloadDigest(intervention.Workload) != workloadDigest(plan.TargetWorkload) || control.OpID == 0 || intervention.OpID == 0 {
		return receipt, errors.New("intervention: unadmitted or mismatched paired evidence")
	}
	receipt.TestedRule = plan.HypothesisID
	receipt.WorkloadSHA256 = workloadDigest(plan.TargetWorkload)
	receipt.OperationID = control.OpID.String()
	receipt.InterventionOperationID = intervention.OpID.String()

	if err := validInterventionAccounting(control.TimeAccounting); err != nil {
		return receipt, fmt.Errorf("control accounting: %w", err)
	}
	if err := validInterventionAccounting(intervention.TimeAccounting); err != nil {
		return receipt, fmt.Errorf("intervention accounting: %w", err)
	}

	// 1. Enforce numerical correctness oracle
	tol := plan.StoppingRules.OracleTolerance
	if err := CheckNumericalCorrectness(plan.ReferenceOutput, controlOut, tol); err != nil {
		receipt.Outcome = fmt.Sprintf("control_oracle_failed: %v", err)
		return receipt, fmt.Errorf("control failed: %w", err)
	}
	if err := CheckNumericalCorrectness(plan.ReferenceOutput, interventionOut, tol); err != nil {
		receipt.Outcome = fmt.Sprintf("correctness_oracle_failed: %v", err)
		return receipt, fmt.Errorf("intervention failed: %w", err)
	}

	// 2. Measure mechanism effect size
	var ctrlMetric, invMetric float64
	switch plan.HypothesisID {
	case string(MechanismHostSubmissionDelay):
		ctrlMetric = float64(control.TimeAccounting.HostAPIDuration.Nanoseconds())
		invMetric = float64(intervention.TimeAccounting.HostAPIDuration.Nanoseconds())
	case string(MechanismDeviceBottleneck):
		ctrlMetric = float64(control.TimeAccounting.BusyDuration.Nanoseconds())
		invMetric = float64(intervention.TimeAccounting.BusyDuration.Nanoseconds())
	default:
		ctrlMetric = float64(control.TimeAccounting.WallInterval.Duration())
		invMetric = float64(intervention.TimeAccounting.WallInterval.Duration())
	}

	if ctrlMetric <= 0 {
		receipt.Outcome = "zero_or_negative_control_metric"
		return receipt, errors.New("control metric is zero or negative; cannot compute effect size")
	}

	relativeImprovement := (ctrlMetric - invMetric) / ctrlMetric
	minEffect := plan.StoppingRules.MinEffectSize
	if relativeImprovement <= 0 || relativeImprovement < minEffect {
		receipt.Outcome = fmt.Sprintf("insufficient_effect: relative improvement %.2f%% below minimum %.2f%%",
			relativeImprovement*100, minEffect*100)
		return receipt, fmt.Errorf("intervention failed: %s", receipt.Outcome)
	}

	// 3. Compute deterministic hashes for provenance
	ctrlBytes, err := json.Marshal(control)
	if err != nil {
		return receipt, fmt.Errorf("marshal control: %w", err)
	}
	invBytes, err := json.Marshal(intervention)
	if err != nil {
		return receipt, fmt.Errorf("marshal intervention: %w", err)
	}
	ctrlHash := sha256.Sum256(ctrlBytes)
	invHash := sha256.Sum256(invBytes)

	receipt.ControlSHA256 = hex.EncodeToString(ctrlHash[:])
	receipt.InterventionSHA256 = hex.EncodeToString(invHash[:])
	receipt.Outcome = fmt.Sprintf("confirmed: relative improvement %.2f%% with oracle preserved", relativeImprovement*100)
	receipt.Verified = true

	return receipt, nil
}

func finiteNonnegative(v float64) bool { return v >= 0 && !math.IsNaN(v) && !math.IsInf(v, 0) }

func workloadDigest(w cudaworkload.WorkloadIdentity) string {
	data, _ := json.Marshal(w)
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func confirms(in *tracecontract.InterventionReceipt, rule MechanismType, target cudaworkload.WorkloadIdentity, opID cudaworkload.OpID) bool {
	return in != nil && in.Verified && in.TestedRule == string(rule) && in.WorkloadSHA256 == workloadDigest(target) && in.OperationID == opID.String()
}

func validInterventionAccounting(t TimeAccounting) error {
	wall := t.WallInterval
	if wall.Clock == "" || wall.Start < 0 || wall.End < wall.Start {
		return errors.New("invalid wall interval")
	}
	if t.HostAPIDuration < 0 || t.BusyDuration < 0 || t.SummedKernelDuration < 0 || t.RunnableDuration < 0 || t.ScheduledDuration < 0 || t.WaitDuration < 0 {
		return errors.New("negative duration")
	}
	span := time.Duration(wall.Duration())
	if t.BusyDuration > span || t.SummedKernelDuration < t.BusyDuration || t.RunnableDuration > span || t.ScheduledDuration > span || t.WaitDuration > span {
		return errors.New("duration exceeds interval or accounting bounds")
	}
	return nil
}
