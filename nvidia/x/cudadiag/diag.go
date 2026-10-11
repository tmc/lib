package cudadiag

import (
	"fmt"
	"math"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// MechanismType identifies the diagnosed execution bottleneck mechanism.
type MechanismType string

const (
	// MechanismNone indicates no mechanism was diagnosed.
	MechanismNone MechanismType = "none"
	// MechanismNoFault indicates nominal execution within baseline expectations.
	MechanismNoFault MechanismType = "no_fault"
	// MechanismHostSubmissionDelay indicates host launch submission latency or CPU starvation.
	MechanismHostSubmissionDelay MechanismType = "host_submission_delay"
	// MechanismDeviceBottleneck indicates device timing exceeding the baseline.
	// The historical name alone does not establish competition or saturation.
	MechanismDeviceBottleneck MechanismType = "competing_device_bottleneck"
	// MechanismAmbiguousSimultaneous indicates multiple concurrent bottleneck mechanisms.
	MechanismAmbiguousSimultaneous MechanismType = "ambiguous_simultaneous"
	// MechanismWithheld indicates diagnosis was withheld due to incomplete or disqualified evidence.
	MechanismWithheld MechanismType = "withheld"
)

// PowerClockEvidence captures power capping and clock throttling telemetry.
type PowerClockEvidence struct {
	ClockThrottled bool                     `json:"clock_throttled"`
	PowerCapped    bool                     `json:"power_capped"`
	Intervals      []tracecontract.Interval `json:"intervals,omitempty"`
}

// Thresholds defines baseline timings and classification ratios.
type Thresholds struct {
	BaselineKernelDuration  time.Duration
	BaselineHostAPIDuration time.Duration
	HostDelayRatio          float64
	DeviceBottleneckRatio   float64
	MaxUncertaintyNS        int64
	MaxLoss                 uint64
	TargetTID               uint64
}

// DefaultThresholds returns sensible default diagnostic thresholds.
func DefaultThresholds() Thresholds {
	return Thresholds{
		BaselineKernelDuration:  100 * time.Microsecond,
		BaselineHostAPIDuration: 20 * time.Microsecond,
		HostDelayRatio:          2.0,
		DeviceBottleneckRatio:   2.0,
		MaxUncertaintyNS:        50,
		MaxLoss:                 0,
	}
}

// DiagnosticInput encapsulates all primary observations and contracts required for diagnosis.
type DiagnosticInput struct {
	WallInterval tracecontract.Interval
	HostCalls    []HostCallEvent
	Kernels      []KernelEvent
	Sched        SchedEvidence
	PowerClock   PowerClockEvidence
	Receipt      tracecontract.Receipt
	Intervention *tracecontract.InterventionReceipt
	Thresholds   Thresholds
}

// DiagnosisResult records the outcome of an evidence-admitted mechanism diagnosis.
type DiagnosisResult struct {
	Workload       cudaworkload.WorkloadIdentity `json:"workload"`
	OpID           cudaworkload.OpID             `json:"op_id"`
	Mechanism      MechanismType                 `json:"mechanism"`
	TimeAccounting TimeAccounting                `json:"time_accounting"`
	PrimaryFinding tracecontract.Finding         `json:"primary_finding"`
	Findings       []tracecontract.Finding       `json:"findings"`
	Admitted       bool                          `json:"admitted"`
	WithheldReason string                        `json:"withheld_reason,omitempty"`
}

// Diagnose evaluates diagnostic rules deterministically against provided evidence.
// It hides any synthetic injection labels and bases conclusions strictly on admitted evidence.
func Diagnose(target cudaworkload.WorkloadIdentity, opID cudaworkload.OpID, in DiagnosticInput) DiagnosisResult {
	res := DiagnosisResult{
		Workload: target,
		OpID:     opID,
	}

	if target.Process.PID == 0 || target.Process.Generation == 0 || opID == 0 || in.WallInterval.Clock == "" || in.WallInterval.Start < 0 || in.WallInterval.End < in.WallInterval.Start || len(in.HostCalls) == 0 || len(in.Kernels) == 0 {
		return withheldDiagnosis(res, in, "missing target identity or timed host/device evidence")
	}
	if _, err := in.Receipt.JSON(); err != nil {
		return withheldDiagnosis(res, in, "invalid source receipt: "+err.Error())
	}
	thresh := in.Thresholds
	if thresh.BaselineHostAPIDuration < 0 || thresh.BaselineKernelDuration < 0 || thresh.HostDelayRatio < 0 || thresh.DeviceBottleneckRatio < 0 || math.IsNaN(thresh.HostDelayRatio) || math.IsNaN(thresh.DeviceBottleneckRatio) || math.IsInf(thresh.HostDelayRatio, 0) || math.IsInf(thresh.DeviceBottleneckRatio, 0) || thresh.MaxUncertaintyNS < 0 {
		return withheldDiagnosis(res, in, "invalid diagnostic thresholds")
	}
	if thresh.HostDelayRatio <= 0 {
		thresh.HostDelayRatio = 2.0
	}
	if thresh.DeviceBottleneckRatio <= 0 {
		thresh.DeviceBottleneckRatio = 2.0
	}
	if thresh.MaxUncertaintyNS <= 0 {
		thresh.MaxUncertaintyNS = 50
	}
	if float64(thresh.BaselineHostAPIDuration)*thresh.HostDelayRatio >= float64(math.MaxInt64) || float64(thresh.BaselineKernelDuration)*thresh.DeviceBottleneckRatio >= float64(math.MaxInt64) {
		return withheldDiagnosis(res, in, "diagnostic threshold overflow")
	}

	// 1. Check host return completeness: an unmatched uprobe return cannot establish duration.
	allReturnsPresent := true
	for _, call := range in.HostCalls {
		if !call.HasReturn {
			allReturnsPresent = false
			break
		}
	}

	// Build core admission requirements.
	reqs := []tracecontract.Requirement{
		{Name: "comparable_workload", ReceiptGate: "workload"},
		{Name: "comparable_shape", ReceiptGate: "shape"},
		{Name: "stable_baseline", ReceiptGate: "stationarity"},
		{Name: "correctness_oracle", ReceiptGate: "correctness"},
		{
			Name:          "host_return_complete",
			Description:   "host launch calls must have matched return events",
			RequireReturn: true,
		},
		{
			Name:             "clock_uncertainty_bound",
			ClockDomain:      in.WallInterval.Clock,
			Description:      "cross-domain clock uncertainty must be within tolerance",
			MaxUncertaintyNS: thresh.MaxUncertaintyNS,
		},
		{
			Name:        "trace_loss_activity",
			Description: "activity record loss must not exceed tolerance",
			Family:      "activity",
			MaxLoss:     thresh.MaxLoss,
		},
		{
			Name:             "sched_probe_active",
			Description:      "scheduler probe must be active for thread scheduling evidence",
			Probe:            "sched",
			MinProbeCoverage: 1.0,
		},
	}
	if in.Intervention != nil && in.Intervention.Verified {
		reqs = append(reqs, tracecontract.Requirement{
			Name:           "intervention_control_verified",
			Description:    "intervention and paired control must be verified",
			RequireControl: true,
		})
	}

	evCtx := tracecontract.EvidenceContext{
		Receipt:      in.Receipt,
		Intervals:    diagnosticIntervals(in),
		HasReturn:    allReturnsPresent,
		ClockDomain:  in.Receipt.ClockDomain,
		Intervention: in.Intervention,
	}

	ta := ComputeTimeAccounting(in.WallInterval, in.HostCalls, in.Kernels, in.Sched)
	res.TimeAccounting = ta
	if err := validInterventionAccounting(ta); err != nil {
		return withheldDiagnosis(res, in, "invalid time accounting: "+err.Error())
	}

	// Verify target scheduler evidence identity
	targetTID := thresh.TargetTID
	if targetTID == 0 && in.Sched.TID != 0 {
		targetTID = in.Sched.TID
	}

	if err := in.Sched.Validate(target.Process, targetTID); err != nil {
		reason := tracecontract.ReasonIdentityMismatch
		if !in.Sched.ProbePresent {
			reason = tracecontract.ReasonProbeMissing
		} else if in.Sched.UnknownLoss {
			reason = tracecontract.ReasonUnknownLoss
		} else if in.Sched.Loss > 0 {
			reason = tracecontract.ReasonLossExceeded
		}

		finding := tracecontract.Admit(
			fmt.Sprintf("diag-%s-%d", opID.String(), targetTID),
			"scheduler_identity_match",
			tracecontract.FindingUnavailable,
			"target PID/TID scheduler evidence mismatch or missing",
			nil,
			"",
			reqs,
			evCtx,
		)
		finding.Admission.Admitted = false
		finding.Admission.Status = "withheld"
		finding.Admission.ReasonCode = reason
		finding.Admission.Reason = err.Error()

		res.Mechanism = MechanismWithheld
		res.Admitted = false
		res.WithheldReason = reason
		res.PrimaryFinding = finding
		res.Findings = []tracecontract.Finding{finding}
		return res
	}

	// Evaluate requirements through tracecontract
	admission := tracecontract.EvaluateRequirements(reqs, evCtx)
	if !admission.Admitted {
		val := float64(ta.HostAPIDuration.Nanoseconds())
		finding := tracecontract.Admit(
			fmt.Sprintf("diag-%s", opID.String()),
			"mechanism_diagnosis",
			tracecontract.FindingUnavailable,
			fmt.Sprintf("diagnosis withheld: %s", admission.Reason),
			&val,
			"ns",
			reqs,
			evCtx,
		)

		res.Mechanism = MechanismWithheld
		res.Admitted = false
		res.WithheldReason = admission.ReasonCode
		res.PrimaryFinding = finding
		res.Findings = []tracecontract.Finding{finding}
		return res
	}

	res.Admitted = true

	// Check for clock or power throttling events.
	// Rule: Overlapping clock/power events are strictly non-causal associations.
	if in.PowerClock.ClockThrottled || in.PowerClock.PowerCapped {
		powerReqs := []tracecontract.Requirement{
			{Name: "clock_domain", ClockDomain: in.Receipt.ClockDomain},
		}
		desc := "clock throttling or power cap detected during execution interval"
		assocFinding := tracecontract.Admit(
			fmt.Sprintf("assoc-powerclock-%s", opID.String()),
			"power_clock_association",
			tracecontract.FindingAssociated,
			desc+" (non-causal association; does not establish causality without intervention)",
			nil,
			"",
			powerReqs,
			evCtx,
		)
		res.Findings = append(res.Findings, assocFinding)
	}

	// Classify mechanisms from separated time accounting and scheduler evidence
	hostDelayThreshold := time.Duration(float64(thresh.BaselineHostAPIDuration) * thresh.HostDelayRatio)
	if hostDelayThreshold == 0 {
		hostDelayThreshold = 40 * time.Microsecond
	}
	deviceBottleneckThreshold := time.Duration(float64(thresh.BaselineKernelDuration) * thresh.DeviceBottleneckRatio)
	if deviceBottleneckThreshold == 0 {
		deviceBottleneckThreshold = 200 * time.Microsecond
	}

	hasHostDelay := ta.HostAPIDuration >= hostDelayThreshold ||
		ta.RunnableDuration >= hostDelayThreshold ||
		ta.WaitDuration >= hostDelayThreshold

	hasDeviceBottleneck := ta.BusyDuration >= deviceBottleneckThreshold

	switch {
	case hasHostDelay && hasDeviceBottleneck:
		res.Mechanism = MechanismAmbiguousSimultaneous
		val := float64(ta.BusyDuration.Nanoseconds())
		msg := fmt.Sprintf("simultaneous timing excess: host delay (%v) and device duration (%v); cause not established",
			ta.HostAPIDuration, ta.BusyDuration)
		res.PrimaryFinding = tracecontract.Admit(
			fmt.Sprintf("diag-%s", opID.String()),
			"ambiguous_simultaneous_fault",
			tracecontract.FindingObserved,
			msg,
			&val,
			"ns",
			reqs,
			evCtx,
		)

	case hasHostDelay:
		res.Mechanism = MechanismHostSubmissionDelay
		val := float64(ta.HostAPIDuration.Nanoseconds())
		schedDetail := fmt.Sprintf("runnable (CPU starvation): %v, wait: %v", ta.RunnableDuration, ta.WaitDuration)
		msg := fmt.Sprintf("host submission delay observed: API duration %v, %s", ta.HostAPIDuration, schedDetail)
		targetClass := tracecontract.FindingObserved
		if confirms(in.Intervention, res.Mechanism, target, opID) {
			targetClass = tracecontract.FindingExperimentallyConfirmed
		}
		res.PrimaryFinding = tracecontract.Admit(
			fmt.Sprintf("diag-%s", opID.String()),
			"host_submission_delay",
			targetClass,
			msg,
			&val,
			"ns",
			reqs,
			evCtx,
		)

	case hasDeviceBottleneck:
		res.Mechanism = MechanismDeviceBottleneck
		val := float64(ta.BusyDuration.Nanoseconds())
		msg := fmt.Sprintf("device busy duration %v exceeds baseline %v; duration alone does not establish competing work",
			ta.BusyDuration, thresh.BaselineKernelDuration)
		targetClass := tracecontract.FindingObserved
		if confirms(in.Intervention, res.Mechanism, target, opID) {
			targetClass = tracecontract.FindingExperimentallyConfirmed
		}
		res.PrimaryFinding = tracecontract.Admit(
			fmt.Sprintf("diag-%s", opID.String()),
			"competing_device_bottleneck",
			targetClass,
			msg,
			&val,
			"ns",
			reqs,
			evCtx,
		)

	default:
		res.Mechanism = MechanismNoFault
		val := float64(ta.BusyDuration.Nanoseconds())
		msg := fmt.Sprintf("no fault observed: host API (%v) and device execution (%v) within baseline",
			ta.HostAPIDuration, ta.BusyDuration)
		res.PrimaryFinding = tracecontract.Admit(
			fmt.Sprintf("diag-%s", opID.String()),
			"no_fault",
			tracecontract.FindingObserved,
			msg,
			&val,
			"ns",
			reqs,
			evCtx,
		)
	}

	res.Findings = append(res.Findings, res.PrimaryFinding)
	return res
}

// diagnosticIntervals keeps every interval subject to clock and capture bounds.
func diagnosticIntervals(in DiagnosticInput) []tracecontract.Interval {
	out := []tracecontract.Interval{in.WallInterval}
	for _, h := range in.HostCalls {
		out = append(out, h.Span)
	}
	for _, k := range in.Kernels {
		out = append(out, k.Span)
	}
	for _, s := range in.Sched.Intervals {
		out = append(out, s.Span)
	}
	return out
}

func withheldDiagnosis(res DiagnosisResult, in DiagnosticInput, reason string) DiagnosisResult {
	res.Mechanism = MechanismWithheld
	res.WithheldReason = tracecontract.ReasonEvidenceUnavailable
	res.PrimaryFinding = tracecontract.Finding{ID: "mechanism-diagnosis", Rule: "mechanism_diagnosis", Class: tracecontract.FindingUnavailable, Admission: tracecontract.Admission{Status: "withheld", ReasonCode: tracecontract.ReasonEvidenceUnavailable, Reason: reason}}
	res.Findings = append(Observe(in), res.PrimaryFinding)
	return res
}
