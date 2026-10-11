package cudaregress

import (
	"time"

	"github.com/tmc/lib/nvidia/x/cudacmp"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// Version identifies the schema version for experiment plans and receipts.
const Version = 1

// Environment identifies the host and GPU device execution environment.
type Environment struct {
	DeviceIndex       int    `json:"device_index"`
	DeviceUUID        string `json:"device_uuid,omitempty"`
	Model             string `json:"model,omitempty"`
	ComputeCapability string `json:"compute_capability,omitempty"`
	Driver            string `json:"driver,omitempty"`
	CUDA              string `json:"cuda,omitempty"`
	Clock             string `json:"clock,omitempty"`
	Domain            string `json:"domain,omitempty"`
	HostArch          string `json:"host_arch,omitempty"`
	KernelVersion     string `json:"kernel_version,omitempty"`
}

// ArtifactRef references an immutable artifact by path, size, and SHA-256 digest.
type ArtifactRef struct {
	Name      string `json:"name"`
	Path      string `json:"path"`
	Digest    string `json:"sha256"`
	SizeBytes int64  `json:"size_bytes"`
	Role      string `json:"role,omitempty"` // source, binary, input, library, trace, reference
}

// LibraryRef captures one dynamic or runtime library in the closure.
type LibraryRef struct {
	Name    string `json:"name"`
	Path    string `json:"path"`
	Version string `json:"version,omitempty"`
	Digest  string `json:"sha256,omitempty"`
}

// InputRef captures an input dataset or tensor specification.
type InputRef struct {
	Name          string  `json:"name"`
	Path          string  `json:"path"`
	Digest        string  `json:"sha256"`
	Seed          int64   `json:"seed,omitempty"`
	ShapeDims     []int64 `json:"shape_dims,omitempty"`
	NumericalMode string  `json:"numerical_mode,omitempty"`
}

// CorrectnessOracle defines how candidate outputs are verified against expectations.
type CorrectnessOracle struct {
	Method          string  `json:"method"` // "digest-exact" or "numerical-tolerance"
	ExpectedDigest  string  `json:"expected_digest,omitempty"`
	Tolerance       float64 `json:"tolerance,omitempty"`
	ReferencePath   string  `json:"reference_path,omitempty"`
	ReferenceDigest string  `json:"reference_digest,omitempty"`
}

// TimingScope specifies the timing measurement channel, phases, and interval name.
type TimingScope struct {
	Channel  cudatrial.TimingKind `json:"channel"`
	Phases   []cudacmp.Phase      `json:"phases,omitempty"`
	Interval string               `json:"interval,omitempty"`
}

// DeviceConfig specifies target GPU hardware requirements and operating modes.
type DeviceConfig struct {
	DeviceIndex         int    `json:"device_index"`
	DeviceUUID          string `json:"device_uuid,omitempty"`
	Model               string `json:"model,omitempty"`
	ComputeCapability   string `json:"compute_capability,omitempty"`
	DriverVersion       string `json:"driver_version,omitempty"`
	CUDAVersion         string `json:"cuda_version,omitempty"`
	TargetClockMHz      int    `json:"target_clock_mhz,omitempty"`
	PowerLimitWatts     int    `json:"power_limit_watts,omitempty"`
	MIGMode             string `json:"mig_mode,omitempty"`
	RequireHardwareLock bool   `json:"require_hardware_lock"`
}

// CaptureConfig specifies collection parameters for non-perturbing telemetry.
type CaptureConfig struct {
	Enabled         bool   `json:"enabled"`
	Collector       string `json:"collector,omitempty"`
	SamplingRateHz  int    `json:"sampling_rate_hz,omitempty"`
	MaxCaptureBytes int64  `json:"max_capture_bytes,omitempty"`
	TraceLossBudget uint64 `json:"trace_loss_budget"`
}

// NullControlConfig configures baseline-vs-baseline noise floor estimation.
type NullControlConfig struct {
	Enabled         bool          `json:"enabled"`
	BasketRuns      int           `json:"basket_runs"`
	MaxNoiseFloorNs time.Duration `json:"max_noise_floor_ns"`
}

// StoppingRules specifies criteria for stopping or reporting regression.
type StoppingRules struct {
	MaxTrials              int     `json:"max_trials"`
	RelativeThreshold      float64 `json:"relative_threshold"` // fractional slowdown, e.g. 0.10 for 10%
	NoiseNanos             float64 `json:"noise_nanos"`
	MinSamples             int     `json:"min_samples"`
	MinEffectSize          float64 `json:"min_effect_size,omitempty"`
	OracleTolerance        float64 `json:"oracle_tolerance,omitempty"`
	MaxConsecutiveFailures int     `json:"max_consecutive_failures,omitempty"`
	HaltOnDrift            bool    `json:"halt_on_drift"`
}

// Incident encapsulates an admitted diagnostic incident and its supporting evidence.
type Incident struct {
	ID                  string                        `json:"id"`
	Admitted            bool                          `json:"admitted"`
	WithheldReason      string                        `json:"withheld_reason,omitempty"`
	Diagnosis           cudadiag.DiagnosisResult      `json:"diagnosis"`
	PrimaryFinding      tracecontract.Finding         `json:"primary_finding"`
	Workload            cudaworkload.WorkloadIdentity `json:"workload"`
	CaptureReceiptRef   ArtifactRef                   `json:"capture_receipt_ref"`
	ObservedEnvironment Environment                   `json:"observed_environment"`
	CreatedAt           time.Time                     `json:"created_at"`
}

// PlanOptions configures experiment plan generation from an incident.
type PlanOptions struct {
	WarmupRuns      int
	Rounds          int
	Order           string
	ExecutionScope  TimingScope
	StoppingRules   StoppingRules
	NullControls    NullControlConfig
	Sources         []ArtifactRef
	Binaries        []ArtifactRef
	Libraries       []LibraryRef
	Inputs          []InputRef
	Oracle          CorrectnessOracle
	DeviceOverrides *DeviceConfig
}

// ExperimentPlan is a bounded, fully specified, replayable experiment.
type ExperimentPlan struct {
	Version           int                           `json:"version"`
	PlanID            string                        `json:"plan_id"`
	PlanDigest        string                        `json:"plan_digest"`
	IncidentID        string                        `json:"incident_id"`
	IncidentDigest    string                        `json:"incident_digest"`
	Workload          cudaworkload.WorkloadIdentity `json:"workload"`
	SourceArtifacts   []ArtifactRef                 `json:"source_artifacts"`
	BinaryArtifacts   []ArtifactRef                 `json:"binary_artifacts"`
	LibraryClosure    []LibraryRef                  `json:"library_closure"`
	Inputs            []InputRef                    `json:"inputs"`
	CorrectnessOracle CorrectnessOracle             `json:"correctness_oracle"`
	TimingScope       TimingScope                   `json:"timing_scope"`
	DeviceConfig      DeviceConfig                  `json:"device_config"`
	CaptureConfig     CaptureConfig                 `json:"capture_config"`
	ExecutionOrder    string                        `json:"execution_order"` // "ABBA"
	WarmupRuns        int                           `json:"warmup_runs"`
	Rounds            int                           `json:"rounds"`
	NullControls      NullControlConfig             `json:"null_controls"`
	StoppingRules     StoppingRules                 `json:"stopping_rules"`
}

// PlanReceipt certifies the generation and validation of an ExperimentPlan.
type PlanReceipt struct {
	PlanID               string        `json:"plan_id"`
	PlanDigest           string        `json:"plan_digest"`
	IncidentID           string        `json:"incident_id"`
	CreatedAt            time.Time     `json:"created_at"`
	ArtifactClosureValid bool          `json:"artifact_closure_valid"`
	ArtifactsVerified    int           `json:"artifacts_verified"`
	ImmutableRefs        []ArtifactRef `json:"immutable_refs"`
}

// ExecutionReceipt certifies the execution of an ExperimentPlan across rounds.
type ExecutionReceipt struct {
	ExecutionID         string              `json:"execution_id"`
	PlanID              string              `json:"plan_id"`
	PlanDigest          string              `json:"plan_digest"`
	StartedAt           time.Time           `json:"started_at"`
	FinishedAt          time.Time           `json:"finished_at"`
	EnvironmentObserved Environment         `json:"environment_observed"`
	WarmupRunsCompleted int                 `json:"warmup_runs_completed"`
	RoundsCompleted     int                 `json:"rounds_completed"`
	ArmReceipts         []cudatrial.Receipt `json:"arm_receipts"`
	NullControlReceipts []cudatrial.Receipt `json:"null_control_receipts"`
	NullDrift           time.Duration       `json:"null_drift"`
	OutputsVerified     bool                `json:"outputs_verified"`
	OutputDigests       map[string]string   `json:"output_digests"`
	Partial             bool                `json:"partial"`
	HaltReason          string              `json:"halt_reason,omitempty"`
}

// ResourceChange captures hardware or resource state divergence between baseline and candidate.
type ResourceChange struct {
	Resource  string `json:"resource"` // e.g. clock_mhz, power_watts, sm_occupancy, memory_bytes
	Baseline  string `json:"baseline"`
	Candidate string `json:"candidate"`
}

// PerformanceChange captures timing metric delta on a specific timing channel.
type PerformanceChange struct {
	Channel   cudatrial.TimingKind `json:"channel"`
	Baseline  time.Duration        `json:"baseline"`
	Candidate time.Duration        `json:"candidate"`
	Delta     time.Duration        `json:"delta"`
	Ratio     float64              `json:"ratio"`
}

// ComparisonReceipt certifies the differential analysis and regression verdict.
type ComparisonReceipt struct {
	ExecutionID          string                     `json:"execution_id"`
	PlanID               string                     `json:"plan_id"`
	EvaluatedAt          time.Time                  `json:"evaluated_at"`
	BaselineArm          string                     `json:"baseline_arm"`
	CandidateArm         string                     `json:"candidate_arm"`
	TimingChannel        cudatrial.TimingKind       `json:"timing_channel"`
	BaselineDuration     time.Duration              `json:"baseline_duration"`
	CandidateDuration    time.Duration              `json:"candidate_duration"`
	Delta                time.Duration              `json:"delta"`
	Ratio                float64                    `json:"ratio"`
	NullDriftNoiseFloor  time.Duration              `json:"null_drift_noise_floor"`
	ExceedsNoiseFloor    bool                       `json:"exceeds_noise_floor"`
	RegressionDetected   bool                       `json:"regression_detected"`
	Verdict              cudaprofilehistory.Verdict `json:"verdict"` // "pass", "fail", "unknown"
	Reason               string                     `json:"reason"`
	MissingFamilies      []string                   `json:"missing_families,omitempty"`
	MeasuredZeroFamilies []string                   `json:"measured_zero_families,omitempty"`
	ResourceChanges      []ResourceChange           `json:"resource_changes,omitempty"`
	PerformanceChanges   []PerformanceChange        `json:"performance_changes,omitempty"`
	HistoryRecordKey     string                     `json:"history_record_key,omitempty"`
}
