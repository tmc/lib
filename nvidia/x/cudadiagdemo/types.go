package cudadiagdemo

import (
	"context"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// JoinEngine is a type alias for the evidence-preserving workload joiner.
type JoinEngine = cudaworkload.Joiner

// Engine encapsulates evidence-admitted diagnostic mechanism evaluation.
type Engine struct {
	Thresholds cudadiag.Thresholds
}

// NewEngine creates an Engine with specified thresholds.
func NewEngine(thresh cudadiag.Thresholds) *Engine {
	return &Engine{Thresholds: thresh}
}

// Diagnose evaluates primary evidence deterministically against admitted contracts.
func (e *Engine) Diagnose(target cudaworkload.WorkloadIdentity, opID cudaworkload.OpID, in cudadiag.DiagnosticInput) cudadiag.DiagnosisResult {
	if in.Thresholds.BaselineKernelDuration == 0 {
		in.Thresholds = e.Thresholds
	}
	return cudadiag.Diagnose(target, opID, in)
}

// FormulatePlan derives an immutable, reproducible ExperimentPlan from an admitted incident.
func FormulatePlan(inc cudaregress.Incident, opts cudaregress.PlanOptions) (*cudaregress.ExperimentPlan, *cudaregress.PlanReceipt, error) {
	return cudaregress.NewPlanFromIncident(inc, opts)
}

// ExecuteReplay coordinates interleaved trial execution across warmup, null controls, and ABBA rounds.
func ExecuteReplay(
	ctx context.Context,
	plan *cudaregress.ExperimentPlan,
	runner cudaregress.ReplayRunner,
	opts cudaregress.ReplayOptions,
) (*cudaregress.ExecutionReceipt, *cudaregress.ComparisonReceipt, error) {
	return cudaregress.Replay(ctx, plan, runner, opts)
}

// CollectionCostReport captures overhead and resource measurements for the telemetry pipeline.
type CollectionCostReport struct {
	TotalSubmissions       uint64        `json:"total_submissions"`
	TotalActivities        uint64        `json:"total_activities"`
	RingBufferCapacity     int           `json:"ring_buffer_capacity"`
	RetainedRecords        int           `json:"retained_records"`
	CaptureBundleBytes     int64         `json:"capture_bundle_bytes"`
	AverageRecordLatencyNs int64         `json:"average_record_latency_ns"`
	ElapsedCaptureDuration time.Duration `json:"elapsed_capture_duration"`
	// MeasurementStatus distinguishes collected counts/bytes from unavailable
	// record latency and matched collector-off/on workload cost.
	MeasurementStatus string `json:"measurement_status"`
}

// DemonstrationConfig configures the end-to-end demonstration.
type DemonstrationConfig struct {
	NumRequests int           `json:"num_requests"`
	Concurrency int           `json:"concurrency"`
	HostDelay   time.Duration `json:"host_delay"`
	DeviceDelay time.Duration `json:"device_delay"`
	// OutputDir retains captures and trial closure, including the executing host
	// binary. When empty, temporary closure files are removed before return;
	// returned file references then cannot reconstruct a later replay.
	OutputDir         string `json:"output_dir"`
	SkipIntervention  bool   `json:"skip_intervention,omitempty"`
	ExerciseCorruptEv bool   `json:"exercise_corrupt_ev,omitempty"`
	ExercisePressure  bool   `json:"exercise_pressure,omitempty"`
}

// DefaultDemonstrationConfig returns balanced demonstration defaults.
func DefaultDemonstrationConfig() DemonstrationConfig {
	return DemonstrationConfig{
		NumRequests:       16,
		Concurrency:       4,
		HostDelay:         15 * time.Millisecond,
		DeviceDelay:       300 * time.Microsecond,
		ExerciseCorruptEv: true,
		ExercisePressure:  true,
	}
}

// DemonstrationResult holds the complete evaluation, diagnostics, trials, and receipts.
type DemonstrationResult struct {
	// Qualification records the evidence scope; this fixture is not native qualification.
	Qualification        string                              `json:"qualification"`
	Synthetic            bool                                `json:"synthetic"`
	AttributionMetrics   cudaworkload.JoinMetrics            `json:"attribution_metrics"`
	AttributionAccuracy  float64                             `json:"attribution_accuracy"`
	ConfusionMatrix      cudadiag.ConfusionMatrix            `json:"confusion_matrix"`
	BaselineDiagnosis    cudadiag.DiagnosisResult            `json:"baseline_diagnosis"`
	HostDelayDiagnosis   cudadiag.DiagnosisResult            `json:"host_delay_diagnosis"`
	DeviceBottleneckDiag cudadiag.DiagnosisResult            `json:"device_bottleneck_diagnosis"`
	WithheldDiagnosis    cudadiag.DiagnosisResult            `json:"withheld_diagnosis"`
	IncidentReceipt      *cudaflightrecorder.IncidentReceipt `json:"incident_receipt,omitempty"`
	TrialPlan            *cudaregress.ExperimentPlan         `json:"trial_plan,omitempty"`
	PlanReceipt          *cudaregress.PlanReceipt            `json:"plan_receipt,omitempty"`
	TrialExecution       *cudaregress.ExecutionReceipt       `json:"trial_execution,omitempty"`
	TrialComparison      *cudaregress.ComparisonReceipt      `json:"trial_comparison,omitempty"`
	ArtifactFiles        map[string]string                   `json:"artifact_files,omitempty"`
	CollectionCost       CollectionCostReport                `json:"collection_cost"`
	// DiagnosticInputs are decoded from retained capture artifacts, independently
	// of GroundTruth. CaptureSHA256 binds each input to its containing bundle.
	DiagnosticInputs map[string]cudadiag.DiagnosticInput `json:"diagnostic_inputs"`
	CaptureSHA256    map[string]string                   `json:"capture_sha256"`
	// CaptureBundles retains the raw portable evidence for export/replay. It is
	// omitted from summary JSON; hashes and diagnostic inputs remain visible.
	CaptureBundles    map[string][]byte         `json:"-"`
	ReplayTimings     []SubmissionTiming        `json:"replay_timings,omitempty"`
	PressureDiagnosis *cudadiag.DiagnosisResult `json:"pressure_diagnosis,omitempty"`
	// PhaseVerification checks one actually executed pair using the distinct
	// pre-entry metric. TrialExecution separately records the full protocol.
	PhaseVerification      *tracecontract.InterventionReceipt `json:"phase_verification,omitempty"`
	PhaseVerificationError string                             `json:"phase_verification_error,omitempty"`
}

// SubmissionTiming is the version-one portable service observation. Service,
// PreEntry and API use the stated process-local monotonic epoch. Device is a
// synthetic interval, never a physical GPU observation. Mode, requested delay,
// expected output and expected mechanism belong in separate ground truth.
type SubmissionTiming struct {
	RequestID string                      `json:"request_id"`
	Service   tracecontract.Interval      `json:"service"`
	PreEntry  tracecontract.Interval      `json:"pre_entry"`
	API       cudadiag.HostCallEvent      `json:"api"`
	Host      cudaworkload.HostSubmission `json:"host"`
	Device    cudaworkload.DeviceActivity `json:"device"`
	Complete  bool                        `json:"complete"`
}

// TimingSource identifies the portable timing observer. Matching these fields
// verifies a declared replay contract, not physical producer authenticity.
type TimingSource struct {
	Version     int                  `json:"version"`
	Producer    string               `json:"producer"`
	SourceClass string               `json:"source_class"`
	ClockEpoch  string               `json:"clock_epoch"`
	Observer    nvtelemetry.Observer `json:"observer"`
}

// CreateDiagnosticInput constructs legacy synthetic protocol inputs. Its gates
// and coverage are fixture declarations, not captured observations. Production
// demonstration inputs are decoded from the returned recorder bundle instead.
func CreateDiagnosticInput(
	wall tracecontract.Interval,
	hostCalls []cudadiag.HostCallEvent,
	kernels []cudadiag.KernelEvent,
	sched cudadiag.SchedEvidence,
	thresh cudadiag.Thresholds,
) cudadiag.DiagnosticInput {
	gates := make(map[string]tracecontract.Gate)
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		gates[name] = tracecontract.Gate{Disposition: "passed", Reason: "qualified fixture", Provenance: "gpudiagdemo"}
	}
	unc := int64(10)
	uncF := 10.0

	receipt := tracecontract.Receipt{
		Version:            1,
		Command:            "gpudiagdemo run",
		InputPath:          "capture.tar",
		InputSHA256:        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		Kind:               "capture",
		ClockDomain:        "monotonic",
		ClockProvenance:    "constructed portable fixture intervals",
		ClockUncertaintyNS: &unc,
		Uncertainty:        &uncF,
		Gates:              gates,
		Backends: []tracecontract.Backend{
			{Name: "synthetic_legacy_fixture", Version: "1"},
		},
		ProbeCoverage: map[string]tracecontract.ProbeCoverage{
			"cupti_activity": {Present: true, Ratio: 1.0},
			"sched":          {Present: true, Ratio: 1.0},
		},
		LostRecords: map[string]tracecontract.Loss{
			"activity": {Dropped: 0, Total: 100, Unknown: false},
			"sched":    {Dropped: 0, Total: 50, Unknown: false},
		},
	}

	return cudadiag.DiagnosticInput{
		WallInterval: wall,
		HostCalls:    hostCalls,
		Kernels:      kernels,
		Sched:        sched,
		Receipt:      receipt,
		Thresholds:   thresh,
	}
}
