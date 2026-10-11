package cudadiagdemo

import (
	"context"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
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
}

// DemonstrationConfig configures the end-to-end demonstration.
type DemonstrationConfig struct {
	NumRequests       int           `json:"num_requests"`
	Concurrency       int           `json:"concurrency"`
	HostDelay         time.Duration `json:"host_delay"`
	DeviceDelay       time.Duration `json:"device_delay"`
	OutputDir         string        `json:"output_dir"`
	SkipIntervention  bool          `json:"skip_intervention,omitempty"`
	ExerciseCorruptEv bool          `json:"exercise_corrupt_ev,omitempty"`
	ExercisePressure  bool          `json:"exercise_pressure,omitempty"`
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
}

// CreateDiagnosticInput constructs a valid DiagnosticInput for mechanism evaluation.
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
		ClockProvenance:    "cupti_timestamp",
		ClockUncertaintyNS: &unc,
		Uncertainty:        &uncF,
		Gates:              gates,
		Backends: []tracecontract.Backend{
			{Name: "native", Version: "12.6"},
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
