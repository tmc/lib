package cudadiageval

import (
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// Phase defines the stage of protocol execution.
type Phase string

const (
	// PhasePilot explores feasible budgets and system behaviors.
	// Results from PhasePilot are not admissible as qualification evidence.
	PhasePilot Phase = "pilot"

	// PhaseQualification executes the frozen protocol against held-out evidence.
	PhaseQualification Phase = "qualification"
)

// FaultLabel identifies an injected or held-out fault mechanism.
type FaultLabel string

const (
	// FaultNone indicates normal nominal execution with no induced defect.
	FaultNone FaultLabel = "none"

	// FaultHostSubmissionDelay indicates OS scheduling or runtime preemption
	// delay on the host submission thread before CUDA driver API entry.
	FaultHostSubmissionDelay FaultLabel = "host_submission_delay"

	// FaultDeviceBottleneck indicates kernel or device resource saturation.
	FaultDeviceBottleneck FaultLabel = "device_bottleneck"

	// FaultSyncStall indicates excessive host-device synchronization stalls.
	FaultSyncStall FaultLabel = "synchronization_stall"

	// FaultCollectiveDelay indicates a delayed peer in multi-GPU communication.
	FaultCollectiveDelay FaultLabel = "collective_peer_delay"
)

// ResilienceMode indicates a system stress or recovery condition.
type ResilienceMode string

const (
	// ResilienceNormal represents standard undisturbed execution.
	ResilienceNormal ResilienceMode = "normal"

	// ResilienceOverflow represents telemetry ring buffer overflow and record drop.
	ResilienceOverflow ResilienceMode = "buffer_overflow"

	// ResilienceRestart represents worker process restart and PID/Gen recycling.
	ResilienceRestart ResilienceMode = "process_restart"

	// ResilienceLateAttach represents collector attaching after workload start.
	ResilienceLateAttach ResilienceMode = "late_attach"

	// ResilienceShutdown represents graceful finalization during inflight work.
	ResilienceShutdown ResilienceMode = "collector_shutdown"
)

// WorkloadSpec configures a frozen evaluation workload.
type WorkloadSpec struct {
	Name             string `json:"name"`
	Concurrency      int    `json:"concurrency"`
	WarmupIterations int    `json:"warmup_iterations"`
	TestIterations   int    `json:"test_iterations"`
	BatchSize        int    `json:"batch_size"`
	InputDim         int    `json:"input_dim"`
}

// HardwareClosure records the precise hardware and environment state.
type HardwareClosure struct {
	GPUModel      string `json:"gpu_model"`
	GPUUUID       string `json:"gpu_uuid"`
	Architecture  string `json:"architecture"`
	SMCount       int    `json:"sm_count"`
	DriverVersion string `json:"driver_version"`
	CUDAVersion   string `json:"cuda_version"`
	HostOS        string `json:"host_os"`
	HostArch      string `json:"host_arch"`
	KernelRelease string `json:"kernel_release"`
	MIGMode       string `json:"mig_mode"`
	ClocksLocked  bool   `json:"clocks_locked"`
	PowerLimitW   int    `json:"power_limit_w"`
}

// PreregisteredBudgets specifies acceptance thresholds frozen before qualification.
type PreregisteredBudgets struct {
	MaxThroughputDegradationPct float64 `json:"max_throughput_degradation_pct"`
	MaxRSSGrowthMB              float64 `json:"max_rss_growth_mb"`
	MaxTailLatencyP99Pct        float64 `json:"max_tail_latency_p99_pct"`
	MaxIncorrectJoins           int     `json:"max_incorrect_joins"`
	MinNoFaultRejectionPct      float64 `json:"min_no_fault_rejection_pct"`
	MinHeldOutDiagnosisPct      float64 `json:"min_held_out_diagnosis_pct"`
}

// SamplingConfig defines collection parameters.
type SamplingConfig struct {
	BufferCapacity int           `json:"buffer_capacity"`
	Window         time.Duration `json:"window"`
	RateHz         int           `json:"rate_hz"`
}

// Protocol defines the preregistered evaluation plan.
type Protocol struct {
	Comparison        *ComparisonSpec      `json:"comparison,omitempty"`
	Tools             []ToolSpec           `json:"tools,omitempty"`
	PropertyLimits    map[string]float64   `json:"property_limits,omitempty"`
	Name              string               `json:"name"`
	Phase             Phase                `json:"phase"`
	Frozen            bool                 `json:"frozen"`
	FrozenDigest      string               `json:"frozen_digest,omitempty"`
	FrozenTimestamp   time.Time            `json:"frozen_timestamp,omitempty"`
	Workloads         []WorkloadSpec       `json:"workloads"`
	Hardware          HardwareClosure      `json:"hardware"`
	Budgets           PreregisteredBudgets `json:"budgets"`
	Sampling          SamplingConfig       `json:"sampling"`
	ExecutionOrder    string               `json:"execution_order"`
	UncertaintyMethod string               `json:"uncertainty_method"`
	StoppingRules     []string             `json:"stopping_rules"`
	ExclusionRules    []string             `json:"exclusion_rules"`
}

// Baseline records configuration and capabilities of an alternative tool.
type Baseline struct {
	Name             string   `json:"name"`
	Category         string   `json:"category"`
	Version          string   `json:"version"`
	ComparisonScope  string   `json:"comparison_scope"`
	Independent      bool     `json:"independent"`
	SupportedModes   []string `json:"supported_modes"`
	UnavailableModes []string `json:"unavailable_modes"`
}

// GroundTruthRecord holds known-answer state for an individual request.
type GroundTruthRecord struct {
	RequestID       string                    `json:"request_id"`
	WorkloadName    string                    `json:"workload_name"`
	ProcessGen      cudaworkload.ProcessGenID `json:"process_gen"`
	TID             uint64                    `json:"tid"`
	StreamID        uint32                    `json:"stream_id"`
	CorrelationID   uint32                    `json:"correlation_id"`
	Fault           FaultLabel                `json:"fault"`
	Resilience      ResilienceMode            `json:"resilience"`
	HostPrepDelay   time.Duration             `json:"host_prep_delay"`
	HostAPIDuration time.Duration             `json:"host_api_duration"`
	KernelDuration  time.Duration             `json:"kernel_duration"`
	TotalDuration   time.Duration             `json:"total_duration"`
	OutputHash      string                    `json:"output_hash"`
}

// PropertyGate holds the pass/fail determination for an evaluated property.
type PropertyGate struct {
	Property      string `json:"property"`
	GateName      string `json:"gate_name"`
	Disposition   string `json:"disposition"` // "passed", "failed", "withheld", "unavailable", "pending"
	MeasuredValue string `json:"measured_value"`
	Threshold     string `json:"threshold"`
	Reason        string `json:"reason"`
	Provenance    string `json:"provenance"`
}

// AttributionReceipt summarizes join accuracy against independent ground truth.
type AttributionReceipt struct {
	TotalRequests   int     `json:"total_requests"`
	JoinedRequests  int     `json:"joined_requests"`
	ExactJoins      int     `json:"exact_joins"`
	IncorrectJoins  int     `json:"incorrect_joins"`
	MissingJoins    int     `json:"missing_joins"`
	AmbiguousJoins  int     `json:"ambiguous_joins"`
	JoinPrecision   float64 `json:"join_precision"`
	JoinRecall      float64 `json:"join_recall"`
	GateDisposition string  `json:"gate_disposition"`
}

// DiagnosticReceipt summarizes causal mechanism inference on held-out faults.
// Counts are ground-truth cases. A Missing count is the number of cases in
// the matching Total that had no diagnostic input; such cases never pass.
// Zero means no input was missing. A Withheld count is the number of cases
// in the matching Total whose input was present but whose diagnosis
// cudadiag withheld or did not admit.
//
// HeldOutCorrect, HeldOutWithheld, HeldOutIncorrect and HeldOutMissing
// partition HeldOutTotal.
//
// NoFaultTotal counts no-fault cases with normal resilience. NoFaultPassed
// counts admitted diagnoses of exactly [cudadiag.MechanismNoFault].
// NoFaultPassed, NoFaultWithheld and NoFaultMissing are disjoint; the rest
// of NoFaultTotal is admitted false alarms. Withheld diagnoses never pass.
// NoFaultPassed/NoFaultTotal is correctness over all declared cases;
// (NoFaultTotal-NoFaultWithheld-NoFaultMissing)/NoFaultTotal is diagnostic
// coverage. Specificity among admitted answers uses that admitted count
// as its denominator and is undefined when no answers were admitted.
//
// CompetingDiverted counts competing device bottleneck cases diagnosed as
// exactly [cudadiag.MechanismDeviceBottleneck] and admitted; only these
// pass. CompetingDiverted, CompetingWithheld and CompetingMissing are
// disjoint, and the rest of CompetingTotal is the competing cases
// misclassified as another admitted mechanism.
//
// ResilienceTotal counts cases with a resilience mode other than
// [ResilienceNormal]. A buffer overflow case passes only if its diagnosis
// was withheld, since its activity record was dropped. Any other mode
// passes only with an admitted diagnosis; ResilienceWithheld counts those
// non-overflow cases whose diagnosis was withheld instead. An overflow
// case's withheld diagnosis is counted in ResiliencePassed, never in
// ResilienceWithheld. ResiliencePassed, ResilienceWithheld and
// ResilienceMissing are disjoint, and the rest of ResilienceTotal is
// overflow cases whose diagnosis was admitted despite the loss.
type DiagnosticReceipt struct {
	HeldOutTotal       int                      `json:"held_out_total"`
	HeldOutCorrect     int                      `json:"held_out_correct"`
	HeldOutWithheld    int                      `json:"held_out_withheld"`
	HeldOutIncorrect   int                      `json:"held_out_incorrect"`
	HeldOutMissing     int                      `json:"held_out_missing"`
	NoFaultTotal       int                      `json:"no_fault_total"`
	NoFaultPassed      int                      `json:"no_fault_passed"`
	NoFaultWithheld    int                      `json:"no_fault_withheld"`
	NoFaultMissing     int                      `json:"no_fault_missing"`
	CompetingTotal     int                      `json:"competing_total"`
	CompetingDiverted  int                      `json:"competing_diverted"`
	CompetingWithheld  int                      `json:"competing_withheld"`
	CompetingMissing   int                      `json:"competing_missing"`
	ResilienceTotal    int                      `json:"resilience_total"`
	ResiliencePassed   int                      `json:"resilience_passed"`
	ResilienceWithheld int                      `json:"resilience_withheld"`
	ResilienceMissing  int                      `json:"resilience_missing"`
	AccuracyPct        float64                  `json:"accuracy_pct"`
	ConfusionMatrix    map[string]int           `json:"confusion_matrix"`
	GateDisposition    string                   `json:"gate_disposition"`
	SampleDiagnosis    cudadiag.DiagnosisResult `json:"sample_diagnosis,omitempty"`
}

// CostReceipt records collection-off versus collection-on paired trial overheads.
type CostReceipt struct {
	BaselineRPS              float64 `json:"baseline_rps"`
	CollectorRPS             float64 `json:"collector_rps"`
	ThroughputDegradationPct float64 `json:"throughput_degradation_pct"`
	BaselineP99LatencyNs     int64   `json:"baseline_p99_latency_ns"`
	CollectorP99LatencyNs    int64   `json:"collector_p99_latency_ns"`
	LatencyP99OverheadPct    float64 `json:"latency_p99_overhead_pct"`
	MemoryGrowthMB           float64 `json:"memory_growth_mb"`
	TotalSubmissions         uint64  `json:"total_submissions"`
	DroppedRecords           uint64  `json:"dropped_records"`
	LossFraction             float64 `json:"loss_fraction"`
	GateDisposition          string  `json:"gate_disposition"`
}

// ReproducibilityReceipt summarizes replay and manifest consistency.
type ReproducibilityReceipt struct {
	ArtifactManifestSHA256 string            `json:"artifact_manifest_sha256"`
	ReplayMatches          bool              `json:"replay_matches"`
	MatrixCompleted        bool              `json:"matrix_completed"`
	ArtifactDigests        map[string]string `json:"artifact_digests"`
	GateDisposition        string            `json:"gate_disposition"`
}

// EvaluationReport contains full receipts across all four evaluation properties.
type EvaluationReport struct {
	// Synthetic indicates fixture-generated evidence rather than native capture.
	Synthetic       bool                   `json:"synthetic"`
	Protocol        Protocol               `json:"protocol"`
	Timestamp       time.Time              `json:"timestamp"`
	Duration        time.Duration          `json:"duration"`
	Baselines       []Baseline             `json:"baselines"`
	Attribution     AttributionReceipt     `json:"attribution"`
	Diagnostic      DiagnosticReceipt      `json:"diagnostic"`
	Cost            CostReceipt            `json:"cost"`
	Reproducibility ReproducibilityReceipt `json:"reproducibility"`
	Gates           []PropertyGate         `json:"gates"`
	Passed          bool                   `json:"passed"`
	Claim           *Claim                 `json:"claim,omitempty"`
}

// AllGates returns all recorded property gates including passes, failures, and withheld gates.
func (r *EvaluationReport) AllGates() []PropertyGate {
	return r.Gates
}

// Claim encapsulates a bounded comparative claim qualified by evaluation evidence.
type Claim struct {
	Task              string   `json:"task"`
	SpecificAdvantage string   `json:"specific_advantage"`
	ComparedWith      []string `json:"compared_with"`
	QualifiedOn       string   `json:"qualified_on"`
	Evidence          string   `json:"evidence"`
	Limits            []string `json:"limits"`
}
