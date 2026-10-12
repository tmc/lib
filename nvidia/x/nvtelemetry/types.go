package nvtelemetry

import (
	"context"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// Schema and version constants.
const (
	SchemaVersionLegacy = 1
	SchemaVersionV2     = 2
	SchemaVersionV3     = 3
	SchemaVersionV4     = 4
	SchemaVersionV5     = 5
	SchemaVersion       = SchemaVersionV2
	CollectorVersion    = "nvtelemetry-2.0.0"
)

// Metric names.
const (
	MetricGPUUtilization      = "gpu_utilization_pct"
	MetricMemoryUtilization   = "memory_utilization_pct"
	MetricMemoryUsedBytes     = "memory_used_bytes"
	MetricMemoryTotalBytes    = "memory_total_bytes"
	MetricPowerUsageMW        = "power_milliwatts"
	MetricTotalEnergyMJ       = "total_energy_millijoules"
	MetricTemperatureC        = "temperature_celsius"
	MetricSMClockMHz          = "sm_clock_mhz"
	MetricMemoryClockMHz      = "memory_clock_mhz"
	MetricClocksEventReasons  = "clocks_event_reasons"
	MetricSMActivity          = "sm_activity_pct"
	MetricTensorActivity      = "tensor_activity_pct"
	MetricDRAMActivity        = "dram_activity_pct"
	MetricPCIeRxBytesPerSec   = "pcie_rx_bytes_per_sec"
	MetricPCIeTxBytesPerSec   = "pcie_tx_bytes_per_sec"
	MetricNVLinkRxBytesPerSec = "nvlink_rx_bytes_per_sec"
	MetricNVLinkTxBytesPerSec = "nvlink_tx_bytes_per_sec"
	MetricThrottleReasons     = "throttle_reasons"
	MetricPCIERxBytes         = "pcie_rx_bytes"
	MetricPCIETxBytes         = "pcie_tx_bytes"
	MetricXIDEvent            = "xid_event"
)

// Canonical units.
const (
	UnitPercent     = "percent"
	UnitBytes       = "bytes"
	UnitBytesPerSec = "bytes_per_sec"
	UnitMilliwatts  = "milliwatts"
	UnitMillijoules = "millijoules"
	UnitCelsius     = "celsius"
	UnitMHz         = "mhz"
	UnitBitmask     = "bitmask"
	UnitJoules      = "joules"
	UnitSeconds     = "seconds"
)

// Evidence classes.
const (
	EvidenceMeasured    = "measured"
	EvidenceDerived     = "derived"
	EvidenceUnavailable = "unavailable"
)

// Unavailable reasons.
const (
	ReasonUnsupported      = "unsupported"
	ReasonPermissionDenied = "permission_denied"
	ReasonGPULost          = "gpu_lost"
	ReasonBlank            = "blank"
	ReasonStale            = "stale"
	ReasonQueryFailure     = "query_failure"
	ReasonGapExceeded      = "gap_exceeded"
	ReasonCounterReset     = "counter_reset"
	ReasonZeroTokens       = "zero_tokens"
	ReasonMisalignedClock  = "misaligned_clock"
	ReasonIntegerPrecision = "integer_not_exactly_representable_as_float64"
)

// WindowSampleValuesUnavailable marks withheld integer extrema and endpoints.
const WindowSampleValuesUnavailable = "unavailable"

// MIG scope constants.
const (
	MIGScopeDisabled         = "disabled"
	MIGScopePhysicalParent   = "physical_parent"
	MIGScopeMIGInstance      = "mig_instance"
	MIGScopeUnsupported      = "unsupported"
	MIGScopePermissionDenied = "permission_denied"
	MIGScopeUnknown          = "unknown"
)

// Device status constants.
const (
	DeviceStatusOk             = "ok"
	DeviceStatusUnidentifiable = "unidentifiable"
)

// Field capability constants.
const (
	CapabilityQueryable        = "queryable"
	CapabilityUnsupported      = "unsupported"
	CapabilityPermissionDenied = "permission_denied"
	CapabilityUnknown          = "unknown"
)

// MetadataFailure records a typed failure during device metadata discovery.
type MetadataFailure struct {
	Property string `json:"property"`
	Reason   string `json:"reason"`
	Error    string `json:"error"`
}

// Standard phase names.
const (
	PhaseWarmup      = "warmup"
	PhaseMeasurement = "measurement"
	PhaseCooldown    = "cooldown"
)

// Reading represents a single hardware telemetry measurement or recorded
// unavailable status.
type Reading struct {
	// Number preserves the exact V3 scalar, including measured zero. Exactly one
	// numeric arm is present. Legacy streams continue using Value and IntValue.
	Number                    *Number        `json:"number,omitempty"`
	Observation               *Observation   `json:"observation,omitempty"`
	Source                    *ReadingSource `json:"source,omitempty"`
	Name                      string         `json:"name"`
	Unit                      string         `json:"unit"`
	Backend                   string         `json:"backend"`
	DeviceUUID                string         `json:"device_uuid"`
	Evidence                  string         `json:"evidence"`
	Value                     float64        `json:"value,omitempty"`
	IntValue                  uint64         `json:"int_value,omitempty"`
	IsInt                     bool           `json:"is_int,omitempty"`
	UnavailableReason         string         `json:"unavailable_reason,omitempty"`
	TimestampMonotonic        int64          `json:"timestamp_monotonic"`
	TimestampUnix             int64          `json:"timestamp_unix"`
	SourceTimestamp           int64          `json:"source_timestamp,omitempty"`
	HasSourceTimestamp        bool           `json:"has_source_timestamp,omitempty"`
	SourceTimestampDomain     string         `json:"source_timestamp_domain,omitempty"`
	WallClockUncertaintyNanos int64          `json:"wall_clock_uncertainty_nanos,omitempty"`
	HasWallClockUncertainty   bool           `json:"has_wall_clock_uncertainty,omitempty"`
	QueryStartNanos           int64          `json:"query_start_nanos,omitempty"`
	QueryEndNanos             int64          `json:"query_end_nanos,omitempty"`
	HasQueryBounds            bool           `json:"has_query_bounds,omitempty"`
	RefreshCadenceNanos       int64          `json:"refresh_cadence_nanos,omitempty"`
	IntegrationIntervalNanos  int64          `json:"integration_interval_nanos,omitempty"`
	IsWarmup                  bool           `json:"is_warmup,omitempty"`
}

// Number preserves one exact scalar. Its zero value has no value; an available
// V3 reading requires exactly one nonnil arm. Float must be finite.
type Number struct {
	Float *float64 `json:"float,omitempty"`
	Int   *int64   `json:"int,omitempty"`
	Uint  *uint64  `json:"uint,omitempty"`
}

// Histogram preserves cumulative bucket counts and their ordered upper bounds.
// Bounds use finite decimal strings or the final "+Inf" bound. The zero value
// is absent; a present histogram requires equally sized Bounds and Counts.
type Histogram struct {
	Bounds []string `json:"bounds"`
	Counts []uint64 `json:"counts"`
	// V5 fields preserve count and optional summary presence.
	Count       *uint64  `json:"count,omitempty"`
	Sum         *float64 `json:"sum,omitempty"`
	Min         *float64 `json:"min,omitempty"`
	Max         *float64 `json:"max,omitempty"`
	Temporality string   `json:"temporality,omitempty"`
	Flags       uint32   `json:"flags,omitempty"`
}

// Observer identifies an observation population and mode. Empty fields remain
// unknown and cannot satisfy consumer admission. Reading JSON does not verify
// that a physical observer ran.
type Observer struct {
	ID               string            `json:"id"`
	Version          string            `json:"version"`
	Mode             string            `json:"mode"`
	Pass             string            `json:"pass"`
	Population       string            `json:"population"`
	Filter           string            `json:"filter"`
	Replay           string            `json:"replay"`
	SamplingCoverage *SamplingCoverage `json:"sampling_coverage,omitempty"`
}

// SamplingCoverage records exact observed and eligible units for one observer
// population. A measured time basis is activity-union duration within the
// containing observation interval; other bases describe bounded populations,
// not elapsed-time duty. Nil means absent, while a present unknown status is
// explicit evidence that the measure is unknown.
type SamplingCoverage struct {
	Status      string  `json:"status"`
	Basis       string  `json:"basis,omitempty"`
	Scope       string  `json:"scope,omitempty"`
	Numerator   *Number `json:"numerator,omitempty"`
	Denominator *Number `json:"denominator,omitempty"`
	Unit        string  `json:"unit,omitempty"`
	Reason      string  `json:"reason,omitempty"`
}

// Observation links a V3 reading to existing subject, workload and trace
// contracts. SourceClass is native, managed, imported, synthetic or modeled;
// a label alone never establishes authenticity. Missing fields stay unknown.
// Kind is gauge, counter, histogram, host_call, kernel or a scheduler state
// (sched_running, sched_runnable, sched_blocked).
type Observation struct {
	// OTLP fields are V5-only assertions, not verified source facts.
	OTLPScopeName   *string                        `json:"otlp_scope_name,omitempty"`
	OTLPClock       *OTLPClockMetadata             `json:"otlp_clock_metadata,omitempty"`
	OTLPSourceTime  *OTLPSourceTime                `json:"otlp_source_time,omitempty"`
	Family          string                         `json:"family"`
	SourceClass     string                         `json:"source_class"`
	Producer        string                         `json:"producer"`
	ProducerVersion string                         `json:"producer_version"`
	ProducerSchema  string                         `json:"producer_schema"`
	RawSHA256       string                         `json:"raw_sha256"`
	Transform       string                         `json:"transform"`
	TransformSHA256 string                         `json:"transform_sha256"`
	Assignment      *EntityAssignment              `json:"assignment,omitempty"`
	Workload        *cudaworkload.WorkloadIdentity `json:"workload,omitempty"`
	OpID            cudaworkload.OpID              `json:"op_id,omitempty"`
	Host            *cudaworkload.HostSubmission   `json:"host,omitempty"`
	Device          *cudaworkload.DeviceActivity   `json:"device,omitempty"`
	Interval        *tracecontract.Interval        `json:"interval,omitempty"`
	ClockEpoch      string                         `json:"clock_epoch"`
	UncertaintyNS   *int64                         `json:"uncertainty_ns,omitempty"`
	Observer        Observer                       `json:"observer"`
	Kind            string                         `json:"kind"`
	CounterEpoch    string                         `json:"counter_epoch,omitempty"`
	CounterReset    bool                           `json:"counter_reset,omitempty"`
	Histogram       *Histogram                     `json:"histogram,omitempty"`
	Denominator     *Number                        `json:"denominator,omitempty"`
	DenominatorUnit string                         `json:"denominator_unit,omitempty"`
	Scope           string                         `json:"scope,omitempty"`
	HasReturn       bool                           `json:"has_return,omitempty"`
	TID             uint64                         `json:"tid,omitempty"`
}

// IsAvailable reports whether the reading represents an active valid measurement.
func (r Reading) IsAvailable() bool {
	return (r.Evidence == EvidenceMeasured || r.Evidence == EvidenceDerived) && r.UnavailableReason == ""
}

// DeviceMetadata records static or discovered device identity.
type DeviceMetadata struct {
	UUID                    string            `json:"uuid"`
	Index                   int               `json:"index"`
	Model                   string            `json:"model"`
	Architecture            string            `json:"architecture,omitempty"`
	DriverVersion           string            `json:"driver_version,omitempty"`
	LibraryVersion          string            `json:"library_version,omitempty"`
	PowerLimitMW            uint32            `json:"power_limit_mw,omitempty"`
	MaxSMClockMHz           uint32            `json:"max_sm_clock_mhz,omitempty"`
	MaxMemoryClockMHz       uint32            `json:"max_memory_clock_mhz,omitempty"`
	MIGEnabled              bool              `json:"mig_enabled,omitempty"`
	MIGParentUUID           string            `json:"mig_parent_uuid,omitempty"`
	MIGInstanceID           int               `json:"mig_instance_id,omitempty"`
	FieldCapabilities       []string          `json:"field_capabilities,omitempty"`
	ClockSettings           string            `json:"clock_settings,omitempty"`
	MIGScope                string            `json:"mig_scope,omitempty"`
	HasMIGInstanceID        bool              `json:"has_mig_instance_id,omitempty"`
	MIGComputeInstanceID    int               `json:"mig_compute_instance_id,omitempty"`
	HasMIGComputeInstanceID bool              `json:"has_mig_compute_instance_id,omitempty"`
	Status                  string            `json:"status,omitempty"`
	Warnings                []string          `json:"warnings,omitempty"`
	MetadataFailures        []MetadataFailure `json:"metadata_failures,omitempty"`
	FieldStatuses           map[string]string `json:"field_statuses,omitempty"`
}

// Sampler samples hardware or mock telemetry on request.
type Sampler interface {
	Sample(ctx context.Context) ([]Reading, error)
	Devices() []DeviceMetadata
}

// Sample packages a single scheduled polling pass across all devices.
type Sample struct {
	Seq                      uint64    `json:"seq"`
	QueryStartNanos          int64     `json:"query_start_nanos"`
	QueryEndNanos            int64     `json:"query_end_nanos"`
	LagNanos                 int64     `json:"lag_nanos"`
	ClockDiscontinuity       bool      `json:"clock_discontinuity,omitempty"`
	ClockDiscontinuityReason string    `json:"clock_discontinuity_reason,omitempty"`
	Readings                 []Reading `json:"readings"`
}

// PhaseMarker marks the boundary of an execution phase.
type PhaseMarker struct {
	Name                string `json:"name"`
	StartMonotonicNanos int64  `json:"start_monotonic_nanos"`
	EndMonotonicNanos   int64  `json:"end_monotonic_nanos"`
}

// Duration returns the total duration of the phase.
func (p PhaseMarker) Duration() time.Duration {
	if p.EndMonotonicNanos <= p.StartMonotonicNanos {
		return 0
	}
	return time.Duration(p.EndMonotonicNanos - p.StartMonotonicNanos)
}

// Contains reports whether a monotonic timestamp falls within the half-open phase interval [start, end).
func (p PhaseMarker) Contains(monoNanos int64) bool {
	return monoNanos >= p.StartMonotonicNanos && monoNanos < p.EndMonotonicNanos
}

// Budget policy options.
const (
	BudgetPolicyFail   = "fail"
	BudgetPolicyDrop   = "drop"
	BudgetExceededFail = BudgetPolicyFail
	BudgetExceededDrop = BudgetPolicyDrop
)

// CollectorConfig configures sampling and limits.
type CollectorConfig struct {
	// SchemaVersion selects the stream version. Zero retains V2. V3 emits exact
	// Number values; its Receipt preserves caller-supplied capture provenance.
	SchemaVersion      int                    `json:"schema_version,omitempty"`
	Receipt            *tracecontract.Receipt `json:"-"`
	IntervalNanos      int64                  `json:"interval_nanos"`
	MaxDurationNanos   int64                  `json:"max_duration_nanos"`
	MaxRecordBudget    int                    `json:"max_record_budget"`
	MaxByteBudget      int                    `json:"max_byte_budget"`
	MaxBufferedRecords int                    `json:"max_buffered_records"`
	BudgetPolicy       string                 `json:"budget_policy,omitempty"`
	DropOnFull         bool                   `json:"drop_on_full"`
}

// Manifest describes a telemetry run and its summary properties.
type Manifest struct {
	// Receipt preserves producer-defined record and loss units independently
	// of the canonical readings below. Absence means unassessed.
	Receipt *tracecontract.Receipt `json:"receipt,omitempty"`
	// In V3, LostRecordsByFamily counts canonical readings carrying each family.
	// Total is attempted readings; Dropped is readings lost by the stream writer.
	// At completion, known Total equals retained readings plus Dropped. Unknown
	// or missing families cannot establish known-zero loss. An initial header
	// or a stream without completion cannot establish final population counts.
	// LostRecords separately counts dropped JSONL records, not readings.
	LostRecordsByFamily         map[string]tracecontract.Loss `json:"lost_records_by_family,omitempty"`
	RunID                       string                        `json:"run_id"`
	SchemaVersion               int                           `json:"schema_version"`
	CollectorVersion            string                        `json:"collector_version"`
	BuildRevision               string                        `json:"build_revision,omitempty"`
	BuildModified               bool                          `json:"build_modified,omitempty"`
	BinaryDigest                string                        `json:"binary_digest,omitempty"`
	MetricRegistryVersion       string                        `json:"metric_registry_version,omitempty"`
	ClockMonotonicDomain        string                        `json:"clock_monotonic_domain,omitempty"`
	ClockMonotonicEpochID       string                        `json:"clock_monotonic_epoch_id,omitempty"`
	ClockWallDomain             string                        `json:"clock_wall_domain,omitempty"`
	ClockProvenance             string                        `json:"clock_provenance,omitempty"`
	ClockMappingValidFrom       int64                         `json:"clock_mapping_valid_from,omitempty"`
	ClockMappingValidUntil      int64                         `json:"clock_mapping_valid_until,omitempty"`
	ClockCalibrationBoundsNanos int64                         `json:"clock_calibration_bounds_nanos,omitempty"`
	HasClockCalibration         bool                          `json:"has_clock_calibration,omitempty"`
	Devices                     []DeviceMetadata              `json:"devices"`
	Config                      CollectorConfig               `json:"config"`
	StartTimeMonotonic          int64                         `json:"start_time_monotonic"`
	StartTimeUnix               int64                         `json:"start_time_unix"`
	EndTimeMonotonic            int64                         `json:"end_time_monotonic"`
	EndTimeUnix                 int64                         `json:"end_time_unix"`
	Phases                      []PhaseMarker                 `json:"phases,omitempty"`
	TotalSamples                uint64                        `json:"total_samples"`
	MissedTicks                 uint64                        `json:"missed_ticks"`
	LostRecords                 uint64                        `json:"lost_records"`
	IsComplete                  bool                          `json:"is_complete"`
	IncompleteReason            string                        `json:"incomplete_reason,omitempty"`
}

// WindowSummary aggregates one metric over a named window.
type WindowSummary struct {
	WindowName string `json:"window_name"`
	DeviceUUID string `json:"device_uuid"`
	MetricName string `json:"metric_name"`
	Unit       string `json:"unit"`
	// TimeWeightedMean is a float64 approximation computed by integrating
	// floating-point sample values over covered intervals.
	TimeWeightedMean float64 `json:"time_weighted_mean"`
	// ArithmeticMean is a float64 approximation computed from valid sample
	// values. It is not an exact average of large integer inputs.
	ArithmeticMean float64 `json:"arithmetic_mean"`
	Min            float64 `json:"min"`
	Max            float64 `json:"max"`
	StartValue     float64 `json:"start_value"`
	EndValue       float64 `json:"end_value"`
	// SampleValuesStatus and SampleValuesReason cover Min, Max, StartValue and
	// EndValue only. They are set when those fields are withheld for precision.
	// An empty status is compatible with older summaries and does not assert
	// either metric unavailability or exact sample values.
	SampleValuesStatus   string  `json:"sample_values_status,omitempty"`
	SampleValuesReason   string  `json:"sample_values_reason,omitempty"`
	CoveredDurationNanos int64   `json:"covered_duration_nanos"`
	TotalDurationNanos   int64   `json:"total_duration_nanos"`
	CoverageRatio        float64 `json:"coverage_ratio"`
	MaxGapNanos          int64   `json:"max_gap_nanos"`
	SampleCount          int     `json:"sample_count"`
}

// EnergySummary reports energy consumed in a window.
type EnergySummary struct {
	StartMonoNanos       int64   `json:"start_mono_nanos"`
	EndMonoNanos         int64   `json:"end_mono_nanos"`
	DeviceUUID           string  `json:"device_uuid"`
	WindowName           string  `json:"window_name"`
	Method               string  `json:"method"` // "cumulative_counter" or "integrated_power"
	StartEnergyMJ        uint64  `json:"start_energy_mj"`
	EndEnergyMJ          uint64  `json:"end_energy_mj"`
	EnergyConsumedMJ     uint64  `json:"energy_consumed_mj"`
	EnergyConsumedJoules float64 `json:"energy_consumed_joules"`
	CoveredDurationNanos int64   `json:"covered_duration_nanos"`
	TotalDurationNanos   int64   `json:"total_duration_nanos"`
	CoverageRatio        float64 `json:"coverage_ratio"`
	ResetDetected        bool    `json:"reset_detected"`
	Valid                bool    `json:"valid"`
	Reason               string  `json:"reason,omitempty"`
}

// BenchmarkRecord represents one request record from an external load generator.
type BenchmarkRecord struct {
	RequestID      string `json:"request_id"`
	StartTimeNanos int64  `json:"start_time_nanos"`
	EndTimeNanos   int64  `json:"end_time_nanos"`
	PromptTokens   int    `json:"prompt_tokens"`
	OutputTokens   int    `json:"output_tokens"`
	Success        bool   `json:"success"`
	Endpoint       string `json:"endpoint,omitempty"`
}

// BenchmarkCorrelation joins a telemetry window with benchmark requests.
type BenchmarkCorrelation struct {
	StartMonoNanos       int64   `json:"start_mono_nanos"`
	EndMonoNanos         int64   `json:"end_mono_nanos"`
	WindowName           string  `json:"window_name"`
	TotalRequests        int     `json:"total_requests"`
	SuccessfulRequests   int     `json:"successful_requests"`
	TotalOutputTokens    int     `json:"total_output_tokens"`
	TotalEnergyJoules    float64 `json:"total_energy_joules"`
	JoulesPerOutputToken float64 `json:"joules_per_output_token"`
	OutputTokensPerJoule float64 `json:"output_tokens_per_joule"`
	TokensAvailable      bool    `json:"tokens_available"`
	EnergyAvailable      bool    `json:"energy_available"`
	ClockAligned         bool    `json:"clock_aligned"`
	RefusalReason        string  `json:"refusal_reason,omitempty"`
}

// OTLPClockMetadata preserves declared clock facts, including known zero bounds.
type OTLPClockMetadata struct {
	ClockDomain                 string `json:"clock_domain"`
	HasClockCalibration         bool   `json:"has_clock_calibration"`
	ClockCalibrationBoundsNanos int64  `json:"clock_calibration_bounds_nanos"`
}

// OTLPSourceTime preserves the complete caller-asserted source-to-Unix transform.
// It establishes internal consistency, not calibration or source authenticity.
type OTLPSourceTime struct {
	SourceClass              string `json:"source_class"`
	SourceClockEpoch         string `json:"source_clock_epoch"`
	SourceMonotonicEndNanos  int64  `json:"source_monotonic_end_nanos"`
	SourceIntervalPresent    bool   `json:"source_interval_present"`
	SourceIntervalClock      string `json:"source_interval_clock"`
	SourceIntervalStartNanos int64  `json:"source_interval_start_nanos"`
	SourceIntervalEndNanos   int64  `json:"source_interval_end_nanos"`
	MappingDomain            string `json:"mapping_domain"`
	MappingEpoch             string `json:"mapping_epoch"`
	MappingValidFromNanos    int64  `json:"mapping_valid_from_nanos"`
	MappingValidUntilNanos   int64  `json:"mapping_valid_until_nanos"`
	MappingUnixOffsetNanos   int64  `json:"mapping_unix_offset_nanos"`
	MappingProvenance        string `json:"mapping_provenance"`
}
