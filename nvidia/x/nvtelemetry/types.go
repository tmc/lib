package nvtelemetry

import (
	"context"
	"time"
)

// Schema and version constants.
const (
	SchemaVersionLegacy = 1
	SchemaVersionV2     = 2
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
)

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
	IntervalNanos      int64  `json:"interval_nanos"`
	MaxDurationNanos   int64  `json:"max_duration_nanos"`
	MaxRecordBudget    int    `json:"max_record_budget"`
	MaxByteBudget      int    `json:"max_byte_budget"`
	MaxBufferedRecords int    `json:"max_buffered_records"`
	BudgetPolicy       string `json:"budget_policy,omitempty"`
	DropOnFull         bool   `json:"drop_on_full"`
}

// Manifest describes a telemetry run and its summary properties.
type Manifest struct {
	RunID                       string           `json:"run_id"`
	SchemaVersion               int              `json:"schema_version"`
	CollectorVersion            string           `json:"collector_version"`
	BuildRevision               string           `json:"build_revision,omitempty"`
	BuildModified               bool             `json:"build_modified,omitempty"`
	BinaryDigest                string           `json:"binary_digest,omitempty"`
	MetricRegistryVersion       string           `json:"metric_registry_version,omitempty"`
	ClockMonotonicDomain        string           `json:"clock_monotonic_domain,omitempty"`
	ClockMonotonicEpochID       string           `json:"clock_monotonic_epoch_id,omitempty"`
	ClockWallDomain             string           `json:"clock_wall_domain,omitempty"`
	ClockProvenance             string           `json:"clock_provenance,omitempty"`
	ClockMappingValidFrom       int64            `json:"clock_mapping_valid_from,omitempty"`
	ClockMappingValidUntil      int64            `json:"clock_mapping_valid_until,omitempty"`
	ClockCalibrationBoundsNanos int64            `json:"clock_calibration_bounds_nanos,omitempty"`
	HasClockCalibration         bool             `json:"has_clock_calibration,omitempty"`
	Devices                     []DeviceMetadata `json:"devices"`
	Config                      CollectorConfig  `json:"config"`
	StartTimeMonotonic          int64            `json:"start_time_monotonic"`
	StartTimeUnix               int64            `json:"start_time_unix"`
	EndTimeMonotonic            int64            `json:"end_time_monotonic"`
	EndTimeUnix                 int64            `json:"end_time_unix"`
	Phases                      []PhaseMarker    `json:"phases,omitempty"`
	TotalSamples                uint64           `json:"total_samples"`
	MissedTicks                 uint64           `json:"missed_ticks"`
	LostRecords                 uint64           `json:"lost_records"`
	IsComplete                  bool             `json:"is_complete"`
	IncompleteReason            string           `json:"incomplete_reason,omitempty"`
}

// WindowSummary aggregates one metric over a named window.
type WindowSummary struct {
	WindowName           string  `json:"window_name"`
	DeviceUUID           string  `json:"device_uuid"`
	MetricName           string  `json:"metric_name"`
	Unit                 string  `json:"unit"`
	TimeWeightedMean     float64 `json:"time_weighted_mean"`
	ArithmeticMean       float64 `json:"arithmetic_mean"`
	Min                  float64 `json:"min"`
	Max                  float64 `json:"max"`
	StartValue           float64 `json:"start_value"`
	EndValue             float64 `json:"end_value"`
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
