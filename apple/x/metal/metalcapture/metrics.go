package metalcapture

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
)

// MetricKind indicates whether a metric represents durations, counts, bytes, or ratios.
type MetricKind string

const (
	MetricKindDuration MetricKind = "duration"
	MetricKindCount    MetricKind = "count"
	MetricKindBytes    MetricKind = "bytes"
	MetricKindRatio    MetricKind = "ratio"
)

// Aggregation defines how samples are combined across dispatches or passes.
type Aggregation string

const (
	AggregationSum  Aggregation = "sum"
	AggregationMean Aggregation = "mean"
	AggregationMax  Aggregation = "max"
	AggregationMin  Aggregation = "min"
	AggregationLast Aggregation = "last"
)

// Scope defines the hardware or logical level of the observation.
type Scope string

const (
	ScopeRun           Scope = "run"
	ScopeCommandBuffer Scope = "command_buffer"
	ScopeStageBoundary Scope = "stage_boundary"
	ScopeEncoder       Scope = "encoder"
	ScopePipeline      Scope = "pipeline"
	ScopeDispatch      Scope = "dispatch"
	ScopeSIMDGroup     Scope = "simd_group"
	ScopeInstruction   Scope = "instruction"
)

// MetricOrigin distinguishes live device measurements from post-capture replay enrichment.
type MetricOrigin string

const (
	OriginLive   MetricOrigin = "live"
	OriginReplay MetricOrigin = "replay"
)

// ObservationStatus records the collection or availability disposition of an observation.
type ObservationStatus string

const (
	StatusCollected   ObservationStatus = "collected"
	StatusUnavailable ObservationStatus = "unavailable"
	StatusUnsupported ObservationStatus = "unsupported"
)

// ObservationValue encapsulates a native numeric value, preserving exact integer representation
// (>2^53) without float64 precision loss while still supporting floating-point metrics (e.g. ratios).
type ObservationValue struct {
	IsInt    bool
	IntVal   int64
	FloatVal float64
}

// NewIntObservationValue creates an exact integer observation value.
func NewIntObservationValue(v int64) ObservationValue {
	return ObservationValue{IsInt: true, IntVal: v, FloatVal: float64(v)}
}

// NewFloatObservationValue creates a floating point observation value.
func NewFloatObservationValue(v float64) ObservationValue {
	return ObservationValue{IsInt: false, FloatVal: v}
}

// Float64 returns the float64 representation of the value.
func (v ObservationValue) Float64() float64 {
	if v.IsInt {
		return float64(v.IntVal)
	}
	return v.FloatVal
}

// Int64 returns the integer value if IsInt is true, or rounded int64 if not.
func (v ObservationValue) Int64() (int64, bool) {
	if v.IsInt {
		return v.IntVal, true
	}
	if math.IsNaN(v.FloatVal) || math.IsInf(v.FloatVal, 0) {
		return 0, false
	}
	if v.FloatVal < float64(math.MinInt64) || v.FloatVal > float64(math.MaxInt64) {
		return 0, false
	}
	return int64(math.Round(v.FloatVal)), true
}

func (v ObservationValue) MarshalJSON() ([]byte, error) {
	if v.IsInt {
		return []byte(strconv.FormatInt(v.IntVal, 10)), nil
	}
	return json.Marshal(v.FloatVal)
}

func (v *ObservationValue) UnmarshalJSON(b []byte) error {
	s := string(b)
	if iv, err := strconv.ParseInt(s, 10, 64); err == nil {
		v.IsInt = true
		v.IntVal = iv
		v.FloatVal = float64(iv)
		return nil
	}
	var fv float64
	if err := json.Unmarshal(b, &fv); err != nil {
		return fmt.Errorf("unmarshal observation value %q: %w", s, err)
	}
	v.IsInt = false
	v.FloatVal = fv
	return nil
}

// Observation describes a typed hardware or software performance measurement.
// It preserves native identifiers, numeric units, aggregation semantics,
// exact integer and float values, and collection provenance.
type Observation struct {
	ID          string            `json:"id"`                     // Canonical or native metric identifier
	Name        string            `json:"name"`                   // Human-readable metric name
	Unit        string            `json:"unit"`                   // Measurement unit
	Kind        MetricKind        `json:"kind"`                   // Data kind (duration, count, bytes, ratio)
	Aggregation Aggregation       `json:"aggregation"`            // How values aggregate (sum, mean, max, min, last)
	Scope       Scope             `json:"scope"`                  // Measurement scope (command_buffer, stage_boundary, pipeline, dispatch, etc.)
	Origin      MetricOrigin      `json:"origin"`                 // Live execution vs post-capture replay
	Status      ObservationStatus `json:"status,omitempty"`       // Availability status: collected, unavailable, unsupported
	Source      string            `json:"source,omitempty"`       // Field path in the retained collector export
	Reason      string            `json:"reason,omitempty"`       // Availability or interpretation explanation
	Device      string            `json:"device,omitempty"`       // Device name or registry identity
	RunID       string            `json:"run_id,omitempty"`       // Correlated run or pass identifier
	Pipeline    string            `json:"pipeline,omitempty"`     // Associated pipeline function name or identifier
	PipelineID  *int              `json:"pipeline_id,omitempty"`  // Associated pipeline numeric ID
	DispatchID  *int              `json:"dispatch_id,omitempty"`  // Associated dispatch sequence index
	Value       ObservationValue  `json:"value"`                  // Tagged native numeric value
	NonAdditive bool              `json:"non_additive,omitempty"` // If true, metric values must not be directly summed across dispatches/threads
}
