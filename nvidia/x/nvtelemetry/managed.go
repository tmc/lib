package nvtelemetry

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/big"
	"strconv"
)

// ReadingSource records import lineage. Exposition timestamps are producer
// publication timestamps, never proof of sensor acquisition time.
type ReadingSource struct {
	Native                  *DCGMReadingSource `json:"native,omitempty"`
	Mapping                 ManagedField       `json:"mapping"`
	MappingDigest           string             `json:"mapping_digest"`
	Schema                  string             `json:"schema"`
	SourceID                string             `json:"source_id"`
	Producer                string             `json:"producer"`
	ProducerVersion         string             `json:"producer_version"`
	NativeOrigin            string             `json:"native_origin"`
	FieldID                 string             `json:"field_id"`
	RawValue                string             `json:"raw_value"`
	Labels                  map[string]string  `json:"labels,omitempty"`
	ExpositionTimestampUnix int64              `json:"exposition_timestamp_unix,omitempty"`
	HasExpositionTimestamp  bool               `json:"has_exposition_timestamp,omitempty"`
	RawTimestamp            string             `json:"raw_timestamp,omitempty"`
	TimestampUnit           string             `json:"timestamp_unit,omitempty"`
	CounterReset            bool               `json:"counter_reset,omitempty"`
}

// ValidateReadingSource checks the bounded additive source contract.
func ValidateReadingSource(s *ReadingSource) error {
	if s == nil {
		return nil
	}
	if s.Schema != "nvtelemetry.managed.v1" {
		return fmt.Errorf("unsupported source schema")
	}
	for _, v := range []string{s.Mapping.Metric, s.Mapping.Name, s.Mapping.Type, s.Mapping.SourceUnit, s.RawTimestamp, s.TimestampUnit} {
		if len(v) > 256 {
			return fmt.Errorf("source mapping or timestamp limit")
		}
	}
	if s.Mapping.Numerator <= 0 || s.Mapping.Denominator <= 0 || s.Mapping.Metric != s.FieldID || s.MappingDigest != managedMappingDigest(s.Mapping) {
		return fmt.Errorf("invalid mapping provenance")
	}
	validateMapping := validateManagedMapping
	if s.Native != nil {
		validateMapping = validateDCGMMapping
	}
	if err := validateMapping(s.Mapping); err != nil {
		return err
	}
	for _, v := range []string{s.SourceID, s.Producer, s.ProducerVersion, s.NativeOrigin, s.FieldID, s.RawValue} {
		if len(v) == 0 || len(v) > 256 {
			return fmt.Errorf("invalid source identity or value")
		}
	}
	if _, err := strconv.ParseFloat(s.RawValue, 64); err != nil {
		return fmt.Errorf("invalid raw numeric value")
	}
	if len(s.Labels) > 16 {
		return fmt.Errorf("source label limit")
	}
	for k, v := range s.Labels {
		if !labelName(k) || len(k)+len(v) > 256 {
			return fmt.Errorf("invalid source label")
		}
	}
	if !s.HasExpositionTimestamp && s.ExpositionTimestampUnix != 0 {
		return fmt.Errorf("unknown source timestamp carries value")
	}
	if s.HasExpositionTimestamp && s.TimestampUnit != "milliseconds" && s.TimestampUnit != "seconds" {
		return fmt.Errorf("invalid exposition timestamp unit")
	}
	if !s.HasExpositionTimestamp && (s.RawTimestamp != "" || s.TimestampUnit != "") {
		return fmt.Errorf("unknown exposition timestamp carries metadata")
	}
	if s.HasExpositionTimestamp {
		rat, ok := boundedDecimal(s.RawTimestamp)
		scale := int64(1000000)
		if s.TimestampUnit == "seconds" {
			scale = 1000000000
		}
		if !ok {
			return fmt.Errorf("invalid raw timestamp")
		}
		rat.Mul(rat, big.NewRat(scale, 1))
		if !rat.IsInt() || !rat.Num().IsInt64() || rat.Num().Int64() != s.ExpositionTimestampUnix {
			return fmt.Errorf("contradictory raw timestamp")
		}
	}
	return nil
}

// ManagedField selects an explicit producer metric and canonical conversion.
// Numerator/Denominator must be positive. Integer fields require exact scaling.
type ManagedField struct {
	Metric, Name, Type     string
	Numerator, Denominator int64
	Integer                bool
	SourceUnit             string
}

// DCGMExporterFields returns mappings from the default counter configuration at
// dcgm-exporter revision fafd151148052628061a80450b4ee037a5fa0c3c.
// ProducerVersion must still identify the actual exporter that wrote the input.
func DCGMExporterFields() []ManagedField {
	return []ManagedField{
		{"DCGM_FI_DEV_SM_CLOCK", "sm_clock_mhz", "gauge", 1, 1, false, "megahertz"},
		{"DCGM_FI_DEV_MEM_CLOCK", "memory_clock_mhz", "gauge", 1, 1, false, "megahertz"},
		{"DCGM_FI_DEV_GPU_TEMP", "temperature_celsius", "gauge", 1, 1, false, "celsius"},
		{"DCGM_FI_DEV_POWER_USAGE", "power_milliwatts", "gauge", 1000, 1, false, "watts"},
		{"DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", "total_energy_millijoules", "counter", 1, 1, true, "millijoules"},
		{"DCGM_FI_DEV_GPU_UTIL", "gpu_utilization_pct", "gauge", 1, 1, false, "percent"},
		{"DCGM_FI_DEV_MEM_COPY_UTIL", "memory_utilization_pct", "gauge", 1, 1, false, "percent"},
		{"DCGM_FI_DEV_FB_USED", "memory_used_bytes", "gauge", 1048576, 1, true, "mebibytes"},
	}
}

// ManagedImport configures one bounded offline snapshot. Query bounds belong to
// the caller's collector clock; exposition timestamps never populate them.
type ManagedImport struct {
	SourceID, Producer, ProducerVersion, NativeOrigin string
	Fields                                            []ManagedField
	AllowedLabels                                     []string
	QueryStartNanos, QueryEndNanos, TimestampUnix     int64
	Exposition                                        ExpositionOptions
	MaxAgeNanos, MaxFutureNanos                       int64
}

// ReadManaged reads explicitly mapped fields into a canonical sample. Unknown
// fields are ignored. Missing fields remain absent; callers must not infer zero.
// Native collector ownership and network acquisition remain with the caller.
func ReadManaged(ctx context.Context, r io.Reader, c ManagedImport) (Sample, error) {
	var out Sample
	if c.QueryStartNanos < 0 || c.QueryEndNanos < c.QueryStartNanos || len(c.Fields) == 0 || len(c.Fields) > 128 {
		return out, fmt.Errorf("managed import: invalid bounds or fields")
	}
	if c.MaxAgeNanos < 0 || c.MaxFutureNanos < 0 || ((c.MaxAgeNanos > 0 || c.MaxFutureNanos > 0) && c.TimestampUnix <= 0) {
		return out, fmt.Errorf("managed import: invalid freshness policy")
	}
	fields := map[string]ManagedField{}
	for _, f := range c.Fields {
		if _, ok := fields[f.Metric]; ok {
			return out, fmt.Errorf("managed import: duplicate mapping")
		}
		if _, ok := LookupMetricUnit(f.Name); !ok || !metricName(f.Metric) || f.Numerator <= 0 || f.Denominator <= 0 {
			return out, fmt.Errorf("managed import: invalid mapping")
		}
		if err := validateManagedMapping(f); err != nil {
			return out, fmt.Errorf("managed import: %w", err)
		}
		fields[f.Metric] = f
	}
	allowed := map[string]bool{"UUID": true, "gpu": true, "device": true, "GPU_I_ID": true, "GPU_CI_ID": true, "modelName": true}
	if len(c.AllowedLabels) > 16 {
		return out, fmt.Errorf("managed import: label allowlist limit")
	}
	for _, k := range c.AllowedLabels {
		if !labelName(k) {
			return out, fmt.Errorf("managed import: invalid label allowlist")
		}
		allowed[k] = true
	}
	values, e := ReadExposition(ctx, r, c.Exposition)
	if e != nil {
		return out, e
	}
	out.QueryStartNanos = c.QueryStartNanos
	out.QueryEndNanos = c.QueryEndNanos
	seen := map[string]bool{}
	for _, v := range values {
		f, ok := fields[v.Name]
		if !ok {
			continue
		}
		if f.Type != "" && v.Type != f.Type {
			return Sample{}, fmt.Errorf("managed import: incompatible metric type for %s", v.Name)
		}
		if v.Unit != "" && (f.SourceUnit == "" || v.Unit != f.SourceUnit) {
			return Sample{}, fmt.Errorf("managed import: incompatible source unit")
		}
		uuid := v.Labels["UUID"]
		if uuid == "" {
			return Sample{}, fmt.Errorf("managed import: missing uuid")
		}
		key := uuid + "/" + f.Name
		if seen[key] {
			return Sample{}, fmt.Errorf("managed import: duplicate canonical source")
		}
		seen[key] = true
		source := &ReadingSource{Mapping: f, MappingDigest: managedMappingDigest(f), Schema: "nvtelemetry.managed.v1", SourceID: c.SourceID, Producer: c.Producer, ProducerVersion: c.ProducerVersion, NativeOrigin: c.NativeOrigin, FieldID: v.Name, RawValue: v.RawValue, Labels: map[string]string{}, ExpositionTimestampUnix: v.TimestampUnix, HasExpositionTimestamp: v.HasTimestamp, RawTimestamp: v.RawTimestamp, TimestampUnit: v.TimestampUnit}
		for k, x := range v.Labels {
			if allowed[k] {
				source.Labels[k] = x
			}
		}
		if e := ValidateReadingSource(source); e != nil {
			return Sample{}, fmt.Errorf("managed import: %w", e)
		}
		unit, _ := LookupMetricUnit(f.Name)
		rd := Reading{Name: f.Name, Unit: unit, Backend: c.Producer, DeviceUUID: uuid, Evidence: EvidenceMeasured, Source: source, QueryStartNanos: c.QueryStartNanos, QueryEndNanos: c.QueryEndNanos, HasQueryBounds: true, TimestampMonotonic: c.QueryEndNanos, TimestampUnix: c.TimestampUnix}
		if !v.HasTimestamp && (c.MaxAgeNanos > 0 || c.MaxFutureNanos > 0) {
			rd.Evidence = EvidenceUnavailable
			rd.UnavailableReason = "unknown_freshness"
			out.Readings = append(out.Readings, rd)
			continue
		}
		if v.HasTimestamp && ((c.MaxAgeNanos > 0 && v.TimestampUnix < c.TimestampUnix && uint64(c.TimestampUnix)-uint64(v.TimestampUnix) > uint64(c.MaxAgeNanos)) || (c.MaxFutureNanos > 0 && v.TimestampUnix > c.TimestampUnix && uint64(v.TimestampUnix)-uint64(c.TimestampUnix) > uint64(c.MaxFutureNanos))) {
			rd.Evidence = EvidenceUnavailable
			rd.UnavailableReason = ReasonStale
			out.Readings = append(out.Readings, rd)
			continue
		}
		x, err := strconv.ParseFloat(v.RawValue, 64)
		if err != nil {
			return Sample{}, err
		}
		if math.IsNaN(x) || math.IsInf(x, 0) {
			rd.Evidence = EvidenceUnavailable
			rd.UnavailableReason = "non_finite"
		} else if c.Producer == "dcgm-exporter" && x >= 140737488355328 && x <= 140737488355343 && !f.Integer {
			rd.Evidence = EvidenceUnavailable
			rd.UnavailableReason = dcgmBlankReason(uint64(x - 140737488355328))
		} else {
			rat, ok := boundedDecimal(v.RawValue)
			if !ok {
				return Sample{}, fmt.Errorf("managed import: unsupported numeric syntax")
			}
			if c.Producer == "dcgm-exporter" && f.Integer && rat.IsInt() && rat.Num().IsUint64() && rat.Num().Uint64() >= 0x7ffffffffffffff0 && rat.Num().Uint64() <= 0x7fffffffffffffff {
				rd.Evidence = EvidenceUnavailable
				rd.UnavailableReason = dcgmBlankReason(rat.Num().Uint64() - 0x7ffffffffffffff0)
				out.Readings = append(out.Readings, rd)
				continue
			}
			rat.Mul(rat, big.NewRat(f.Numerator, f.Denominator))
			if f.Integer {
				if !rat.IsInt() || !rat.Num().IsUint64() {
					return Sample{}, fmt.Errorf("managed import: integer conversion overflow or precision loss")
				}
				rd.IsInt = true
				rd.IntValue = rat.Num().Uint64()
				rd.Value = float64(rd.IntValue)
			} else {
				rd.Value, _ = rat.Float64()
				if math.IsInf(rd.Value, 0) {
					return Sample{}, fmt.Errorf("managed import: conversion overflow")
				}
			}
		}
		out.Readings = append(out.Readings, rd)
	}
	return out, nil
}

func dcgmBlankReason(offset uint64) string {
	switch offset {
	case 1:
		return ReasonQueryFailure
	case 2:
		return ReasonUnsupported
	case 3:
		return ReasonPermissionDenied
	}
	return ReasonBlank
}

func managedMappingDigest(f ManagedField) string {
	b, _ := json.Marshal(f)
	return fmt.Sprintf("sha256:%x", sha256.Sum256(b))
}

func validateManagedMapping(f ManagedField) error {
	kind, ok := LookupMetricKind(f.Name)
	if !ok || kind == KindBitmask || kind == KindEvent || (kind == KindCounter && !f.Integer) || (f.Type != "" && f.Type != kind) {
		return fmt.Errorf("unsupported metric kind, source type or inexact counter mapping")
	}
	return nil
}

func validateManagedValue(r Reading) error {
	s := r.Source
	if s == nil {
		return nil
	}
	if err := validateDCGMNative(r); err != nil {
		return err
	}
	f := s.Mapping
	unit, ok := LookupMetricUnit(f.Name)
	if !ok || f.Name != r.Name || unit != r.Unit {
		return fmt.Errorf("mapping metric contradiction")
	}
	if !r.IsAvailable() {
		return nil
	}
	x, ok := boundedDecimal(s.RawValue)
	if !ok {
		return fmt.Errorf("available raw value is not finite")
	}
	x.Mul(x, big.NewRat(f.Numerator, f.Denominator))
	if f.Integer {
		if !r.IsInt || !x.IsInt() || !x.Num().IsUint64() || r.IntValue != x.Num().Uint64() || r.Value != float64(r.IntValue) {
			return fmt.Errorf("integer mapping contradiction")
		}
	} else {
		v, _ := x.Float64()
		if r.IsInt || math.IsNaN(v) || math.IsInf(v, 0) || r.Value != v {
			return fmt.Errorf("float mapping contradiction")
		}
	}
	return nil
}
