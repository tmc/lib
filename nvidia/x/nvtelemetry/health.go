package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"

	"github.com/tmc/lib/ebpfhost/hostmetrics"
)

// HealthSource describes the availability of an explicitly configured reader.
// Its state describes the reader, not the health of its device.
type HealthSource struct {
	ID               string `json:"id"`
	Kind             string `json:"kind"`
	Version          string `json:"version"`
	State            string `json:"state"` // available, unsupported, denied, failed
	LastSuccessNanos int64  `json:"last_success_nanos"`
	HasLastSuccess   bool   `json:"has_last_success"`
}

// HealthObservation records raw evidence or a separately labeled judgment.
// Reading timestamps use the enclosing snapshot's clock domain and epoch.
type HealthObservation struct {
	SourceID    string  `json:"source_id"`
	Reading     Reading `json:"reading"`
	Class       string  `json:"class"` // raw, vendor_judgment, inference
	Judgment    string  `json:"judgment,omitempty"`
	Scope       string  `json:"scope,omitempty"`
	NativeError string  `json:"native_error,omitempty"`
}

// TopologyEdge is a directed, time-bounded reported relationship. Capability
// is a nominal description; it is never interpreted as measured traffic.
type TopologyEdge struct {
	SourceID        string `json:"source_id"`
	From            string `json:"from"`
	To              string `json:"to"`
	Kind            string `json:"kind"`
	Scope           string `json:"scope"`
	State           string `json:"state"` // available, unsupported, denied, failed
	Capability      string `json:"capability,omitempty"`
	ValidFromNanos  int64  `json:"valid_from_nanos"`
	ValidUntilNanos int64  `json:"valid_until_nanos"`
}

// HealthSnapshot is a read-only NVIDIA incident bundle. It has no action or
// remediation API. DeviceStates are reported explicitly; a failed source does
// not imply that a GPU disappeared. Node fields require an explicit GPU join.
type HealthSnapshot struct {
	Version       int                 `json:"version"`
	NodeID        string              `json:"node_id"`
	TopologyEpoch string              `json:"topology_epoch"`
	ClockDomain   string              `json:"clock_domain"`
	ClockEpoch    string              `json:"clock_epoch"`
	AtNanos       int64               `json:"at_nanos"`
	DeviceStates  map[string]string   `json:"device_states"` // present, lost, unknown
	Sources       []HealthSource      `json:"sources"`
	Observations  []HealthObservation `json:"observations"`
	Edges         []TopologyEdge      `json:"edges"`
	LostRecords   uint64              `json:"lost_records"`
	Complete      bool                `json:"complete"`
}

// HealthLimits bounds a bundle before decoding. All limits must be positive.
type HealthLimits struct {
	MaxBytes        int64
	MaxDevices      int
	MaxSources      int
	MaxObservations int
	MaxEdges        int
}

// ReadHealthSnapshot reads one version 1 bundle, rejecting unknown fields,
// excess input, invalid units and joins, duplicate evidence and contradictions.
func ReadHealthSnapshot(r io.Reader, limits HealthLimits) (*HealthSnapshot, error) {
	if r == nil {
		return nil, fmt.Errorf("health snapshot reader is nil")
	}
	if err := limits.validate(); err != nil {
		return nil, err
	}
	b, err := io.ReadAll(io.LimitReader(r, limits.MaxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read health snapshot: %w", err)
	}
	if int64(len(b)) > limits.MaxBytes {
		return nil, fmt.Errorf("health snapshot exceeds byte limit")
	}
	var s HealthSnapshot
	if err := decodeOperationsJSON(b, &s); err != nil {
		return nil, fmt.Errorf("decode health snapshot: %w", err)
	}
	if err := s.Validate(limits); err != nil {
		return nil, err
	}
	return &s, nil
}

// WriteHealthSnapshot validates and writes one bounded bundle. It rejects
// short writes; callers own the writer and its retention policy.
func WriteHealthSnapshot(w io.Writer, s HealthSnapshot, limits HealthLimits) error {
	if w == nil {
		return fmt.Errorf("health snapshot writer is nil")
	}
	if err := s.Validate(limits); err != nil {
		return err
	}
	b, err := json.Marshal(s)
	if err != nil {
		return fmt.Errorf("encode health snapshot: %w", err)
	}
	b = append(b, '\n')
	if int64(len(b)) > limits.MaxBytes {
		return fmt.Errorf("health snapshot exceeds byte limit")
	}
	n, err := w.Write(b)
	if err != nil {
		return fmt.Errorf("write health snapshot: %w", err)
	}
	if n != len(b) {
		return io.ErrShortWrite
	}
	return nil
}

func (l HealthLimits) validate() error {
	if l.MaxBytes <= 0 || l.MaxBytes == math.MaxInt64 || l.MaxDevices <= 0 || l.MaxSources <= 0 || l.MaxObservations <= 0 || l.MaxEdges <= 0 {
		return fmt.Errorf("health limits must be positive and bounded")
	}
	return nil
}

// Validate checks the snapshot contract without assigning health judgments.
func (s HealthSnapshot) Validate(l HealthLimits) error {
	if err := l.validate(); err != nil {
		return err
	}
	if s.Version != 1 || s.NodeID == "" || s.TopologyEpoch == "" || s.ClockDomain == "" || s.ClockEpoch == "" || s.AtNanos < 0 {
		return fmt.Errorf("invalid health snapshot identity or version")
	}
	if len(s.DeviceStates) == 0 || len(s.DeviceStates) > l.MaxDevices || len(s.Sources) > l.MaxSources || len(s.Observations) > l.MaxObservations || len(s.Edges) > l.MaxEdges {
		return fmt.Errorf("health snapshot exceeds record limits")
	}
	if s.Complete && s.LostRecords != 0 {
		return fmt.Errorf("complete health snapshot reports lost records")
	}
	for id, state := range s.DeviceStates {
		if id == "" || (state != "present" && state != "lost" && state != "unknown") {
			return fmt.Errorf("invalid device state for %q", id)
		}
	}
	sources := make(map[string]HealthSource)
	for _, source := range s.Sources {
		if source.ID == "" || source.Kind == "" || source.Version == "" || !healthSourceState(source.State) {
			return fmt.Errorf("invalid health source %q", source.ID)
		}
		if _, ok := sources[source.ID]; ok {
			return fmt.Errorf("duplicate health source %q", source.ID)
		}
		if source.HasLastSuccess && (source.LastSuccessNanos < 0 || source.LastSuccessNanos > s.AtNanos) {
			return fmt.Errorf("invalid health source time")
		}
		if !source.HasLastSuccess && source.LastSuccessNanos != 0 {
			return fmt.Errorf("health source time lacks presence flag")
		}
		sources[source.ID] = source
	}
	seen := make(map[string]bool)
	for _, o := range s.Observations {
		r := o.Reading
		source, ok := sources[o.SourceID]
		if !ok || (r.Source == nil && r.Backend != o.SourceID) || (r.Source != nil && (r.Source.SourceID != o.SourceID || r.Source.Producer != r.Backend || r.Source.ProducerVersion != source.Version)) {
			return fmt.Errorf("health observation source mismatch")
		}
		state, ok := s.DeviceStates[r.DeviceUUID]
		if !ok {
			return fmt.Errorf("health observation device mismatch")
		}
		if _, node := hostmetrics.Unit(r.Name); node {
			prefix := "node/" + s.NodeID + "/"
			if !strings.HasPrefix(o.Scope, prefix) || len(o.Scope) == len(prefix) || r.HasSourceTimestamp {
				return fmt.Errorf("invalid node observation scope or sensor timestamp")
			}
			if !r.HasQueryBounds || r.TimestampMonotonic != r.QueryEndNanos {
				return fmt.Errorf("node observation requires query bounds")
			}
			check := hostmetrics.Snapshot{Version: 1, Context: hostmetrics.Context{NodeID: s.NodeID, SourceID: o.SourceID, Version: source.Version, ClockDomain: s.ClockDomain, ClockEpoch: s.ClockEpoch, QueryStartNanos: r.QueryStartNanos, QueryEndNanos: r.QueryEndNanos}, Metrics: []hostmetrics.Metric{{Name: r.Name, Unit: r.Unit, Resource: strings.TrimPrefix(o.Scope, prefix), Value: r.Value, Integer: r.IsInt, UintValue: r.IntValue}}}
			if err := check.Validate(); err != nil {
				return fmt.Errorf("node observation: %w", err)
			}
		}
		unit, known := healthUnit(r.Name)
		if !known || unit != r.Unit {
			return fmt.Errorf("invalid health field unit for %q", r.Name)
		}
		if err := ValidateReadingSource(r.Source); err != nil {
			return fmt.Errorf("health reading source: %w", err)
		}
		if err := validateManagedValue(r); err != nil {
			return fmt.Errorf("health mapped reading: %w", err)
		}
		if o.Class != "raw" && o.Class != "vendor_judgment" && o.Class != "inference" {
			return fmt.Errorf("invalid health evidence class")
		}
		if (o.Class == "raw") != (o.Judgment == "") {
			return fmt.Errorf("health judgment class mismatch")
		}
		if r.Evidence != EvidenceMeasured && r.Evidence != EvidenceUnavailable && r.Evidence != EvidenceDerived {
			return fmt.Errorf("invalid health reading evidence")
		}
		if o.Class == "raw" && r.Evidence == EvidenceDerived {
			return fmt.Errorf("derived health reading is not raw")
		}
		if r.Evidence == EvidenceUnavailable {
			if r.UnavailableReason == "" || r.Value != 0 || r.IntValue != 0 {
				return fmt.Errorf("invalid unavailable health reading")
			}
		} else if r.UnavailableReason != "" || source.State != "available" || state != "present" || math.IsNaN(r.Value) || math.IsInf(r.Value, 0) {
			return fmt.Errorf("health value contradicts source or device state")
		}
		if r.TimestampMonotonic < 0 || r.TimestampMonotonic > s.AtNanos || (r.HasQueryBounds && (r.QueryStartNanos < 0 || r.QueryEndNanos < r.QueryStartNanos || r.QueryEndNanos > s.AtNanos)) {
			return fmt.Errorf("invalid health observation time")
		}
		if (!r.IsInt && r.IntValue != 0) || (!r.HasQueryBounds && (r.QueryStartNanos != 0 || r.QueryEndNanos != 0)) || (!r.HasSourceTimestamp && (r.SourceTimestamp != 0 || r.SourceTimestampDomain != "")) || (r.HasSourceTimestamp && r.SourceTimestampDomain == "") || (r.HasWallClockUncertainty && r.WallClockUncertaintyNanos < 0) || (!r.HasWallClockUncertainty && r.WallClockUncertaintyNanos != 0) {
			return fmt.Errorf("health reading has contradictory presence flags")
		}
		if r.IsInt && r.Value != 0 && r.Value != float64(r.IntValue) {
			return fmt.Errorf("health integer float copy contradicts exact value")
		}
		if r.HasSourceTimestamp && r.SourceTimestamp < 0 {
			return fmt.Errorf("health source timestamp is negative")
		}
		if unit == "boolean" && r.Evidence != EvidenceUnavailable && ((r.IsInt && r.IntValue > 1) || (!r.IsInt && r.Value != 0 && r.Value != 1)) {
			return fmt.Errorf("invalid health boolean")
		}
		key := fmt.Sprintf("%s\x00%s\x00%s\x00%d\x00%s\x00%s", o.SourceID, r.DeviceUUID, r.Name, r.TimestampMonotonic, o.Class, o.Scope)
		if seen[key] {
			return fmt.Errorf("duplicate health observation")
		}
		seen[key] = true
	}
	seen = make(map[string]bool)
	for _, e := range s.Edges {
		source, ok := sources[e.SourceID]
		if !ok || e.From == e.To || e.Kind == "" || e.Scope == "" || !healthSourceState(e.State) {
			return fmt.Errorf("invalid topology edge")
		}
		_, from := s.DeviceStates[e.From]
		_, to := s.DeviceStates[e.To]
		if !from || !to {
			return fmt.Errorf("topology device mismatch")
		}
		if e.State == "available" && (s.DeviceStates[e.From] != "present" || s.DeviceStates[e.To] != "present") {
			return fmt.Errorf("topology edge contradicts device state")
		}
		if e.ValidFromNanos < 0 || e.ValidUntilNanos < e.ValidFromNanos || (e.State == "available" && (source.State != "available" || s.AtNanos < e.ValidFromNanos || s.AtNanos >= e.ValidUntilNanos)) {
			return fmt.Errorf("topology edge outside validity or source unavailable")
		}
		key := e.SourceID + "\x00" + e.From + "\x00" + e.To + "\x00" + e.Kind + "\x00" + e.Scope
		if seen[key] {
			return fmt.Errorf("duplicate topology edge")
		}
		seen[key] = true
	}
	return nil
}

func healthSourceState(s string) bool {
	return s == "available" || s == "unsupported" || s == "denied" || s == "failed"
}

func healthUnit(name string) (string, bool) {
	if unit, ok := hostmetrics.Unit(name); ok {
		return unit, true
	}
	if unit, ok := LookupMetricUnit(name); ok {
		return unit, true
	}
	unit, ok := map[string]string{
		MetricXIDEvent: "code", MetricThrottleReasons: UnitBitmask,
		"ecc_corrected_errors": "count", "ecc_uncorrected_errors": "count", "remapped_rows": "count", "repair_pending": "boolean",
		"corrected_remapped_rows": "count", "uncorrected_remapped_rows": "count", "remap_failure": "boolean",
		"pcie_link_width": "lanes", "pcie_link_generation": "generation", "nvlink_state": "boolean", "fabric_health": "code",
		"fabric_state": "code", "fabric_status": "code", "fabric_clique_id": "code", "fabric_health_mask": UnitBitmask, "fabric_health_summary": "code",
		"imex_domain_status": "code", "imex_daemon_status": "code",
		"cpu_pressure_pct": UnitPercent, "host_memory_available_bytes": UnitBytes, "rdma_errors": "count", "network_rx_bytes_per_sec": UnitBytesPerSec,
		"storage_available_bytes": UnitBytes, "chassis_temperature_celsius": UnitCelsius,
	}[name]
	return unit, ok
}

// QueryStale reports whether a reader's last successful query is too old.
// It does not establish sensor freshness or convert query time to sensor time.
func (s HealthSource) QueryStale(nowNanos, maxAgeNanos int64) bool {
	return maxAgeNanos < 0 || nowNanos < 0 || s.LastSuccessNanos < 0 || !s.HasLastSuccess || nowNanos < s.LastSuccessNanos || nowNanos-s.LastSuccessNanos > maxAgeNanos
}

func decodeOperationsJSON(b []byte, v any) error {
	keys := json.NewDecoder(bytes.NewReader(b))
	if err := checkOperationsJSON(keys, 0, reflect.TypeOf(v)); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(v); err != nil {
		return err
	}
	var trailing any
	if err := d.Decode(&trailing); err != io.EOF {
		return fmt.Errorf("unexpected trailing json")
	}
	return nil
}

func checkOperationsJSON(d *json.Decoder, depth int, typ reflect.Type) error {
	if depth > 64 {
		return fmt.Errorf("json nesting exceeds limit")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	open, ok := t.(json.Delim)
	if !ok {
		if t == nil && typ != nil {
			for typ.Kind() == reflect.Pointer {
				typ = typ.Elem()
			}
			if typ.Kind() != reflect.Slice && typ.Kind() != reflect.Map {
				return fmt.Errorf("null json scalar or object")
			}
		}
		return nil
	}
	if typ != nil {
		for typ.Kind() == reflect.Pointer {
			typ = typ.Elem()
		}
	}
	if open == '{' {
		fields := make(map[string]reflect.Type)
		if typ != nil && typ.Kind() == reflect.Struct {
			for i := 0; i < typ.NumField(); i++ {
				field := typ.Field(i)
				name := strings.Split(field.Tag.Get("json"), ",")[0]
				if name == "" && field.PkgPath == "" {
					name = field.Name
				}
				if name != "" && name != "-" {
					fields[name] = field.Type
				}
			}
		}
		seen := make(map[string]bool)
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || seen[name] {
				return fmt.Errorf("duplicate or invalid json key")
			}
			seen[name] = true
			if typ != nil && typ.Kind() == reflect.Struct {
				if _, known := fields[name]; !known {
					return fmt.Errorf("noncanonical or unknown json field name %q", name)
				}
			}
			next := fields[name]
			if typ != nil && typ.Kind() == reflect.Map {
				next = typ.Elem()
			}
			if err := checkOperationsJSON(d, depth+1, next); err != nil {
				return err
			}
		}
	} else if open == '[' {
		var next reflect.Type
		if typ != nil && (typ.Kind() == reflect.Slice || typ.Kind() == reflect.Array) {
			next = typ.Elem()
		}
		for d.More() {
			if err := checkOperationsJSON(d, depth+1, next); err != nil {
				return err
			}
		}
	} else {
		return fmt.Errorf("invalid json delimiter")
	}
	_, err = d.Token()
	return err
}
