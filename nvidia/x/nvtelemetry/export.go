package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"unicode"
	"unicode/utf8"
)

// ExportOptions selects a bounded snapshot encoding. Metrics use private names
// nvtelemetry.*, not the developmental OpenTelemetry hardware conventions.
// CounterStartUnixNanos must be explicitly known for cumulative OTLP counters.
type ExportOptions struct {
	UnixTimeMapping       *UnixTimeMapping
	HistogramBindings     []HistogramBinding
	MaxMetrics            int
	MaxExplicitBounds     int
	MaxReadings, MaxBytes int
	OpenMetrics           bool
	CounterStartUnixNanos int64
	HasCounterStart       bool
	// ClockDomain identifies the caller-declared fixed clock cohort. Empty is unknown.
	ClockDomain string
	// HasClockCalibration distinguishes a known zero bound from an unknown bound.
	// It requires ClockDomain; the caller supplies the calibration evidence.
	HasClockCalibration         bool
	ClockCalibrationBoundsNanos int64
}

func exportLimits(o ExportOptions) (int, int, error) {
	if !exportText(o.ClockDomain) || o.HasClockCalibration && o.ClockDomain == "" || o.ClockCalibrationBoundsNanos < 0 || !o.HasClockCalibration && o.ClockCalibrationBoundsNanos != 0 {
		return 0, 0, fmt.Errorf("export: invalid clock metadata")
	}
	n, b := o.MaxReadings, o.MaxBytes
	if n == 0 {
		n = 4096
	}
	if b == 0 {
		b = 1 << 20
	}
	if n < 1 || n > 65536 || b < 1 || b > 64<<20 {
		return 0, 0, fmt.Errorf("export: invalid limit")
	}
	return n, b, nil
}

func exportWrite(w io.Writer, data []byte, limit int) error {
	if w == nil {
		return fmt.Errorf("export: nil writer")
	}
	if len(data) > limit {
		return fmt.Errorf("export: byte limit")
	}
	n, e := w.Write(data)
	if e == nil && n != len(data) {
		e = io.ErrShortWrite
	}
	return e
}

func exportText(s string) bool {
	if len(s) > 256 || !utf8.ValidString(s) {
		return false
	}
	for _, c := range s {
		if !unicode.IsPrint(c) {
			return false
		}
	}
	return true
}

type exportLabel struct{ key, value string }

func exportLabels(r Reading, o ExportOptions) []exportLabel {
	labels := []exportLabel{
		{"device_uuid", r.DeviceUUID}, {"evidence", string(r.Evidence)},
		{"export_schema", "metrics.v1"}, {"backend_status", "unknown"},
		{"source_status", "unknown"}, {"native_origin_status", "unknown"},
		{"clock_domain_known", strconv.FormatBool(o.ClockDomain != "")},
		{"sensor_timestamp_known", strconv.FormatBool(r.HasSourceTimestamp)},
		{"wall_uncertainty_known", strconv.FormatBool(r.HasWallClockUncertainty)},
		{"calibration_known", strconv.FormatBool(o.HasClockCalibration)},
	}
	if r.Backend != "" {
		labels[3].value = "reported"
		labels = append(labels, exportLabel{"backend", r.Backend})
	}
	if r.Source != nil {
		labels[4].value = "reported"
		labels[5].value = "reported"
		labels = append(labels, exportLabel{"source_producer", r.Source.Producer}, exportLabel{"source_version", r.Source.ProducerVersion}, exportLabel{"source_native_origin", r.Source.NativeOrigin})
	}
	if o.ClockDomain != "" {
		labels = append(labels, exportLabel{"clock_domain", o.ClockDomain})
	}
	if r.HasSourceTimestamp {
		labels = append(labels, exportLabel{"sensor_clock_domain", r.SourceTimestampDomain})
	}
	return labels
}

func expositionLabels(labels []exportLabel) string {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, l := range labels {
		if i != 0 {
			b.WriteByte(',')
		}
		b.WriteString(l.key)
		b.WriteByte('=')
		b.WriteString(strconv.Quote(l.value))
	}
	b.WriteByte('}')
	return b.String()
}

func otlpAttributes(r Reading, o ExportOptions) []any {
	var attrs []any
	for _, l := range exportLabels(r, o) {
		key := "nvtelemetry." + l.key
		if l.key == "device_uuid" {
			key = "nvtelemetry.device.uuid"
		}
		attrs = append(attrs, map[string]any{"key": key, "value": map[string]string{"stringValue": l.value}})
	}
	add := func(key string, value int64) {
		attrs = append(attrs, map[string]any{"key": "nvtelemetry." + key, "value": map[string]string{"intValue": strconv.FormatInt(value, 10)}})
	}
	if r.HasSourceTimestamp {
		add("sensor_timestamp", r.SourceTimestamp)
	}
	if r.HasWallClockUncertainty {
		add("wall_uncertainty_nanos", r.WallClockUncertaintyNanos)
	}
	if o.HasClockCalibration {
		add("calibration_bounds_nanos", o.ClockCalibrationBoundsNanos)
	}
	return attrs
}

func exportReading(r Reading) error {
	if !exportText(r.Backend) || !exportText(r.SourceTimestampDomain) {
		return fmt.Errorf("export: invalid provenance")
	}
	if err := ValidateReadingSource(r.Source); err != nil {
		return fmt.Errorf("export: source: %w", err)
	}
	// Managed validators consume the legacy view, as in ReadStream's V3
	// admission. This view is never encoded; Number remains exact in output.
	managedReading := r
	if r.Number != nil && r.Source != nil {
		if r.IsInt || r.IntValue != 0 || r.Value != 0 {
			return fmt.Errorf("export: conflicting scalar representations")
		}
		if err := validNumber(r.Number); err != nil {
			return fmt.Errorf("export: invalid canonical number")
		}
		switch {
		case r.Number.Uint != nil:
			managedReading.IsInt = true
			managedReading.IntValue = *r.Number.Uint
			managedReading.Value = float64(*r.Number.Uint)
		case r.Number.Float != nil:
			managedReading.Value = *r.Number.Float
		default:
			return fmt.Errorf("export: signed managed mapping unsupported")
		}
	}
	if err := validateManagedValue(managedReading); err != nil {
		return fmt.Errorf("export: source value: %w", err)
	}
	if r.Source != nil {
		for _, s := range []string{r.Source.Producer, r.Source.ProducerVersion, r.Source.NativeOrigin} {
			if !exportText(s) {
				return fmt.Errorf("export: invalid source label")
			}
		}
	}
	if r.HasSourceTimestamp && r.SourceTimestampDomain == "" || !r.HasSourceTimestamp && (r.SourceTimestamp != 0 || r.SourceTimestampDomain != "") || r.WallClockUncertaintyNanos < 0 || !r.HasWallClockUncertainty && r.WallClockUncertaintyNanos != 0 {
		return fmt.Errorf("export: contradictory clock metadata")
	}

	if !utf8.ValidString(r.DeviceUUID) {
		return fmt.Errorf("export: invalid utf-8 device identity")
	}
	for _, c := range r.DeviceUUID {
		if !unicode.IsPrint(c) {
			return fmt.Errorf("export: control character in device identity")
		}
	}
	if (r.Evidence == EvidenceUnavailable) != (r.UnavailableReason != "") {
		return fmt.Errorf("export: contradictory availability")
	}
	kind, _ := LookupMetricKind(r.Name)
	if kind == KindCounter && !r.IsInt && r.Value < 0 {
		return fmt.Errorf("export: negative cumulative counter")
	}
	return nil
}

// WriteExposition emits canonical-unit private metrics and explicit availability
// gauges. Evidence and reported producer/backend metadata use at most 16 fixed
// labels; raw source IDs and labels are omitted. Clock presence flags distinguish
// known zero from unknown; numeric clock bounds remain in diagnostic JSON or OTLP.
// Reported native origin does not establish independent acquisition. The fixed
// exposition profile has device scope; entity-assigned readings require OTLP.
func WriteExposition(w io.Writer, readings []Reading, o ExportOptions) error {
	n, b, e := exportLimits(o)
	if e != nil {
		return e
	}
	if len(readings) > n {
		return fmt.Errorf("export: reading limit")
	}
	var out bytes.Buffer
	seen := map[string]bool{}
	declared := map[string]bool{}
	for _, r := range readings {
		if e := exportReading(r); e != nil {
			return e
		}
		if r.Observation != nil && (r.Observation.Assignment != nil || r.Observation.Scope == "process") {
			return fmt.Errorf("export: entity attribution requires OTLP")
		}
		if r.Evidence != EvidenceMeasured && r.Evidence != EvidenceDerived && r.Evidence != EvidenceUnavailable {
			return fmt.Errorf("export: invalid evidence")
		}
		if !metricName(r.Name) || r.DeviceUUID == "" || len(r.DeviceUUID) > 256 {
			return fmt.Errorf("export: invalid identity")
		}
		u, ok := LookupMetricUnit(r.Name)
		if !ok || u != r.Unit {
			return fmt.Errorf("export: invalid unit")
		}
		key := r.Name + "/" + r.DeviceUUID
		if seen[key] {
			return fmt.Errorf("export: duplicate series")
		}
		seen[key] = true
		label := expositionLabels(exportLabels(r, o))
		name := "nvtelemetry_" + r.Name
		kind, ok := LookupMetricKind(r.Name)
		if !ok || kind == KindBitmask || kind == KindEvent {
			return fmt.Errorf("export: unsupported metric kind")
		}
		family := name
		if kind == KindCounter {
			name += "_total"
			if !o.OpenMetrics {
				family = name
			}
		}
		if !declared[family] {
			fmt.Fprintf(&out, "# TYPE %s %s\n# TYPE %s_available gauge\n", family, kind, name)
			declared[family] = true
		}
		available := 0
		if r.IsAvailable() {
			available = 1
			_, value, err := exportScalar(r)
			if err != nil {
				return err
			}
			if kind == KindCounter && len(value) > 0 && value[0] == '-' && value != "-0" {
				return fmt.Errorf("export: negative cumulative counter")
			}

			fmt.Fprintf(&out, "%s%s %s\n", name, label, value)
		}
		fmt.Fprintf(&out, "%s_available%s %d\n", name, label, available)
		if out.Len() > b {
			return fmt.Errorf("export: byte limit")
		}
	}
	if o.OpenMetrics {
		out.WriteString("# EOF\n")
	}
	return exportWrite(w, out.Bytes(), b)
}

// WriteOTLPMetrics writes an OTLP/HTTP JSON ExportMetricsServiceRequest. Gauges
// have no invented start time. Cumulative counters require an explicit epoch.
// Bitmasks/events and unavailable readings are refused, not miscast as gauges.
// Resource attributes preserve bounded caller-asserted entity relationships.
// Owner attributes require explicit process scope, complete identity and a current
// matching clock epoch; missing attribution remains device-only. This does not
// divide or duplicate whole-device measurements across allocation recipients.
// Point attributes preserve evidence, bounded reported provenance, and clock
// presence. Known clock bounds use exact decimal intValue strings. Sensor
// timestamps retain their declared domain without an inferred unit. timeUnixNano
// remains the collector wall timestamp, not a sensor timestamp. Attribute names
// have the nvtelemetry. prefix followed by the exposition label name, except
// device_uuid, which uses nvtelemetry.device.uuid.
// Mapped V5 histograms allow 32 ordinary attributes plus the exact 13
// source-time attributes; this does not extend the closed attribute profile.
func WriteOTLPMetrics(w io.Writer, readings []Reading, o ExportOptions) error {
	histogram := len(o.HistogramBindings) > 0 || o.MaxMetrics != 0 || o.MaxExplicitBounds != 0 || o.UnixTimeMapping != nil
	for _, r := range readings {
		if r.Observation != nil && (r.Observation.Histogram != nil || r.Observation.OTLPScopeName != nil || r.Observation.OTLPClock != nil || r.Observation.OTLPSourceTime != nil) {
			histogram = true
		}
	}
	if histogram {
		return writeHistogramMetrics(w, readings, o)
	}
	return writeScalarOTLPMetrics(w, readings, o)
}

func writeScalarOTLPMetrics(w io.Writer, readings []Reading, o ExportOptions) error {
	n, b, e := exportLimits(o)
	if e != nil {
		return e
	}
	if len(readings) > n {
		return fmt.Errorf("export: reading limit")
	}
	resources := newOTLPEntityExport("Metrics", b)
	seen := map[string]bool{}
	for _, r := range readings {
		if e := exportReading(r); e != nil {
			return e
		}
		if r.Evidence != EvidenceMeasured && r.Evidence != EvidenceDerived {
			return fmt.Errorf("export: invalid or unavailable evidence")
		}
		if !r.IsAvailable() {
			return fmt.Errorf("export: unavailable reading requires diagnostic stream")
		}
		unit, ok := LookupMetricUnit(r.Name)
		if !ok || unit != r.Unit || r.TimestampUnix <= 0 || r.DeviceUUID == "" || len(r.DeviceUUID) > 256 {
			return fmt.Errorf("export: invalid identity, unit or timestamp")
		}
		attrs, resourceKey, err := exportEntityResource(r, o)
		if err != nil {
			return err
		}
		key := exportSeriesKey(r, resourceKey)
		if seen[key] {
			return fmt.Errorf("export: duplicate series")
		}
		seen[key] = true
		kind, ok := LookupMetricKind(r.Name)
		if !ok || kind == KindBitmask || kind == KindEvent {
			return fmt.Errorf("export: unsupported metric kind")
		}
		point := map[string]any{"timeUnixNano": strconv.FormatInt(r.TimestampUnix, 10), "attributes": otlpAttributes(r, o)}
		valueKind, value, err := exportScalar(r)
		if err != nil {
			return err
		}
		if kind == KindCounter && len(value) > 0 && value[0] == '-' && value != "-0" {
			return fmt.Errorf("export: negative cumulative counter")
		}
		if valueKind == "float" {
			point["asDouble"] = json.Number(value)
		} else {
			if valueKind == "uint" {
				n, _ := strconv.ParseUint(value, 10, 64)
				if n > math.MaxInt64 {
					return fmt.Errorf("export: otlp signed integer overflow")
				}
			}
			point["asInt"] = value
		}
		ucum := map[string]string{"percent": "%", "bytes": "By", "bytes_per_sec": "By/s", "milliwatts": "mW", "millijoules": "mJ", "celsius": "Cel", "mhz": "MHz", "joules": "J"}[unit]
		if ucum == "" {
			return fmt.Errorf("export: unsupported unit")
		}
		metric := map[string]any{"name": "nvtelemetry." + r.Name, "unit": ucum}
		points := []any{point}
		if kind == KindCounter {
			if !o.HasCounterStart || o.CounterStartUnixNanos <= 0 || o.CounterStartUnixNanos > r.TimestampUnix {
				return fmt.Errorf("export: unknown counter epoch")
			}
			point["startTimeUnixNano"] = strconv.FormatInt(o.CounterStartUnixNanos, 10)
			metric["sum"] = map[string]any{"dataPoints": points, "aggregationTemporality": 2, "isMonotonic": true}
		} else {
			metric["gauge"] = map[string]any{"dataPoints": points}
		}
		if err := resources.add(resourceKey, attrs, metric); err != nil {
			return err
		}
	}
	return resources.write(w)
}

// exportLogReason admits fixed status classes, never arbitrary driver text.
func exportLogReason(reason string) string {
	if safeUnavailableReason(reason) {
		return reason
	}
	switch reason {
	case "unsupported_field_type", "redfish_state_unknown", "redfish_reading_unavailable", "redfish_disabled", "redfish_absent", "redfish_unavailableoffline", "redfish_standbyoffline", "redfish_standbyspare", "redfish_starting", "redfish_updating":
		return reason
	}
	return "unknown"
}

// WriteOTLPLogs writes bounded diagnostic readings as an OTLP/HTTP JSON
// ExportLogsServiceRequest. The private nvtelemetry.logs.v1 body preserves events,
// bitmasks, unavailable status, and exact unsigned values as decimal strings.
// No severity or sensor time is inferred. An unknown collector wall timestamp
// omits timeUnixNano. Unavailable reasons retain fixed canonical status classes;
// unrecognized reasons become "unknown". Raw labels, source IDs, and native
// error text are omitted. Resource identity follows WriteOTLPMetrics. Canonical
// signed diagnostic values use value_int64; uint64 values never pass through floats.
func WriteOTLPLogs(w io.Writer, readings []Reading, o ExportOptions) error {
	n, b, err := exportLimits(o)
	if err != nil {
		return err
	}
	if len(readings) > n {
		return fmt.Errorf("export logs: reading limit")
	}
	resources := newOTLPEntityExport("Logs", b)
	for _, r := range readings {
		if err := exportReading(r); err != nil {
			return err
		}
		if r.Evidence != EvidenceMeasured && r.Evidence != EvidenceDerived && r.Evidence != EvidenceUnavailable {
			return fmt.Errorf("export logs: invalid evidence")
		}
		unit, ok := healthUnit(r.Name)
		if !ok || unit != r.Unit || r.DeviceUUID == "" || len(r.DeviceUUID) > 256 || r.TimestampUnix < 0 {
			return fmt.Errorf("export logs: invalid identity, unit or timestamp")
		}
		kind, ok := LookupMetricKind(r.Name)
		if !ok {
			return fmt.Errorf("export logs: unsupported metric kind")
		}
		if !r.IsAvailable() && r.Number != nil {
			return fmt.Errorf("export logs: unavailable reading has a number")
		}
		fields := []any{}
		add := func(key, value string) {
			fields = append(fields, map[string]any{"key": key, "value": map[string]string{"stringValue": value}})
		}
		add("schema", "nvtelemetry.logs.v1")
		add("name", r.Name)
		add("unit", r.Unit)
		add("kind", string(kind))
		add("status", "unavailable")
		if r.Evidence == EvidenceUnavailable {
			add("unavailable_reason", exportLogReason(r.UnavailableReason))
		}
		if r.IsAvailable() {
			fields[4] = map[string]any{"key": "status", "value": map[string]string{"stringValue": "available"}}
			valueKind, value, err := exportScalar(r)
			if err != nil {
				return err
			}
			if (kind == KindBitmask || kind == KindEvent) && valueKind != "uint" {
				return fmt.Errorf("export logs: event or bitmask requires unsigned integer")
			}
			if kind == KindCounter && len(value) > 0 && value[0] == '-' && value != "-0" {
				return fmt.Errorf("export logs: negative cumulative counter")
			}
			switch valueKind {
			case "uint":
				add("value_uint64", value)
			case "int":
				add("value_int64", value)
			default:
				add("value_float64", value)
			}
		}
		attrs := otlpAttributes(r, o)
		// The log envelope has a distinct private schema from numeric metrics.
		for _, a := range attrs {
			attr := a.(map[string]any)
			if attr["key"] == "nvtelemetry.export_schema" {
				attr["value"] = map[string]string{"stringValue": "logs.v1"}
			}
		}
		attrs = append(attrs, map[string]any{"key": "nvtelemetry.collector_timestamp_known", "value": map[string]string{"stringValue": strconv.FormatBool(r.TimestampUnix > 0)}})
		record := map[string]any{"body": map[string]any{"kvlistValue": map[string]any{"values": fields}}, "attributes": attrs}
		if r.TimestampUnix > 0 {
			record["timeUnixNano"] = strconv.FormatInt(r.TimestampUnix, 10)
		}
		resourceAttrs, resourceKey, err := exportEntityResource(r, o)
		if err != nil {
			return err
		}
		if err := resources.add(resourceKey, resourceAttrs, record); err != nil {
			return err
		}
	}
	return resources.write(w)
}
