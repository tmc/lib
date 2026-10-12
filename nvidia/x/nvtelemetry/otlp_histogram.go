package nvtelemetry

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

// HistogramBinding binds one private canonical metric name to its unscaled unit.
// Names use [a-z][a-z0-9_]{0,127} and must not shadow a scalar registry entry.
type HistogramBinding struct{ Name, Unit string }

// OTLPReadOptions bounds one restricted OTLP histogram request. Zero selects
// defaults: 1 MiB, 4096 metrics, 4096 points and 256 explicit bounds. Maxima are
// 64 MiB, 65536 metrics, 65536 points and 4096 explicit bounds, independently.
type OTLPReadOptions struct {
	HistogramBindings []HistogramBinding
	MaxBytes          int64
	MaxMetrics        int
	MaxReadings       int
	MaxExplicitBounds int
}

// OTLPHistogramReadResult contains owned canonical readings and the digest of
// the complete accepted current entity. TransportSHA256 is not producer lineage
// and never populates Observation.RawSHA256 or a point attribute.
type OTLPHistogramReadResult struct {
	Readings        []Reading
	TransportSHA256 string
}

var histogramUnits = map[string]string{
	"percent": "%", "bytes": "By", "bytes_per_sec": "By/s", "milliwatts": "mW",
	"millijoules": "mJ", "celsius": "Cel", "mhz": "MHz", "joules": "J", "seconds": "s",
}

func histogramBindings(bindings []HistogramBinding) (map[string]string, error) {
	out := make(map[string]string, len(bindings))
	for _, b := range bindings {
		if len(b.Name) < 1 || len(b.Name) > 128 || b.Name[0] < 'a' || b.Name[0] > 'z' {
			return nil, errors.New("otlp histogram: invalid binding name")
		}
		for _, c := range b.Name {
			if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_') {
				return nil, errors.New("otlp histogram: invalid binding name")
			}
		}
		if _, ok := out[b.Name]; ok {
			return nil, errors.New("otlp histogram: duplicate binding")
		}
		if _, ok := LookupMetricKind(b.Name); ok {
			return nil, errors.New("otlp histogram: scalar binding collision")
		}
		if histogramUnits[b.Unit] == "" {
			return nil, errors.New("otlp histogram: unsupported unit")
		}
		out[b.Name] = b.Unit
	}
	return out, nil
}

func histogramLimit(n, def, max int) (int, error) {
	if n == 0 {
		n = def
	}
	if n < 1 || n > max {
		return 0, errors.New("otlp histogram: invalid limit")
	}
	return n, nil
}

type otlpObject map[string]any

func (m otlpObject) only(keys ...string) error {
	for k := range m {
		found := false
		for _, want := range keys {
			if k == want {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("otlp histogram: unsupported field %q", k)
		}
	}
	return nil
}
func otlpMap(v any) (otlpObject, error) {
	m, ok := v.(otlpObject)
	if !ok {
		return nil, errors.New("otlp histogram: expected object")
	}
	return m, nil
}
func otlpArray(m otlpObject, k string) ([]any, error) {
	v, ok := m[k]
	if !ok {
		return nil, nil
	}
	a, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("otlp histogram: expected array %q", k)
	}
	return a, nil
}
func otlpText(m otlpObject, k string) (string, error) {
	v, ok := m[k]
	if !ok {
		return "", nil
	}
	s, ok := v.(string)
	if !ok || !exportText(s) {
		return "", fmt.Errorf("otlp histogram: invalid string %q", k)
	}
	return s, nil
}
func otlpUint(v any, allowString bool) (uint64, error) {
	var s string
	switch x := v.(type) {
	case string:
		if !allowString {
			return 0, errors.New("otlp histogram: integer token required")
		}
		s = x
	case json.Number:
		s = string(x)
	default:
		return 0, errors.New("otlp histogram: integer required")
	}
	if s == "" || len(s) > 20 || len(s) > 1 && s[0] == '0' {
		return 0, errors.New("otlp histogram: noncanonical unsigned integer")
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, errors.New("otlp histogram: noncanonical unsigned integer")
		}
	}
	v64, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		return 0, errors.New("otlp histogram: unsigned integer overflow")
	}
	return v64, nil
}
func otlpZeroDrops(m otlpObject) error {
	if v, ok := m["droppedAttributesCount"]; ok {
		n, e := otlpUint(v, false)
		if e != nil || n > math.MaxUint32 || n != 0 {
			return errors.New("otlp histogram: producer attribute loss")
		}
	}
	return nil
}
func otlpEmptyAttributes(m otlpObject) error {
	a, e := otlpArray(m, "attributes")
	if e != nil {
		return e
	}
	if len(a) != 0 {
		return errors.New("otlp histogram: unsupported resource/scope attributes")
	}
	return otlpZeroDrops(m)
}

// decodeOTLPValue retains exact numeric tokens and rejects every present null,
// duplicate/case-alias key and excessive nesting before typed admission.
func decodeOTLPValue(ctx context.Context, d *json.Decoder, depth int) (any, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if depth > 32 {
		return nil, errors.New("otlp histogram: JSON depth limit")
	}
	token, err := d.Token()
	if err != nil {
		return nil, err
	}
	if token == nil {
		return nil, errors.New("otlp histogram: null property")
	}
	delim, ok := token.(json.Delim)
	if !ok {
		if s, ok := token.(string); ok && !exportText(s) {
			return nil, errors.New("otlp histogram: string limit or encoding")
		}
		return token, nil
	}
	switch delim {
	case '{':
		m := otlpObject{}
		seen := map[string]bool{}
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return nil, err
			}
			k, ok := key.(string)
			if !ok || !exportText(k) {
				return nil, errors.New("otlp histogram: invalid key")
			}
			fold := strings.ToLower(k)
			if seen[fold] {
				return nil, errors.New("otlp histogram: duplicate/case-alias key")
			}
			seen[fold] = true
			v, e := decodeOTLPValue(ctx, d, depth+1)
			if e != nil {
				return nil, e
			}
			m[k] = v
		}
		_, err = d.Token()
		return m, err
	case '[':
		a := []any{}
		for d.More() {
			v, e := decodeOTLPValue(ctx, d, depth+1)
			if e != nil {
				return nil, e
			}
			a = append(a, v)
		}
		_, err = d.Token()
		return a, err
	}
	return nil, errors.New("otlp histogram: invalid JSON delimiter")
}

func readOTLPEntity(ctx context.Context, r io.Reader, limit int64) ([]byte, error) {
	if ctx == nil || r == nil {
		return nil, errors.New("otlp histogram: nil context or reader")
	}
	var out bytes.Buffer
	var buf [32768]byte
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		remaining := limit + 1 - int64(out.Len())
		if remaining <= 0 {
			return nil, errors.New("otlp histogram: byte limit")
		}
		size := len(buf)
		if int64(size) > remaining {
			size = int(remaining)
		}
		n, err := r.Read(buf[:size])
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		if n < 0 || n > size {
			return nil, errors.New("otlp histogram: invalid reader count")
		}
		out.Write(buf[:n])
		if int64(out.Len()) > limit {
			return nil, errors.New("otlp histogram: byte limit")
		}
		if err == io.EOF {
			return out.Bytes(), nil
		}
		if err != nil {
			return nil, fmt.Errorf("otlp histogram: read: %w", err)
		}
		if n == 0 { // io.Reader permits empty reads; cancellation is checked again.
			continue
		}
	}
}

// ReadOTLPHistogramMetrics reads one bounded, strict OTLP/HTTP JSON histogram
// request with exactly one resource and instrumentation scope. It accepts only
// the reversible private attribute mapping; no vendor or native origin is inferred.
// The reader must unblock cooperatively when ctx is canceled.
func ReadOTLPHistogramMetrics(ctx context.Context, r io.Reader, o OTLPReadOptions) (OTLPHistogramReadResult, error) {
	fail := func(e error) (OTLPHistogramReadResult, error) { return OTLPHistogramReadResult{}, e }
	bindings, err := histogramBindings(o.HistogramBindings)
	if err != nil {
		return fail(err)
	}
	metricsMax, err := histogramLimit(o.MaxMetrics, 4096, 65536)
	if err != nil {
		return fail(err)
	}
	readingsMax, err := histogramLimit(o.MaxReadings, 4096, 65536)
	if err != nil {
		return fail(err)
	}
	boundsMax, err := histogramLimit(o.MaxExplicitBounds, 256, 4096)
	if err != nil {
		return fail(err)
	}
	limit := o.MaxBytes
	if limit == 0 {
		limit = 1 << 20
	}
	if limit < 1 || limit > 64<<20 {
		return fail(errors.New("otlp histogram: invalid byte limit"))
	}
	raw, err := readOTLPEntity(ctx, r, limit)
	if err != nil {
		return fail(err)
	}
	if !utf8.Valid(raw) {
		return fail(errors.New("otlp histogram: invalid UTF-8"))
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	value, err := decodeOTLPValue(ctx, d, 0)
	if err != nil {
		return fail(err)
	}
	if _, err = d.Token(); err != io.EOF {
		return fail(errors.New("otlp histogram: trailing JSON data"))
	}
	envelope, err := otlpMap(value)
	if err != nil {
		return fail(err)
	}
	if err = envelope.only("resourceMetrics"); err != nil {
		return fail(err)
	}
	resources, err := otlpArray(envelope, "resourceMetrics")
	if err != nil || len(resources) != 1 {
		return fail(errors.New("otlp histogram: exactly one resource required"))
	}
	rm, err := otlpMap(resources[0])
	if err != nil {
		return fail(err)
	}
	if err = rm.only("resource", "scopeMetrics", "schemaUrl"); err != nil {
		return fail(err)
	}
	url, err := otlpText(rm, "schemaUrl")
	if err != nil || url != "" {
		return fail(errors.New("otlp histogram: unsupported resource schema"))
	}
	if v, ok := rm["resource"]; ok {
		resource, e := otlpMap(v)
		if e != nil {
			return fail(e)
		}
		if e = resource.only("attributes", "droppedAttributesCount"); e != nil {
			return fail(e)
		}
		if e = otlpEmptyAttributes(resource); e != nil {
			return fail(e)
		}
	}
	scopes, err := otlpArray(rm, "scopeMetrics")
	if err != nil || len(scopes) != 1 {
		return fail(errors.New("otlp histogram: exactly one scope required"))
	}
	sm, err := otlpMap(scopes[0])
	if err != nil {
		return fail(err)
	}
	if err = sm.only("scope", "metrics", "schemaUrl"); err != nil {
		return fail(err)
	}
	url, err = otlpText(sm, "schemaUrl")
	if err != nil || url != "" {
		return fail(errors.New("otlp histogram: unsupported scope schema"))
	}
	scope := ""
	if v, ok := sm["scope"]; ok {
		s, e := otlpMap(v)
		if e != nil {
			return fail(e)
		}
		if e = s.only("name", "version", "attributes", "droppedAttributesCount"); e != nil {
			return fail(e)
		}
		scope, e = otlpText(s, "name")
		if e != nil || scope != "" && scope != otlpScopeName {
			return fail(errors.New("otlp histogram: unsupported scope name"))
		}
		version, e := otlpText(s, "version")
		if e != nil || version != "" {
			return fail(errors.New("otlp histogram: unsupported scope version"))
		}
		if e = otlpEmptyAttributes(s); e != nil {
			return fail(e)
		}
	}
	metrics, err := otlpArray(sm, "metrics")
	if err != nil {
		return fail(err)
	}
	if len(metrics) > metricsMax {
		return fail(errors.New("otlp histogram: metric limit"))
	}
	readings := []Reading{}
	seen := map[[2]string]bool{}
	temporalities := map[string]string{}
	var clock *OTLPClockMetadata
	for _, v := range metrics {
		if err = ctx.Err(); err != nil {
			return fail(err)
		}
		metric, e := otlpMap(v)
		if e != nil {
			return fail(e)
		}
		if e = metric.only("name", "unit", "description", "metadata", "histogram"); e != nil {
			return fail(e)
		}
		description, e := otlpText(metric, "description")
		if e != nil || description != "" {
			return fail(errors.New("otlp histogram: unsupported description"))
		}
		metadata, e := otlpArray(metric, "metadata")
		if e != nil || len(metadata) > 0 {
			return fail(errors.New("otlp histogram: unsupported metadata"))
		}
		wire, e := otlpText(metric, "name")
		if e != nil {
			return fail(e)
		}
		name := strings.TrimPrefix(wire, "nvtelemetry.")
		unit, ok := bindings[name]
		if !ok || wire != "nvtelemetry."+name {
			return fail(errors.New("otlp histogram: unbound metric"))
		}
		wireUnit, e := otlpText(metric, "unit")
		if e != nil || wireUnit != histogramUnits[unit] {
			return fail(errors.New("otlp histogram: unit conflict"))
		}
		histogram, e := otlpMap(metric["histogram"])
		if e != nil {
			return fail(e)
		}
		if e = histogram.only("dataPoints", "aggregationTemporality"); e != nil {
			return fail(e)
		}
		temp, e := otlpText(histogram, "aggregationTemporality")
		if e != nil {
			return fail(e)
		}
		switch temp {
		case "AGGREGATION_TEMPORALITY_DELTA":
			temp = "delta"
		case "AGGREGATION_TEMPORALITY_CUMULATIVE":
			temp = "cumulative"
		default:
			return fail(errors.New("otlp histogram: unsupported temporality"))
		}
		if old, ok := temporalities[name]; ok && old != temp {
			return fail(errors.New("otlp histogram: conflicting temporality"))
		}
		temporalities[name] = temp
		points, e := otlpArray(histogram, "dataPoints")
		if e != nil {
			return fail(e)
		}
		if len(points) > readingsMax-len(readings) {
			return fail(errors.New("otlp histogram: reading limit"))
		}
		for _, v := range points {
			if err = ctx.Err(); err != nil {
				return fail(err)
			}
			point, e := otlpMap(v)
			if e != nil {
				return fail(e)
			}
			reading, e := readHistogramPoint(point, name, unit, temp, scope, boundsMax)
			if e != nil {
				return fail(e)
			}
			key := [2]string{reading.Name, reading.DeviceUUID}
			if seen[key] {
				return fail(errors.New("otlp histogram: duplicate series"))
			}
			seen[key] = true
			c := reading.Observation.OTLPClock
			if clock != nil && *clock != *c {
				return fail(errors.New("otlp histogram: conflicting clock metadata"))
			}
			clock = c
			readings = append(readings, reading)
		}
	}
	if err = ctx.Err(); err != nil {
		return fail(err)
	}
	digest := sha256.Sum256(raw)
	return OTLPHistogramReadResult{Readings: readings, TransportSHA256: hex.EncodeToString(digest[:])}, nil
}

func readHistogramPoint(p otlpObject, name, unit, temp, scope string, boundsMax int) (Reading, error) {
	if err := p.only("attributes", "startTimeUnixNano", "timeUnixNano", "count", "sum", "bucketCounts", "explicitBounds", "exemplars", "flags", "min", "max"); err != nil {
		return Reading{}, err
	}
	exemplars, err := otlpArray(p, "exemplars")
	if err != nil {
		return Reading{}, err
	}
	if len(exemplars) > 0 {
		return Reading{}, errors.New("otlp histogram: unsupported exemplars")
	}
	count := uint64(0)
	if v, ok := p["count"]; ok {
		count, err = otlpUint(v, true)
		if err != nil {
			return Reading{}, err
		}
	}
	end, err := otlpUint(p["timeUnixNano"], true)
	if err != nil || end == 0 || end > math.MaxInt64 {
		return Reading{}, errors.New("otlp histogram: invalid Unix end")
	}
	h := &Histogram{Count: &count, Temporality: temp, Bounds: []string{}, Counts: []uint64{}}
	if v, ok := p["flags"]; ok {
		flags, e := otlpUint(v, false)
		if e != nil || flags > math.MaxUint32 {
			return Reading{}, errors.New("otlp histogram: invalid flags")
		}
		h.Flags = uint32(flags)
	}
	for _, entry := range []struct {
		key string
		dst **float64
	}{{"sum", &h.Sum}, {"min", &h.Min}, {"max", &h.Max}} {
		if v, ok := p[entry.key]; ok {
			number, ok := v.(json.Number)
			if !ok {
				return Reading{}, errors.New("otlp histogram: summary number required")
			}
			x, e := number.Float64()
			if e != nil || math.IsInf(x, 0) || math.IsNaN(x) {
				return Reading{}, errors.New("otlp histogram: invalid summary")
			}
			*entry.dst = &x
		}
	}
	bounds, err := otlpArray(p, "explicitBounds")
	if err != nil {
		return Reading{}, err
	}
	if len(bounds) > boundsMax {
		return Reading{}, errors.New("otlp histogram: explicit bound limit")
	}
	buckets, err := otlpArray(p, "bucketCounts")
	if err != nil {
		return Reading{}, err
	}
	if len(bounds) > 0 || len(buckets) > 0 {
		if len(buckets) != len(bounds)+1 {
			return Reading{}, errors.New("otlp histogram: bucket length mismatch")
		}
		for i, b := range bounds {
			token, ok := b.(json.Number)
			if !ok {
				return Reading{}, errors.New("otlp histogram: bound number required")
			}
			v, e := histogramBound(string(token))
			if e != nil {
				return Reading{}, e
			}
			if i > 0 {
				prev, _ := strconv.ParseFloat(h.Bounds[i-1], 64)
				if v <= prev {
					return Reading{}, errors.New("otlp histogram: collapsed or unordered bounds")
				}
			}
			h.Bounds = append(h.Bounds, strconv.FormatFloat(v, 'g', -1, 64))
		}
		h.Bounds = append(h.Bounds, "+Inf")
		total := uint64(0)
		for _, b := range buckets {
			v, e := otlpUint(b, true)
			if e != nil {
				return Reading{}, e
			}
			if v > math.MaxUint64-total {
				return Reading{}, errors.New("otlp histogram: bucket count overflow")
			}
			total += v
			h.Counts = append(h.Counts, total)
		}
		if total != count {
			return Reading{}, errors.New("otlp histogram: count mismatch")
		}
	}
	r := Reading{Name: name, Unit: unit, TimestampUnix: int64(end), Observation: &Observation{Histogram: h, OTLPScopeName: &scope, Observer: Observer{SamplingCoverage: &SamplingCoverage{Status: "unknown"}}}}
	if v, ok := p["startTimeUnixNano"]; ok {
		start, e := otlpUint(v, true)
		if e != nil || start == 0 || start > end {
			return Reading{}, errors.New("otlp histogram: invalid Unix start")
		}
		r.Observation.Interval = &tracecontract.Interval{Clock: "unix", Start: int64(start), End: int64(end)}
	}
	attrs, e := readHistogramAttributes(p)
	if e != nil {
		return Reading{}, e
	}
	if e = applyHistogramAttributes(&r, attrs); e != nil {
		return Reading{}, e
	}
	if h.Flags == 1 {
		r.UnavailableReason = "no_recorded_value"
	}
	if e = validHistogramReading(r); e != nil {
		return Reading{}, e
	}
	expected, e := histogramAttributes(r)
	if e != nil {
		return Reading{}, e
	}
	if !reflect.DeepEqual(attrs, expected) {
		return Reading{}, errors.New("otlp histogram: missing, unsupported or inconsistent attributes")
	}
	return r, nil
}

// Attribute values use their exact OTLP primitive arm. This representation is
// internal wire data; Reading/Observation remain the only canonical value model.
type histogramAttribute struct {
	arm   string
	value any
}

func readHistogramAttributes(p otlpObject) (map[string]histogramAttribute, error) {
	list, err := otlpArray(p, "attributes")
	if err != nil {
		return nil, err
	}
	if len(list) > 32 {
		return nil, errors.New("otlp histogram: attribute limit")
	}
	attrs := map[string]histogramAttribute{}
	seen := map[string]bool{}
	for _, v := range list {
		a, e := otlpMap(v)
		if e != nil {
			return nil, e
		}
		if e = a.only("key", "value"); e != nil {
			return nil, e
		}
		key, e := otlpText(a, "key")
		if e != nil || key == "" {
			return nil, errors.New("otlp histogram: invalid attribute key")
		}
		fold := strings.ToLower(key)
		if seen[fold] {
			return nil, errors.New("otlp histogram: duplicate/case-alias attribute")
		}
		seen[fold] = true
		value, e := otlpMap(a["value"])
		if e != nil {
			return nil, e
		}
		if len(value) != 1 {
			return nil, errors.New("otlp histogram: AnyValue requires one arm")
		}
		for arm, v := range value {
			switch arm {
			case "stringValue":
				s, ok := v.(string)
				if !ok || !exportText(s) {
					return nil, errors.New("otlp histogram: invalid string attribute")
				}
			case "boolValue":
				if _, ok := v.(bool); !ok {
					return nil, errors.New("otlp histogram: bool attribute required")
				}
			case "intValue":
				if _, e := histogramInt(v); e != nil {
					return nil, e
				}
			default:
				return nil, errors.New("otlp histogram: unsupported AnyValue arm")
			}
			attrs[key] = histogramAttribute{arm, v}
		}
	}
	return attrs, nil
}

func histogramInt(v any) (int64, error) {
	s, ok := v.(string)
	if !ok {
		return 0, errors.New("otlp histogram: decimal intValue string required")
	}
	n, e := strconv.ParseInt(s, 10, 64)
	if e != nil || strconv.FormatInt(n, 10) != s {
		return 0, errors.New("otlp histogram: noncanonical signed integer")
	}
	return n, nil
}
func histogramAttr(attrs map[string]histogramAttribute, key, arm string, required bool) (any, error) {
	a, ok := attrs["nvtelemetry."+key]
	if !ok {
		if required {
			return nil, fmt.Errorf("otlp histogram: missing attribute %q", key)
		}
		return nil, nil
	}
	if a.arm != arm {
		return nil, fmt.Errorf("otlp histogram: wrong attribute arm %q", key)
	}
	return a.value, nil
}

func applyHistogramAttributes(r *Reading, attrs map[string]histogramAttribute) error {
	o := r.Observation
	stringsToSet := []struct {
		key      string
		dst      *string
		required bool
	}{
		{"device.uuid", &r.DeviceUUID, true}, {"evidence", &r.Evidence, true}, {"backend", &r.Backend, false},
		{"observation.family", &o.Family, false}, {"observation.source_class", &o.SourceClass, true},
		{"observation.producer", &o.Producer, false}, {"observation.producer_version", &o.ProducerVersion, false},
		{"observation.producer_schema", &o.ProducerSchema, false}, {"observation.raw_sha256", &o.RawSHA256, false},
		{"observation.transform", &o.Transform, false}, {"observation.transform_sha256", &o.TransformSHA256, false},
		{"observation.clock_epoch", &o.ClockEpoch, true}, {"observation.kind", &o.Kind, true},
		{"sensor_clock_domain", &r.SourceTimestampDomain, false},
	}
	for _, entry := range stringsToSet {
		v, e := histogramAttr(attrs, entry.key, "stringValue", entry.required)
		if e != nil {
			return e
		}
		if v != nil {
			*entry.dst = v.(string)
		}
	}
	clock := &OTLPClockMetadata{}
	o.OTLPClock = clock
	v, e := histogramAttr(attrs, "clock_domain", "stringValue", false)
	if e != nil {
		return e
	}
	if v != nil {
		clock.ClockDomain = v.(string)
	}
	for _, entry := range []struct {
		key string
		dst *bool
	}{{"calibration_known", &clock.HasClockCalibration}, {"sensor_timestamp_known", &r.HasSourceTimestamp}, {"wall_uncertainty_known", &r.HasWallClockUncertainty}} {
		v, e := histogramAttr(attrs, entry.key, "boolValue", true)
		if e != nil {
			return e
		}
		*entry.dst = v.(bool)
	}
	for _, entry := range []struct {
		key   string
		dst   *int64
		known bool
	}{{"sensor_timestamp", &r.SourceTimestamp, r.HasSourceTimestamp}, {"wall_uncertainty_nanos", &r.WallClockUncertaintyNanos, r.HasWallClockUncertainty}, {"calibration_bounds_nanos", &clock.ClockCalibrationBoundsNanos, clock.HasClockCalibration}} {
		v, e := histogramAttr(attrs, entry.key, "intValue", entry.known)
		if e != nil {
			return e
		}
		if v != nil {
			*entry.dst, e = histogramInt(v)
			if e != nil {
				return e
			}
		}
	}
	// Every source-time member is required together, including false/zero fields.
	count := 0
	for k := range attrs {
		if strings.HasPrefix(k, "nvtelemetry.source_time.") {
			count++
		}
	}
	if count != 0 {
		s := &OTLPSourceTime{}
		value := reflect.ValueOf(s).Elem()
		typ := value.Type()
		if count != typ.NumField() {
			return errors.New("otlp histogram: incomplete source-time family")
		}
		for i := 0; i < value.NumField(); i++ {
			key := "source_time." + typ.Field(i).Tag.Get("json")
			field := value.Field(i)
			arm := "stringValue"
			if field.Kind() == reflect.Bool {
				arm = "boolValue"
			}
			v, e := histogramAttr(attrs, key, arm, true)
			if e != nil {
				return e
			}
			switch field.Kind() {
			case reflect.String:
				field.SetString(v.(string))
			case reflect.Bool:
				field.SetBool(v.(bool))
			case reflect.Int64:
				n, e := histogramInt(v)
				if e != nil {
					return e
				}
				field.SetInt(n)
			}
		}
		o.OTLPSourceTime = s
	}
	return nil
}

func histogramAttributes(r Reading) (map[string]histogramAttribute, error) {
	attrs := map[string]histogramAttribute{}
	add := func(k, arm string, v any) { attrs["nvtelemetry."+k] = histogramAttribute{arm, v} }
	text := func(k, s string) {
		if s != "" {
			add(k, "stringValue", s)
		}
	}
	integer := func(k string, n int64) { add(k, "intValue", strconv.FormatInt(n, 10)) }
	o := r.Observation
	c := o.OTLPClock
	add("device.uuid", "stringValue", r.DeviceUUID)
	add("evidence", "stringValue", r.Evidence)
	add("export_schema", "stringValue", "metrics.v1")
	add("backend_status", "stringValue", "unknown")
	if r.Backend != "" {
		add("backend_status", "stringValue", "reported")
		text("backend", r.Backend)
	}
	add("source_status", "stringValue", "unknown")
	add("native_origin_status", "stringValue", "unknown")
	add("clock_domain_known", "boolValue", c.ClockDomain != "")
	text("clock_domain", c.ClockDomain)
	add("sensor_timestamp_known", "boolValue", r.HasSourceTimestamp)
	if r.HasSourceTimestamp {
		integer("sensor_timestamp", r.SourceTimestamp)
		text("sensor_clock_domain", r.SourceTimestampDomain)
	}
	add("wall_uncertainty_known", "boolValue", r.HasWallClockUncertainty)
	if r.HasWallClockUncertainty {
		integer("wall_uncertainty_nanos", r.WallClockUncertaintyNanos)
	}
	add("calibration_known", "boolValue", c.HasClockCalibration)
	if c.HasClockCalibration {
		integer("calibration_bounds_nanos", c.ClockCalibrationBoundsNanos)
	}
	for _, entry := range []struct{ k, v string }{{"family", o.Family}, {"source_class", o.SourceClass}, {"producer", o.Producer}, {"producer_version", o.ProducerVersion}, {"producer_schema", o.ProducerSchema}, {"raw_sha256", o.RawSHA256}, {"transform", o.Transform}, {"transform_sha256", o.TransformSHA256}, {"clock_epoch", o.ClockEpoch}, {"kind", o.Kind}} {
		text("observation."+entry.k, entry.v)
	}
	if o.Interval != nil {
		text("interval_clock", o.Interval.Clock)
	}
	if o.OTLPSourceTime != nil {
		value := reflect.ValueOf(*o.OTLPSourceTime)
		typ := value.Type()
		for i := 0; i < value.NumField(); i++ {
			k := "source_time." + typ.Field(i).Tag.Get("json")
			v := value.Field(i)
			switch v.Kind() {
			case reflect.String:
				add(k, "stringValue", v.String())
			case reflect.Bool:
				add(k, "boolValue", v.Bool())
			case reflect.Int64:
				add(k, "stringValue", strconv.FormatInt(v.Int(), 10))
			}
		}
	}
	if len(attrs) > 32 {
		return nil, errors.New("otlp histogram: attribute limit")
	}
	return attrs, nil
}

func validHistogramReading(r Reading) error {
	o := r.Observation
	if o == nil || o.Histogram == nil || o.Kind != "histogram" || o.SourceClass != "imported" || o.ClockEpoch != "unix" {
		return errors.New("otlp histogram: imported projection required")
	}
	if err := validHistogramV5(o.Histogram); err != nil {
		return err
	}
	if err := validOTLPObservation(r); err != nil {
		return err
	}
	if r.Evidence != EvidenceMeasured && r.Evidence != EvidenceDerived && !noRecordedHistogram(r) {
		return errors.New("otlp histogram: invalid evidence")
	}
	if r.Evidence != EvidenceUnavailable && r.UnavailableReason != "" {
		return errors.New("otlp histogram: available value has unavailable reason")
	}
	if r.DeviceUUID == "" {
		return errors.New("otlp histogram: missing device UUID")
	}
	for _, s := range []string{r.Name, r.Unit, r.DeviceUUID, r.Backend, r.SourceTimestampDomain, o.Family, o.SourceClass, o.Producer, o.ProducerVersion, o.ProducerSchema, o.RawSHA256, o.Transform, o.TransformSHA256, o.ClockEpoch, o.Kind} {
		if !exportText(s) {
			return errors.New("otlp histogram: string limit or encoding")
		}
	}
	for _, hash := range []string{o.RawSHA256, o.TransformSHA256} {
		if hash != "" && !streamHash(hash) {
			return errors.New("otlp histogram: invalid digest assertion")
		}
	}
	if !r.HasSourceTimestamp && (r.SourceTimestamp != 0 || r.SourceTimestampDomain != "") || r.HasSourceTimestamp && r.SourceTimestampDomain == "" || r.WallClockUncertaintyNanos < 0 || !r.HasWallClockUncertainty && r.WallClockUncertaintyNanos != 0 {
		return errors.New("otlp histogram: inconsistent timestamp knownness")
	}
	observer := o.Observer
	coverage := observer.SamplingCoverage
	if coverage == nil || !reflect.DeepEqual(*coverage, SamplingCoverage{Status: "unknown"}) {
		return errors.New("otlp histogram: unsupported coverage")
	}
	observer.SamplingCoverage = nil
	if !reflect.ValueOf(observer).IsZero() {
		return errors.New("otlp histogram: unsupported observer identity")
	}
	// Zero only mapped fields. Any current or future unmapped nonzero field refuses.
	extra := r
	extra.Name = ""
	extra.Unit = ""
	extra.DeviceUUID = ""
	extra.Backend = ""
	extra.Evidence = ""
	extra.UnavailableReason = ""
	extra.TimestampUnix = 0
	extra.SourceTimestamp = 0
	extra.HasSourceTimestamp = false
	extra.SourceTimestampDomain = ""
	extra.WallClockUncertaintyNanos = 0
	extra.HasWallClockUncertainty = false
	extra.Observation = nil
	if !reflect.ValueOf(extra).IsZero() {
		return errors.New("otlp histogram: unsupported reading field")
	}
	other := *o
	other.Family = ""
	other.SourceClass = ""
	other.Producer = ""
	other.ProducerVersion = ""
	other.ProducerSchema = ""
	other.RawSHA256 = ""
	other.Transform = ""
	other.TransformSHA256 = ""
	other.ClockEpoch = ""
	other.Kind = ""
	other.Histogram = nil
	other.Interval = nil
	other.Observer = Observer{}
	other.OTLPScopeName = nil
	other.OTLPClock = nil
	other.OTLPSourceTime = nil
	if !reflect.ValueOf(other).IsZero() {
		return errors.New("otlp histogram: unsupported observation field")
	}
	if o.Interval != nil && (o.Interval.Start <= 0 || o.Interval.End != r.TimestampUnix || o.Interval.Clock != "unix") {
		return errors.New("otlp histogram: invalid interval")
	}
	return nil
}

func histogramPoint(r Reading, boundsMax int) (map[string]any, error) {
	if err := validHistogramReading(r); err != nil {
		return nil, err
	}
	h := r.Observation.Histogram
	explicit := len(h.Bounds)
	if explicit > 0 {
		explicit--
	}
	if explicit > boundsMax {
		return nil, errors.New("otlp histogram: explicit bound limit")
	}
	attrs, err := histogramAttributes(r)
	if err != nil {
		return nil, err
	}
	// Fixed ordering makes output deterministic without claiming input attribute order.
	keys := make([]string, 0, len(attrs))
	for k := range attrs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	wireAttrs := []any{}
	for _, k := range keys {
		v := attrs[k]
		wireAttrs = append(wireAttrs, map[string]any{"key": k, "value": map[string]any{v.arm: v.value}})
	}
	point := map[string]any{"timeUnixNano": strconv.FormatInt(r.TimestampUnix, 10), "count": strconv.FormatUint(*h.Count, 10), "attributes": wireAttrs, "bucketCounts": []string{}, "explicitBounds": []float64{}}
	if r.Observation.Interval != nil {
		point["startTimeUnixNano"] = strconv.FormatInt(r.Observation.Interval.Start, 10)
	}
	if h.Flags != 0 {
		point["flags"] = h.Flags
	}
	for _, entry := range []struct {
		k string
		v *float64
	}{{"sum", h.Sum}, {"min", h.Min}, {"max", h.Max}} {
		if entry.v != nil {
			point[entry.k] = *entry.v
		}
	}
	if len(h.Bounds) > 0 {
		bounds := make([]float64, explicit)
		for i := range bounds {
			bounds[i], _ = strconv.ParseFloat(h.Bounds[i], 64)
		}
		counts := make([]string, len(h.Counts))
		previous := uint64(0)
		for i, n := range h.Counts {
			if n < previous {
				return nil, errors.New("otlp histogram: count subtraction underflow")
			}
			counts[i] = strconv.FormatUint(n-previous, 10)
			previous = n
		}
		point["explicitBounds"] = bounds
		point["bucketCounts"] = counts
	}
	return point, nil
}

func resolvedHistogramClock(r Reading, o ExportOptions) (OTLPClockMetadata, error) {
	options := OTLPClockMetadata{o.ClockDomain, o.HasClockCalibration, o.ClockCalibrationBoundsNanos}
	if !validOTLPClock(options) {
		return OTLPClockMetadata{}, errors.New("otlp histogram: invalid clock options")
	}
	if r.Observation == nil || r.Observation.OTLPClock == nil {
		if r.Observation != nil && r.Observation.SourceClass == "imported" && r.Observation.Histogram != nil {
			return OTLPClockMetadata{}, errors.New("otlp histogram: missing imported clock")
		}
		return options, nil
	}
	c := *r.Observation.OTLPClock
	if !validOTLPClock(c) || options != (OTLPClockMetadata{}) && options != c {
		return OTLPClockMetadata{}, errors.New("otlp histogram: conflicting clock options")
	}
	return c, nil
}

func writeHistogramMetrics(w io.Writer, readings []Reading, o ExportOptions) error {
	bindings, err := histogramBindings(o.HistogramBindings)
	if err != nil {
		return err
	}
	maxReadings, maxBytes, err := exportLimits(o)
	if err != nil {
		return err
	}
	if len(readings) > maxReadings {
		return errors.New("otlp histogram: reading limit")
	}
	maxMetrics, err := histogramLimit(o.MaxMetrics, 4096, 65536)
	if err != nil {
		return err
	}
	maxBounds, err := histogramLimit(o.MaxExplicitBounds, 256, 4096)
	if err != nil {
		return err
	}
	metrics := []any{}
	groups := map[string]map[string]any{}
	seen := map[[2]string]bool{}
	var batchClock *OTLPClockMetadata
	batchScope := otlpScopeName
	scopeSet := false
	for _, r := range readings {
		isHistogram := r.Observation != nil && r.Observation.Histogram != nil
		if !isHistogram && r.Observation != nil && (r.Observation.OTLPScopeName != nil || r.Observation.OTLPClock != nil || r.Observation.OTLPSourceTime != nil) {
			return errors.New("otlp histogram: metadata on scalar reading")
		}

		scope := otlpScopeName
		if isHistogram && r.Observation.SourceClass == "imported" {
			if r.Observation.OTLPScopeName == nil {
				return errors.New("otlp histogram: missing imported scope")
			}
			scope = *r.Observation.OTLPScopeName
			if scope != "" && scope != otlpScopeName {
				return errors.New("otlp histogram: invalid scope")
			}
		}
		if scopeSet && scope != batchScope {
			return errors.New("otlp histogram: mixed scope provenance")
		}
		scopeSet = true
		batchScope = scope
		clock, e := resolvedHistogramClock(r, o)
		if e != nil {
			return e
		}
		if batchClock != nil && *batchClock != clock {
			return errors.New("otlp histogram: mixed clock provenance")
		}
		clockCopy := clock
		batchClock = &clockCopy
		key := [2]string{r.Name, r.DeviceUUID}
		if seen[key] {
			return errors.New("otlp histogram: duplicate series")
		}
		seen[key] = true
		if isHistogram {
			unit, ok := bindings[r.Name]
			if !ok || unit != r.Unit {
				return errors.New("otlp histogram: unbound histogram")
			}
			projected, e := projectHistogram(r, o, clock)
			if e != nil {
				return e
			}
			point, e := histogramPoint(projected, maxBounds)
			if e != nil {
				return e
			}
			h := projected.Observation.Histogram
			temp := "AGGREGATION_TEMPORALITY_DELTA"
			if h.Temporality == "cumulative" {
				temp = "AGGREGATION_TEMPORALITY_CUMULATIVE"
			}
			if group, ok := groups[r.Name]; ok {
				histogram := group["histogram"].(map[string]any)
				if histogram["aggregationTemporality"] != temp {
					return errors.New("otlp histogram: conflicting temporality")
				}
				histogram["dataPoints"] = append(histogram["dataPoints"].([]any), point)
			} else {
				metric := map[string]any{"name": "nvtelemetry." + r.Name, "unit": histogramUnits[unit], "histogram": map[string]any{"dataPoints": []any{point}, "aggregationTemporality": temp}}
				groups[r.Name] = metric
				metrics = append(metrics, metric)
			}
		} else {
			var scalar bytes.Buffer
			options := o
			options.HistogramBindings = nil
			options.UnixTimeMapping = nil
			options.MaxMetrics = 0
			options.MaxExplicitBounds = 0
			options.ClockDomain = clock.ClockDomain
			options.HasClockCalibration = clock.HasClockCalibration
			options.ClockCalibrationBoundsNanos = clock.ClockCalibrationBoundsNanos
			if e := writeScalarOTLPMetrics(&scalar, []Reading{r}, options); e != nil {
				return e
			}
			var wire struct {
				ResourceMetrics []struct{ ScopeMetrics []struct{ Metrics []any } }
			}
			if e := json.Unmarshal(scalar.Bytes(), &wire); e != nil {
				return e
			}
			metrics = append(metrics, wire.ResourceMetrics[0].ScopeMetrics[0].Metrics...)
		}
		if len(metrics) > maxMetrics {
			return errors.New("otlp histogram: metric limit")
		}
	}
	scope := map[string]string{}
	if batchScope != "" {
		scope["name"] = batchScope
	}
	data, err := json.Marshal(map[string]any{"resourceMetrics": []any{map[string]any{"scopeMetrics": []any{map[string]any{"scope": scope, "metrics": metrics}}}}})
	if err != nil {
		return err
	}
	return exportWrite(w, data, maxBytes)
}

// UnixTimeMapping asserts a 1:1 nanosecond transform within an inclusive source
// range. The zero value is invalid. These facts are supplied by the caller and
// establish no independently verified calibration or source authenticity.
type UnixTimeMapping struct {
	SourceClockDomain     string
	SourceClockEpoch      string
	SourceValidFromNanos  int64
	SourceValidUntilNanos int64
	UnixOffsetNanos       int64
	Provenance            string
}

func projectHistogram(r Reading, o ExportOptions, c OTLPClockMetadata) (Reading, error) {
	r = cloneReadings([]Reading{r})[0]
	observation := r.Observation
	if observation.SourceClass == "imported" {
		if o.UnixTimeMapping != nil {
			return Reading{}, errors.New("otlp histogram: mapping on imported reading")
		}
		if observation.OTLPClock == nil {
			return Reading{}, errors.New("otlp histogram: missing imported clock")
		}
		return r, nil
	}
	mapping := o.UnixTimeMapping
	if mapping == nil || r.TimestampMonotonic == 0 {
		return Reading{}, errors.New("otlp histogram: local time mapping required")
	}
	switch observation.SourceClass {
	case "native", "managed", "synthetic", "modeled":
	default:
		return Reading{}, errors.New("otlp histogram: invalid local source class")
	}
	if observation.OTLPSourceTime != nil || observation.OTLPScopeName != nil {
		return Reading{}, errors.New("otlp histogram: OTLP import provenance on local reading")
	}
	for _, v := range []string{mapping.SourceClockDomain, mapping.SourceClockEpoch, mapping.Provenance} {
		if v == "" || !exportText(v) {
			return Reading{}, errors.New("otlp histogram: invalid mapping assertion")
		}
	}
	if observation.ClockEpoch != mapping.SourceClockEpoch || mapping.SourceValidFromNanos > mapping.SourceValidUntilNanos || r.TimestampMonotonic < mapping.SourceValidFromNanos || r.TimestampMonotonic > mapping.SourceValidUntilNanos {
		return Reading{}, errors.New("otlp histogram: source outside mapping")
	}
	end, err := checkedUnixTime(r.TimestampMonotonic, mapping.UnixOffsetNanos)
	if err != nil {
		return Reading{}, err
	}
	if r.TimestampUnix != 0 && r.TimestampUnix != end {
		return Reading{}, errors.New("otlp histogram: Unix timestamp mismatch")
	}
	source := &OTLPSourceTime{SourceClass: observation.SourceClass, SourceClockEpoch: observation.ClockEpoch, SourceMonotonicEndNanos: r.TimestampMonotonic, MappingDomain: mapping.SourceClockDomain, MappingEpoch: mapping.SourceClockEpoch, MappingValidFromNanos: mapping.SourceValidFromNanos, MappingValidUntilNanos: mapping.SourceValidUntilNanos, MappingUnixOffsetNanos: mapping.UnixOffsetNanos, MappingProvenance: mapping.Provenance}
	if interval := observation.Interval; interval != nil {
		if interval.Clock != mapping.SourceClockDomain || interval.Start <= 0 || interval.End != r.TimestampMonotonic || interval.Start > interval.End || interval.Start < mapping.SourceValidFromNanos || interval.Start > mapping.SourceValidUntilNanos {
			return Reading{}, errors.New("otlp histogram: source interval mismatch")
		}
		start, err := checkedUnixTime(interval.Start, mapping.UnixOffsetNanos)
		if err != nil || start > end {
			return Reading{}, errors.New("otlp histogram: invalid mapped interval")
		}
		source.SourceIntervalPresent = true
		source.SourceIntervalClock = interval.Clock
		source.SourceIntervalStartNanos = interval.Start
		source.SourceIntervalEndNanos = interval.End
		observation.Interval = &tracecontract.Interval{Clock: "unix", Start: start, End: end}
	}
	scope := otlpScopeName
	observation.OTLPScopeName = &scope
	observation.OTLPClock = &c
	observation.OTLPSourceTime = source
	observation.SourceClass = "imported"
	observation.ClockEpoch = "unix"
	r.TimestampUnix = end
	r.TimestampMonotonic = 0
	return r, nil
}
