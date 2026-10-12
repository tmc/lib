package nvtelemetry_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"

	n "github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func otlpV5Options() n.ExportOptions {
	return n.ExportOptions{HistogramBindings: []n.HistogramBinding{{Name: "latency", Unit: n.UnitSeconds}}}
}
func otlpV5ReadOptions() n.OTLPReadOptions {
	return n.OTLPReadOptions{HistogramBindings: otlpV5Options().HistogramBindings}
}
func otlpV5Write(t *testing.T, readings []n.Reading, o n.ExportOptions) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := n.WriteOTLPMetrics(&b, readings, o); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}
func otlpV5Read(t *testing.T, raw []byte) n.OTLPHistogramReadResult {
	t.Helper()
	result, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), otlpV5ReadOptions())
	if err != nil {
		t.Fatal(err)
	}
	return result
}
func otlpV5Wire(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	return m
}
func otlpV5Scope(m map[string]any) map[string]any {
	return m["resourceMetrics"].([]any)[0].(map[string]any)["scopeMetrics"].([]any)[0].(map[string]any)
}
func otlpV5Metric(m map[string]any) map[string]any {
	return otlpV5Scope(m)["metrics"].([]any)[0].(map[string]any)
}
func otlpV5Point(m map[string]any) map[string]any {
	return otlpV5Metric(m)["histogram"].(map[string]any)["dataPoints"].([]any)[0].(map[string]any)
}
func otlpV5Attr(p map[string]any, key string) map[string]any {
	for _, a := range p["attributes"].([]any) {
		m := a.(map[string]any)
		if m["key"] == "nvtelemetry."+key {
			return m["value"].(map[string]any)
		}
	}
	return nil
}
func otlpV5JSON(t *testing.T, m any) []byte {
	t.Helper()
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestOTLPHistogramImportedRoundTrip(t *testing.T) {
	for _, scope := range []string{"github.com/tmc/lib/nvidia/x/nvtelemetry", ""} {
		for _, temp := range []string{"delta", "cumulative"} {
			for _, shape := range []string{"distributed", "bucketless", "terminal", "zero", "no-recorded"} {
				t.Run(scope+"/"+temp+"/"+shape, func(t *testing.T) {
					r := v5Reading()
					o := r.Observation
					h := o.Histogram
					h.Temporality = temp
					o.OTLPScopeName = v5String(scope)
					switch shape {
					case "bucketless":
						h.Bounds, h.Counts = []string{}, []uint64{}
					case "terminal":
						h.Bounds, h.Counts = []string{"+Inf"}, []uint64{3}
					case "zero", "no-recorded":
						h.Bounds, h.Counts = []string{}, []uint64{}
						h.Count = v5Uint(0)
						h.Sum, h.Min, h.Max = nil, nil, nil
						if shape == "no-recorded" {
							h.Flags = 1
							r.Evidence = n.EvidenceUnavailable
							r.UnavailableReason = "no_recorded_value"
						}
					}
					want, _ := json.Marshal(r)
					raw := otlpV5Write(t, []n.Reading{r}, otlpV5Options())
					result := otlpV5Read(t, raw)
					got, _ := json.Marshal(result.Readings[0])
					if !bytes.Equal(want, got) {
						t.Fatalf("field loss: %s != %s", got, want)
					}
					stream, err := v5Stream(t, 5, result.Readings[0])
					if err != nil {
						t.Fatal(err)
					}
					_, _, samples, err := n.ReadStream(bytes.NewReader(stream), 0)
					if err != nil {
						t.Fatal(err)
					}
					got, _ = json.Marshal(samples[0].Readings[0])
					if !bytes.Equal(want, got) {
						t.Fatal("stream field loss")
					}
					again := otlpV5Write(t, samples[0].Readings, otlpV5Options())
					if !bytes.Equal(raw, again) {
						t.Fatalf("re-export changed bytes:\n%s\n%s", raw, again)
					}
					digest := sha256.Sum256(raw)
					if result.TransportSHA256 != hex.EncodeToString(digest[:]) || result.Readings[0].Observation.RawSHA256 != "" {
						t.Fatal("transport/lineage confusion")
					}
				})
			}
		}
	}
}

func TestOTLPHistogramDefaultsAndNumbers(t *testing.T) {
	base := otlpV5Write(t, []n.Reading{v5Reading()}, otlpV5Options())
	tests := []struct {
		name   string
		mutate func(map[string]any)
		good   bool
		check  func(n.Reading)
	}{
		{"equal-counts", func(p map[string]any) { p["bucketCounts"] = []any{"0", "3"} }, true, func(r n.Reading) {
			if !reflect.DeepEqual(r.Observation.Histogram.Counts, []uint64{0, 3}) {
				t.Fatal("count conversion")
			}
		}},
		{"count-token", func(p map[string]any) { p["count"] = 3 }, true, nil},
		{"missing-default-zero", func(p map[string]any) {
			delete(p, "count")
			delete(p, "sum")
			delete(p, "min")
			delete(p, "max")
			p["bucketCounts"] = []any{"0", "0"}
		}, true, func(r n.Reading) {
			if r.Observation.Histogram.Count == nil || *r.Observation.Histogram.Count != 0 {
				t.Fatal("missing count was not known zero")
			}
		}},
		{"sum-zero", func(p map[string]any) { p["sum"] = 0 }, true, func(r n.Reading) {
			if r.Observation.Histogram.Sum == nil || *r.Observation.Histogram.Sum != 0 {
				t.Fatal("summary zero absent")
			}
		}},
		{"sum-absent", func(p map[string]any) { delete(p, "sum") }, true, func(r n.Reading) {
			if r.Observation.Histogram.Sum != nil {
				t.Fatal("invented summary")
			}
		}},
		{"signed-zero", func(p map[string]any) { p["explicitBounds"] = []any{json.Number("-0")} }, true, func(r n.Reading) {
			if r.Observation.Histogram.Bounds[0] != "0" {
				t.Fatal("signed zero not normalized")
			}
		}},
		{"subnormal", func(p map[string]any) { p["explicitBounds"] = []any{json.Number("5e-324")} }, true, nil},
		{"underflow", func(p map[string]any) { p["explicitBounds"] = []any{json.Number("1e-500")} }, false, nil},
		{"overflow", func(p map[string]any) { p["explicitBounds"] = []any{json.Number("1e500")} }, false, nil},
		{"collapse", func(p map[string]any) {
			p["explicitBounds"] = []any{json.Number("9007199254740992"), json.Number("9007199254740993")}
			p["bucketCounts"] = []any{"1", "1", "1"}
		}, false, nil},
		{"unordered", func(p map[string]any) { p["explicitBounds"] = []any{2, 1}; p["bucketCounts"] = []any{"1", "1", "1"} }, false, nil},
		{"count-overflow", func(p map[string]any) { p["count"] = "18446744073709551616" }, false, nil},
		{"bucket-overflow", func(p map[string]any) {
			p["count"] = "18446744073709551615"
			p["bucketCounts"] = []any{"18446744073709551615", "1"}
		}, false, nil},
		{"max-count", func(p map[string]any) {
			p["count"] = "18446744073709551615"
			p["bucketCounts"] = []any{"18446744073709551615", "0"}
		}, true, nil},
		{"count-exp", func(p map[string]any) { p["count"] = json.Number("3e0") }, false, nil},
		{"count-leading-zero", func(p map[string]any) { p["count"] = "03" }, false, nil},
		{"count-negative", func(p map[string]any) { p["count"] = "-1" }, false, nil},
		{"count-fraction", func(p map[string]any) { p["count"] = json.Number("3.0") }, false, nil},
		{"future-flags", func(p map[string]any) { p["flags"] = 2 }, false, nil},
		{"flags-string", func(p map[string]any) { p["flags"] = "0" }, false, nil},
		{"flags-overflow", func(p map[string]any) { p["flags"] = json.Number("4294967296") }, false, nil},
		{"invalid-recorded-flag", func(p map[string]any) { p["flags"] = 1 }, false, nil},
		{"exemplars", func(p map[string]any) { p["exemplars"] = []any{map[string]any{}} }, false, nil},
		{"empty-exemplars", func(p map[string]any) { p["exemplars"] = []any{} }, true, nil},
		{"end-zero", func(p map[string]any) { p["timeUnixNano"] = "0" }, false, nil},
		{"end-overflow", func(p map[string]any) { p["timeUnixNano"] = "9223372036854775808" }, false, nil},
		{"end-exp", func(p map[string]any) { p["timeUnixNano"] = json.Number("2e1") }, false, nil},
		{"start-zero", func(p map[string]any) { p["startTimeUnixNano"] = "0" }, false, nil},
		{"start-reversed", func(p map[string]any) { p["startTimeUnixNano"] = "21" }, false, nil},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			wire := otlpV5Wire(t, base)
			test.mutate(otlpV5Point(wire))
			raw := otlpV5JSON(t, wire)
			result, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), otlpV5ReadOptions())
			if test.good {
				if err != nil {
					t.Fatal(err)
				}
				if test.check != nil {
					test.check(result.Readings[0])
				}
			} else {
				if err == nil || len(result.Readings) != 0 || result.TransportSHA256 != "" {
					t.Fatal("invalid input returned successful or partial data")
				}
			}
		})
	}
}

type otlpCountingWriter struct {
	calls int
	b     bytes.Buffer
}

func (w *otlpCountingWriter) Write(p []byte) (int, error) { w.calls++; return w.b.Write(p) }

func TestOTLPHistogramBindingAndBatchRefusal(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*n.ExportOptions, *[]n.Reading)
	}{
		{"unbound", func(o *n.ExportOptions, r *[]n.Reading) { o.HistogramBindings = nil }},
		{"unit", func(o *n.ExportOptions, r *[]n.Reading) { o.HistogramBindings[0].Unit = n.UnitBytes }},
		{"bad-unit", func(o *n.ExportOptions, r *[]n.Reading) { o.HistogramBindings[0].Unit = "kiloseconds" }},
		{"collision", func(o *n.ExportOptions, r *[]n.Reading) { o.HistogramBindings[0].Name = n.MetricGPUUtilization }},
		{"duplicate-binding", func(o *n.ExportOptions, r *[]n.Reading) {
			o.HistogramBindings = append(o.HistogramBindings, o.HistogramBindings[0])
		}},
		{"duplicate-series", func(o *n.ExportOptions, r *[]n.Reading) { *r = append(*r, (*r)[0]) }},
		{"mixed-scope", func(o *n.ExportOptions, r *[]n.Reading) {
			x := v5Reading()
			x.DeviceUUID = "GPU-other"
			x.Observation.OTLPScopeName = v5String("")
			*r = append(*r, x)
		}},
		{"mixed-clock", func(o *n.ExportOptions, r *[]n.Reading) {
			x := v5Reading()
			x.DeviceUUID = "GPU-other"
			x.Observation.OTLPClock.ClockDomain = "clock"
			*r = append(*r, x)
		}},
		{"mixed-temporality", func(o *n.ExportOptions, r *[]n.Reading) {
			x := v5Reading()
			x.DeviceUUID = "GPU-other"
			x.Observation.Histogram.Temporality = "cumulative"
			*r = append(*r, x)
		}},
		{"nil-imported-clock", func(o *n.ExportOptions, r *[]n.Reading) {
			(*r)[0].Observation.OTLPClock = nil
			o.ClockDomain = "cannot-repair"
		}},
		{"unsupported-reason", func(o *n.ExportOptions, r *[]n.Reading) {
			(*r)[0].Evidence = n.EvidenceUnavailable
			(*r)[0].UnavailableReason = "unsupported"
		}},
	}
	for _, name := range []string{"", "Latency", "nvtelemetry.latency", "latency.x", "/latency", " latency", "latency ", strings.Repeat("a", 129)} {
		bad := name
		tests = append(tests, struct {
			name   string
			mutate func(*n.ExportOptions, *[]n.Reading)
		}{"name/" + bad, func(o *n.ExportOptions, r *[]n.Reading) { o.HistogramBindings[0].Name = bad }})
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			o := otlpV5Options()
			r := []n.Reading{v5Reading()}
			test.mutate(&o, &r)
			var w otlpCountingWriter
			if err := n.WriteOTLPMetrics(&w, r, o); err == nil || w.calls != 0 {
				t.Fatalf("batch refusal after writes: %v %d", err, w.calls)
			}
		})
	}
	a := v5Reading()
	b := v5Reading()
	b.DeviceUUID = "GPU-other"
	b.Observation.Histogram.Counts = []uint64{1, 3}
	wire := otlpV5Wire(t, otlpV5Write(t, []n.Reading{a, b}, otlpV5Options()))
	metrics := otlpV5Scope(wire)["metrics"].([]any)
	if len(metrics) != 1 {
		t.Fatal("same binding was not grouped")
	}
	points := otlpV5Metric(wire)["histogram"].(map[string]any)["dataPoints"].([]any)
	if len(points) != 2 {
		t.Fatal("lost grouped points")
	}
	result := otlpV5Read(t, otlpV5JSON(t, wire))
	if result.Readings[0].DeviceUUID != a.DeviceUUID || result.Readings[1].DeviceUUID != b.DeviceUUID {
		t.Fatal("point order changed")
	}
	// Repeated input Metric entries preserve encounter order and must agree on temporality.
	m := otlpV5Metric(wire)
	second := otlpV5Wire(t, otlpV5Write(t, []n.Reading{b}, otlpV5Options()))
	m["histogram"].(map[string]any)["dataPoints"] = points[:1]
	otlpV5Scope(wire)["metrics"] = []any{m, otlpV5Metric(second)}
	result = otlpV5Read(t, otlpV5JSON(t, wire))
	if len(result.Readings) != 2 || result.Readings[1].DeviceUUID != b.DeviceUUID {
		t.Fatal("repeated metric order changed")
	}
	otlpV5Metric(second)["histogram"].(map[string]any)["aggregationTemporality"] = "AGGREGATION_TEMPORALITY_CUMULATIVE"
	if _, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(otlpV5JSON(t, wire)), otlpV5ReadOptions()); err == nil {
		t.Fatal("repeated metric picked a temporality")
	}
}

func TestOTLPHistogramStrictEnvelope(t *testing.T) {
	base := otlpV5Write(t, []n.Reading{v5Reading()}, otlpV5Options())
	tests := []struct {
		name   string
		mutate func(map[string]any)
	}{
		{"top-extra", func(m map[string]any) { m["extra"] = 0 }},
		{"resource-extra", func(m map[string]any) { m["resourceMetrics"].([]any)[0].(map[string]any)["extra"] = 0 }},
		{"resource-attrs", func(m map[string]any) {
			m["resourceMetrics"].([]any)[0].(map[string]any)["resource"] = map[string]any{"attributes": []any{map[string]any{}}}
		}},
		{"resource-drop", func(m map[string]any) {
			m["resourceMetrics"].([]any)[0].(map[string]any)["resource"] = map[string]any{"droppedAttributesCount": 1}
		}},
		{"resource-entity", func(m map[string]any) {
			m["resourceMetrics"].([]any)[0].(map[string]any)["resource"] = map[string]any{"entityRefs": []any{}}
		}},
		{"resource-schema", func(m map[string]any) { m["resourceMetrics"].([]any)[0].(map[string]any)["schemaUrl"] = "x" }},
		{"scope-extra", func(m map[string]any) { otlpV5Scope(m)["extra"] = 0 }},
		{"scope-name", func(m map[string]any) { otlpV5Scope(m)["scope"] = map[string]any{"name": "other"} }},
		{"scope-version", func(m map[string]any) { otlpV5Scope(m)["scope"] = map[string]any{"version": "v1"} }},
		{"scope-drop", func(m map[string]any) { otlpV5Scope(m)["scope"] = map[string]any{"droppedAttributesCount": 1} }},
		{"scope-schema", func(m map[string]any) { otlpV5Scope(m)["schemaUrl"] = "x" }},
		{"description", func(m map[string]any) { otlpV5Metric(m)["description"] = "discarded" }},
		{"metadata", func(m map[string]any) { otlpV5Metric(m)["metadata"] = []any{map[string]any{}} }},
		{"metric-extra", func(m map[string]any) { otlpV5Metric(m)["gauge"] = map[string]any{} }},
		{"histogram-extra", func(m map[string]any) { otlpV5Metric(m)["histogram"].(map[string]any)["extra"] = 0 }},
		{"point-extra", func(m map[string]any) { otlpV5Point(m)["extra"] = 0 }},
		{"attr-extra", func(m map[string]any) {
			p := otlpV5Point(m)
			p["attributes"] = append(p["attributes"].([]any), map[string]any{"key": "unknown", "value": map[string]any{"stringValue": "x"}})
		}},
		{"alias-attr", func(m map[string]any) {
			p := otlpV5Point(m)
			p["attributes"] = append(p["attributes"].([]any), map[string]any{"key": "NVTELEMETRY.EVIDENCE", "value": map[string]any{"stringValue": "measured"}})
		}},
		{"wrong-arm", func(m map[string]any) {
			a := otlpV5Attr(otlpV5Point(m), "sensor_timestamp_known")
			delete(a, "boolValue")
			a["stringValue"] = "false"
		}},
		{"multiple-arm", func(m map[string]any) { otlpV5Attr(otlpV5Point(m), "evidence")["boolValue"] = false }},
		{"source-status", func(m map[string]any) { otlpV5Attr(otlpV5Point(m), "source_status")["stringValue"] = "reported" }},
		{"class", func(m map[string]any) {
			otlpV5Attr(otlpV5Point(m), "observation.source_class")["stringValue"] = "native"
		}},
		{"epoch", func(m map[string]any) { otlpV5Attr(otlpV5Point(m), "observation.clock_epoch")["stringValue"] = "mono" }},
		{"unit", func(m map[string]any) { otlpV5Metric(m)["unit"] = "ms" }},
		{"future-temp", func(m map[string]any) {
			otlpV5Metric(m)["histogram"].(map[string]any)["aggregationTemporality"] = "FUTURE"
		}},
		{"numeric-temp", func(m map[string]any) { otlpV5Metric(m)["histogram"].(map[string]any)["aggregationTemporality"] = 1 }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			m := otlpV5Wire(t, base)
			test.mutate(m)
			result, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(otlpV5JSON(t, m)), otlpV5ReadOptions())
			if err == nil || len(result.Readings) != 0 || result.TransportSHA256 != "" {
				t.Fatal("strict refusal returned partial/successful data")
			}
		})
	}
	for _, raw := range [][]byte{append(append([]byte(nil), base...), []byte("{}")...), bytes.Replace(base, []byte(`"resourceMetrics":`), []byte(`"ResourceMetrics":`), 1), bytes.Replace(base, []byte(`"count":"3"`), []byte(`"count":"3","COUNT":"3"`), 1), []byte("{not-json}")} {
		if _, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), otlpV5ReadOptions()); err == nil {
			t.Fatal("malformed/alias/duplicate envelope accepted")
		}
	}
	// Replace every property, recursively, with null. Each refusal remains whole-call.
	var paths [][]any
	var walk func(any, []any)
	walk = func(v any, path []any) {
		switch x := v.(type) {
		case map[string]any:
			for k, v := range x {
				next := append(append([]any(nil), path...), k)
				paths = append(paths, next)
				walk(v, next)
			}
		case []any:
			for i, v := range x {
				walk(v, append(append([]any(nil), path...), i))
			}
		}
	}
	walk(otlpV5Wire(t, base), nil)
	for _, path := range paths {
		m := otlpV5Wire(t, base)
		var node any = m
		for _, k := range path[:len(path)-1] {
			switch k := k.(type) {
			case string:
				node = node.(map[string]any)[k]
			case int:
				node = node.([]any)[k]
			}
		}
		node.(map[string]any)[path[len(path)-1].(string)] = nil
		result, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(otlpV5JSON(t, m)), otlpV5ReadOptions())
		if err == nil || len(result.Readings) != 0 {
			t.Fatalf("null accepted at %v", path)
		}
	}
}

func TestOTLPHistogramFieldAllowlist(t *testing.T) {
	readingFields := []string{"Number", "Source", "Value", "IntValue", "IsInt", "TimestampMonotonic", "QueryStartNanos", "QueryEndNanos", "HasQueryBounds", "RefreshCadenceNanos", "IntegrationIntervalNanos", "IsWarmup"}
	observationFields := []string{"Assignment", "Workload", "OpID", "Host", "Device", "UncertaintyNS", "CounterEpoch", "CounterReset", "Denominator", "DenominatorUnit", "Scope", "HasReturn", "TID"}
	nonzero := func(v reflect.Value) {
		switch v.Kind() {
		case reflect.Pointer:
			v.Set(reflect.New(v.Type().Elem()))
		case reflect.String:
			v.SetString("unsupported")
		case reflect.Bool:
			v.SetBool(true)
		case reflect.Int, reflect.Int64:
			v.SetInt(1)
		case reflect.Uint, reflect.Uint64:
			v.SetUint(1)
		case reflect.Float64:
			v.SetFloat(1)
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				if v.Field(i).CanSet() {
					switch v.Field(i).Kind() {
					case reflect.String:
						v.Field(i).SetString("unsupported")
						return
					}
				}
			}
		default:
			t.Fatalf("no nonzero probe for %v", v.Type())
		}
	}
	for _, fields := range []struct {
		target string
		names  []string
	}{{"reading", readingFields}, {"observation", observationFields}} {
		for _, name := range fields.names {
			t.Run(fields.target+"/"+name, func(t *testing.T) {
				r := v5Reading()
				value := reflect.ValueOf(&r).Elem()
				if fields.target == "observation" {
					value = reflect.ValueOf(r.Observation).Elem()
				}
				nonzero(value.FieldByName(name))
				var w otlpCountingWriter
				if err := n.WriteOTLPMetrics(&w, []n.Reading{r}, otlpV5Options()); err == nil || w.calls != 0 {
					t.Fatalf("unsupported %s lost before output: %v", name, err)
				}
			})
		}
	}
	for _, name := range []string{"ID", "Version", "Mode", "Pass", "Population", "Filter", "Replay"} {
		t.Run("observer/"+name, func(t *testing.T) {
			r := v5Reading()
			reflect.ValueOf(&r.Observation.Observer).Elem().FieldByName(name).SetString("unsupported")
			var w otlpCountingWriter
			if err := n.WriteOTLPMetrics(&w, []n.Reading{r}, otlpV5Options()); err == nil || w.calls != 0 {
				t.Fatal("observer metadata dropped")
			}
		})
	}
	for _, name := range []string{"Status", "Basis", "Scope", "Numerator", "Denominator", "Unit", "Reason"} {
		t.Run("coverage/"+name, func(t *testing.T) {
			r := v5Reading()
			nonzero(reflect.ValueOf(r.Observation.Observer.SamplingCoverage).Elem().FieldByName(name))
			var w otlpCountingWriter
			if err := n.WriteOTLPMetrics(&w, []n.Reading{r}, otlpV5Options()); err == nil || w.calls != 0 {
				t.Fatal("coverage metadata dropped")
			}
		})
	}
	// Each optional ordinary mapped field survives with its exact primitive arm.
	positive := []struct {
		name   string
		mutate func(*n.Reading)
	}{
		{"backend", func(r *n.Reading) { r.Backend = "reported-backend" }},
		{"sensor-zero", func(r *n.Reading) { r.HasSourceTimestamp = true; r.SourceTimestampDomain = "sensor" }},
		{"sensor-negative", func(r *n.Reading) {
			r.HasSourceTimestamp = true
			r.SourceTimestamp = -2
			r.SourceTimestampDomain = "sensor"
		}},
		{"wall-zero", func(r *n.Reading) { r.HasWallClockUncertainty = true }},
		{"wall-positive", func(r *n.Reading) { r.HasWallClockUncertainty = true; r.WallClockUncertaintyNanos = 2 }},
		{"clock-domain", func(r *n.Reading) { r.Observation.OTLPClock.ClockDomain = "unix" }},
		{"calibration-zero", func(r *n.Reading) {
			r.Observation.OTLPClock = &n.OTLPClockMetadata{ClockDomain: "unix", HasClockCalibration: true}
		}},
		{"calibration-positive", func(r *n.Reading) {
			r.Observation.OTLPClock = &n.OTLPClockMetadata{ClockDomain: "unix", HasClockCalibration: true, ClockCalibrationBoundsNanos: 3}
		}},
		{"producer", func(r *n.Reading) { r.Observation.Producer = "producer" }},
		{"producer-version", func(r *n.Reading) { r.Observation.ProducerVersion = "version" }},
		{"producer-schema", func(r *n.Reading) { r.Observation.ProducerSchema = "schema" }},
		{"raw", func(r *n.Reading) { r.Observation.RawSHA256 = strings.Repeat("a", 64) }},
		{"transform", func(r *n.Reading) { r.Observation.Transform = "transform" }},
		{"transform-hash", func(r *n.Reading) { r.Observation.TransformSHA256 = strings.Repeat("b", 64) }},
	}
	for _, test := range positive {
		t.Run(test.name, func(t *testing.T) {
			r := v5Reading()
			test.mutate(&r)
			want, _ := json.Marshal(r)
			raw := otlpV5Write(t, []n.Reading{r}, otlpV5Options())
			result := otlpV5Read(t, raw)
			got, _ := json.Marshal(result.Readings[0])
			if !bytes.Equal(want, got) {
				t.Fatalf("mapped field dropped: %s != %s", got, want)
			}
		})
	}
	for _, mutate := range []func(*n.Reading){func(r *n.Reading) { r.UnavailableReason = "unsupported" }, func(r *n.Reading) { r.SourceTimestamp = 1 }, func(r *n.Reading) { r.SourceTimestampDomain = "sensor" }, func(r *n.Reading) { r.HasSourceTimestamp = true }, func(r *n.Reading) { r.WallClockUncertaintyNanos = 1 }, func(r *n.Reading) { r.HasWallClockUncertainty = true; r.WallClockUncertaintyNanos = -1 }, func(r *n.Reading) { r.Observation.RawSHA256 = strings.Repeat("A", 64) }} {
		r := v5Reading()
		mutate(&r)
		var w otlpCountingWriter
		if err := n.WriteOTLPMetrics(&w, []n.Reading{r}, otlpV5Options()); err == nil || w.calls != 0 {
			t.Fatal("inconsistent mapped state emitted")
		}
	}
}

func TestOTLPHistogramLimitsAndDigests(t *testing.T) {
	raw := otlpV5Write(t, []n.Reading{v5Reading()}, otlpV5Options())
	first := otlpV5Read(t, raw)
	whitespace := append(append([]byte(" \n"), raw...), []byte("\t\n ")...)
	second := otlpV5Read(t, whitespace)
	if first.TransportSHA256 == second.TransportSHA256 || second.Readings[0].Observation.RawSHA256 != "" {
		t.Fatal("entity bytes did not change transport-only digest")
	}
	digest := sha256.Sum256(whitespace)
	if second.TransportSHA256 != hex.EncodeToString(digest[:]) {
		t.Fatal("whitespace was not hashed")
	}
	r := v5Reading()
	r.Observation.RawSHA256 = strings.Repeat("a", 64)
	asserted := otlpV5Write(t, []n.Reading{r}, otlpV5Options())
	result := otlpV5Read(t, asserted)
	if result.Readings[0].Observation.RawSHA256 != r.Observation.RawSHA256 || result.TransportSHA256 == r.Observation.RawSHA256 {
		t.Fatal("lineage overwritten")
	}
	for _, field := range []string{"MaxMetrics", "MaxReadings", "MaxExplicitBounds"} {
		for _, value := range []int{-1, 65537} {
			o := otlpV5ReadOptions()
			reflect.ValueOf(&o).Elem().FieldByName(field).SetInt(int64(value))
			if _, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), o); err == nil {
				t.Fatalf("invalid %s=%d admitted", field, value)
			}
		}
	}
	for _, limit := range []int64{-1, 64<<20 + 1, int64(len(raw) - 1)} {
		o := otlpV5ReadOptions()
		o.MaxBytes = limit
		if _, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), o); err == nil {
			t.Fatalf("reader bytes %d admitted", limit)
		}
	}
	o := otlpV5ReadOptions()
	o.MaxBytes = int64(len(raw))
	if _, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), o); err != nil {
		t.Fatal(err)
	}
	wire := otlpV5Wire(t, raw)
	metric := otlpV5Metric(wire)
	empty := otlpV5Wire(t, raw)
	otlpV5Metric(empty)["histogram"].(map[string]any)["dataPoints"] = []any{}
	otlpV5Scope(wire)["metrics"] = []any{metric, otlpV5Metric(empty)}
	raw = otlpV5JSON(t, wire)
	o = otlpV5ReadOptions()
	o.MaxMetrics = 1
	if _, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), o); err == nil || !strings.Contains(err.Error(), "metric limit") {
		t.Fatalf("empty metric did not count: %v", err)
	}
	for _, field := range []string{"MaxMetrics", "MaxReadings", "MaxExplicitBounds"} {
		o := otlpV5Options()
		reflect.ValueOf(&o).Elem().FieldByName(field).SetInt(-1)
		var w otlpCountingWriter
		if err := n.WriteOTLPMetrics(&w, []n.Reading{v5Reading()}, o); err == nil || w.calls != 0 {
			t.Fatalf("writer invalid %s emitted", field)
		}
	}
	opts := otlpV5Options()
	opts.MaxBytes = 1
	var w otlpCountingWriter
	if err := n.WriteOTLPMetrics(&w, []n.Reading{v5Reading()}, opts); err == nil || w.calls != 0 {
		t.Fatal("byte cap emitted output")
	}
}

type otlpCancelReader struct {
	ctx     context.Context
	entered chan struct{}
}

func (r *otlpCancelReader) Read(p []byte) (int, error) {
	close(r.entered)
	<-r.ctx.Done()
	return 0, io.EOF
}

type otlpErrorReader struct{ err error }

func (r otlpErrorReader) Read([]byte) (int, error) { return 0, r.err }

type otlpShortWriter struct{}

func (otlpShortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }
func TestOTLPHistogramIOAndCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	r := &otlpCancelReader{ctx: ctx, entered: make(chan struct{})}
	done := make(chan error, 1)
	go func() {
		result, err := n.ReadOTLPHistogramMetrics(ctx, r, otlpV5ReadOptions())
		if len(result.Readings) != 0 || result.TransportSHA256 != "" {
			done <- errors.New("partial canceled result")
			return
		}
		done <- err
	}()
	<-r.entered
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("entered cancellation: %v", err)
	}
	marker := errors.New("reader-marker")
	if _, err := n.ReadOTLPHistogramMetrics(context.Background(), otlpErrorReader{marker}, otlpV5ReadOptions()); !errors.Is(err, marker) {
		t.Fatalf("read error lost: %v", err)
	}
	if err := n.WriteOTLPMetrics(otlpShortWriter{}, []n.Reading{v5Reading()}, otlpV5Options()); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write lost: %v", err)
	}
}

func ExampleReadOTLPHistogramMetrics() {
	count := uint64(0)
	scope := ""
	reading := n.Reading{Name: "latency", Unit: n.UnitSeconds, DeviceUUID: "GPU-example", Evidence: n.EvidenceMeasured, TimestampUnix: 1, Observation: &n.Observation{SourceClass: "imported", ClockEpoch: "unix", Kind: "histogram", OTLPScopeName: &scope, OTLPClock: &n.OTLPClockMetadata{}, Observer: n.Observer{SamplingCoverage: &n.SamplingCoverage{Status: "unknown"}}, Histogram: &n.Histogram{Count: &count, Temporality: "delta"}}}
	bindings := []n.HistogramBinding{{Name: "latency", Unit: n.UnitSeconds}}
	var wire bytes.Buffer
	if err := n.WriteOTLPMetrics(&wire, []n.Reading{reading}, n.ExportOptions{HistogramBindings: bindings}); err != nil {
		panic(err)
	}
	result, err := n.ReadOTLPHistogramMetrics(context.Background(), &wire, n.OTLPReadOptions{HistogramBindings: bindings})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Readings[0].Name, *result.Readings[0].Observation.Histogram.Count)
	// Output: latency 0
}

func otlpV5Local() (n.Reading, n.ExportOptions) {
	r := v5Reading()
	r.TimestampUnix = 0
	r.TimestampMonotonic = 100
	o := r.Observation
	o.SourceClass = "synthetic"
	o.ClockEpoch = "mono-v1"
	o.OTLPScopeName = nil
	o.OTLPClock = nil
	o.Interval = &tracecontract.Interval{Start: 90, End: 100, Clock: "mono"}
	options := otlpV5Options()
	options.UnixTimeMapping = &n.UnixTimeMapping{SourceClockDomain: "mono", SourceClockEpoch: "mono-v1", SourceValidFromNanos: 0, SourceValidUntilNanos: 200, UnixOffsetNanos: 1000, Provenance: "independent-control-caller-assertion"}
	return r, options
}

func TestOTLPHistogramMappingRoundTrip(t *testing.T) {
	local, options := otlpV5Local()
	// These values are frozen independently in the reviewed r7 expected-field artifact.
	want := v5Reading()
	want.TimestampUnix = 1100
	want.Observation.Interval = &tracecontract.Interval{Start: 1090, End: 1100, Clock: "unix"}
	want.Observation.OTLPSourceTime = &n.OTLPSourceTime{SourceClass: "synthetic", SourceClockEpoch: "mono-v1", SourceMonotonicEndNanos: 100, SourceIntervalPresent: true, SourceIntervalClock: "mono", SourceIntervalStartNanos: 90, SourceIntervalEndNanos: 100, MappingDomain: "mono", MappingEpoch: "mono-v1", MappingValidFromNanos: 0, MappingValidUntilNanos: 200, MappingUnixOffsetNanos: 1000, MappingProvenance: "independent-control-caller-assertion"}
	expected, _ := json.Marshal(want)
	original, _ := json.Marshal(local)
	raw := otlpV5Write(t, []n.Reading{local}, options)
	p := otlpV5Point(otlpV5Wire(t, raw))
	if p["timeUnixNano"] != "1100" || p["startTimeUnixNano"] != "1090" {
		t.Fatal("mapped native endpoints")
	}
	source := reflect.ValueOf(*want.Observation.OTLPSourceTime)
	typ := source.Type()
	for i := 0; i < source.NumField(); i++ {
		key := "source_time." + typ.Field(i).Tag.Get("json")
		actual := otlpV5Attr(p, key)
		field := source.Field(i)
		switch field.Kind() {
		case reflect.String:
			if !reflect.DeepEqual(actual, map[string]any{"stringValue": field.String()}) {
				t.Fatalf("source-time field %s: %v", key, actual)
			}
		case reflect.Bool:
			if !reflect.DeepEqual(actual, map[string]any{"boolValue": field.Bool()}) {
				t.Fatalf("source-time field %s: %v", key, actual)
			}
		case reflect.Int64:
			if !reflect.DeepEqual(actual, map[string]any{"stringValue": fmt.Sprint(field.Int())}) {
				t.Fatalf("source-time field %s: %v", key, actual)
			}
		}
	}
	result := otlpV5Read(t, raw)
	got, _ := json.Marshal(result.Readings[0])
	if !bytes.Equal(got, expected) {
		t.Fatalf("projection mismatch:\n%s\n%s", got, expected)
	}
	unchanged, _ := json.Marshal(local)
	if !bytes.Equal(unchanged, original) {
		t.Fatal("mapping mutated producer")
	}
	stream, err := v5Stream(t, 5, result.Readings[0])
	if err != nil {
		t.Fatal(err)
	}
	_, _, samples, err := n.ReadStream(bytes.NewReader(stream), 0)
	if err != nil {
		t.Fatal(err)
	}
	got, _ = json.Marshal(samples[0].Readings[0])
	if !bytes.Equal(got, expected) {
		t.Fatal("mapping metadata lost at stream boundary")
	}
	again := otlpV5Write(t, samples[0].Readings, otlpV5Options())
	if !bytes.Equal(raw, again) {
		t.Fatal("imported offset applied again or assertion changed")
	}
	var w otlpCountingWriter
	if err := n.WriteOTLPMetrics(&w, result.Readings, options); err == nil || w.calls != 0 {
		t.Fatal("mapping option accepted for imported data")
	}
	// Absence fixture is distinct from the mapped local fixture's fixed scope.
	absent := v5Reading()
	absent.Observation.OTLPScopeName = v5String("")
	absentResult := otlpV5Read(t, otlpV5Write(t, []n.Reading{absent}, otlpV5Options()))
	if absentResult.Readings[0].Observation.OTLPSourceTime != nil || *absentResult.Readings[0].Observation.OTLPScopeName != "" {
		t.Fatal("manufactured source/scope assertion")
	}
	// Every member is mandatory. Test omission, null and wrong arm independently.
	for i := 0; i < typ.NumField(); i++ {
		key := "nvtelemetry.source_time." + typ.Field(i).Tag.Get("json")
		for _, change := range []string{"omit", "null", "wrong-arm"} {
			t.Run(typ.Field(i).Name+"/"+change, func(t *testing.T) {
				wire := otlpV5Wire(t, raw)
				point := otlpV5Point(wire)
				attrs := point["attributes"].([]any)
				for j, a := range attrs {
					m := a.(map[string]any)
					if m["key"] == key {
						switch change {
						case "omit":
							attrs = append(attrs[:j], attrs[j+1:]...)
						case "null":
							m["value"] = nil
						case "wrong-arm":
							m["value"] = map[string]any{"doubleValue": 0}
						}
						break
					}
				}
				point["attributes"] = attrs
				result, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(otlpV5JSON(t, wire)), otlpV5ReadOptions())
				if err == nil || len(result.Readings) != 0 || result.TransportSHA256 != "" {
					t.Fatal("partial source-time accepted")
				}
			})
		}
	}
	wire := otlpV5Wire(t, raw)
	otlpV5Attr(otlpV5Point(wire), "source_time.mapping_unix_offset_nanos")["stringValue"] = "1001"
	if _, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(otlpV5JSON(t, wire)), otlpV5ReadOptions()); err == nil {
		t.Fatal("inconsistent source-time arithmetic accepted")
	}
}

func TestOTLPHistogramMappingRefusal(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*n.Reading, *n.ExportOptions)
	}{
		{"Unix-mismatch", func(r *n.Reading, o *n.ExportOptions) { r.TimestampUnix = 1099 }},
		{"no-map", func(r *n.Reading, o *n.ExportOptions) { o.UnixTimeMapping = nil }},
		{"no-end", func(r *n.Reading, o *n.ExportOptions) { r.TimestampMonotonic = 0 }},
		{"epoch", func(r *n.Reading, o *n.ExportOptions) { r.Observation.ClockEpoch = "other" }},
		{"domain", func(r *n.Reading, o *n.ExportOptions) { r.Observation.Interval.Clock = "other" }},
		{"reverse-validity", func(r *n.Reading, o *n.ExportOptions) { o.UnixTimeMapping.SourceValidFromNanos = 201 }},
		{"outside-end", func(r *n.Reading, o *n.ExportOptions) { o.UnixTimeMapping.SourceValidUntilNanos = 99 }},
		{"outside-start", func(r *n.Reading, o *n.ExportOptions) { o.UnixTimeMapping.SourceValidFromNanos = 95 }},
		{"interval-end", func(r *n.Reading, o *n.ExportOptions) { r.Observation.Interval.End = 99 }},
		{"interval-start", func(r *n.Reading, o *n.ExportOptions) { r.Observation.Interval.Start = 0 }},
		{"interval-reversed", func(r *n.Reading, o *n.ExportOptions) { r.Observation.Interval.Start = 101 }},
		{"mapped-zero", func(r *n.Reading, o *n.ExportOptions) { o.UnixTimeMapping.UnixOffsetNanos = -100 }},
		{"mapped-negative", func(r *n.Reading, o *n.ExportOptions) { o.UnixTimeMapping.UnixOffsetNanos = -101 }},
		{"overflow", func(r *n.Reading, o *n.ExportOptions) { o.UnixTimeMapping.UnixOffsetNanos = math.MaxInt64 }},
		{"underflow", func(r *n.Reading, o *n.ExportOptions) {
			r.TimestampMonotonic = math.MinInt64
			r.Observation.Interval = nil
			o.UnixTimeMapping.SourceValidFromNanos = math.MinInt64
			o.UnixTimeMapping.SourceValidUntilNanos = 0
			o.UnixTimeMapping.UnixOffsetNanos = -1
		}},
		{"empty-domain", func(r *n.Reading, o *n.ExportOptions) { o.UnixTimeMapping.SourceClockDomain = "" }},
		{"empty-epoch", func(r *n.Reading, o *n.ExportOptions) { o.UnixTimeMapping.SourceClockEpoch = "" }},
		{"empty-provenance", func(r *n.Reading, o *n.ExportOptions) { o.UnixTimeMapping.Provenance = "" }},
		{"local-source-time", func(r *n.Reading, o *n.ExportOptions) { r.Observation.OTLPSourceTime = &n.OTLPSourceTime{} }},
		{"local-scope-pointer", func(r *n.Reading, o *n.ExportOptions) { r.Observation.OTLPScopeName = v5String("") }},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			r, o := otlpV5Local()
			test.mutate(&r, &o)
			var w otlpCountingWriter
			if err := n.WriteOTLPMetrics(&w, []n.Reading{r}, o); err == nil || w.calls != 0 {
				t.Fatalf("invalid mapping wrote output: %v", err)
			}
		})
	}
	for _, stamp := range []int64{0, 1100} {
		r, o := otlpV5Local()
		r.TimestampUnix = stamp
		result := otlpV5Read(t, otlpV5Write(t, []n.Reading{r}, o))
		if result.Readings[0].TimestampUnix != 1100 {
			t.Fatal("Unix match failed")
		}
	}
	// No interval emits no start, but all false-arm source assertion members remain.
	r, o := otlpV5Local()
	r.Observation.Interval = nil
	raw := otlpV5Write(t, []n.Reading{r}, o)
	point := otlpV5Point(otlpV5Wire(t, raw))
	if _, ok := point["startTimeUnixNano"]; ok {
		t.Fatal("invented interval")
	}
	result := otlpV5Read(t, raw)
	source := result.Readings[0].Observation.OTLPSourceTime
	if source.SourceIntervalPresent || source.SourceIntervalClock != "" || source.SourceIntervalStartNanos != 0 || source.SourceIntervalEndNanos != 0 {
		t.Fatal("false interval arm lost")
	}
}

func TestOTLPHistogramClockResolution(t *testing.T) {
	for _, clock := range []n.OTLPClockMetadata{{}, {ClockDomain: "mono"}, {ClockDomain: "mono", HasClockCalibration: true}, {ClockDomain: "mono", HasClockCalibration: true, ClockCalibrationBoundsNanos: 3}} {
		a, o := otlpV5Local()
		b, _ := otlpV5Local()
		b.DeviceUUID = "other"
		b.Observation.OTLPClock = &clock
		o.ClockDomain = clock.ClockDomain
		o.HasClockCalibration = clock.HasClockCalibration
		o.ClockCalibrationBoundsNanos = clock.ClockCalibrationBoundsNanos
		result := otlpV5Read(t, otlpV5Write(t, []n.Reading{a, b}, o))
		if !reflect.DeepEqual(*result.Readings[0].Observation.OTLPClock, clock) || !reflect.DeepEqual(*result.Readings[1].Observation.OTLPClock, clock) {
			t.Fatal("clock knownness changed")
		}
		if clock != (n.OTLPClockMetadata{}) {
			o.ClockDomain = ""
			o.HasClockCalibration = false
			o.ClockCalibrationBoundsNanos = 0
			var w otlpCountingWriter
			if err := n.WriteOTLPMetrics(&w, []n.Reading{a, b}, o); err == nil || w.calls != 0 {
				t.Fatal("unresolved clock mixture accepted")
			}
		}
	}
	a := v5Reading()
	a.Observation.OTLPClock = &n.OTLPClockMetadata{ClockDomain: "mono", HasClockCalibration: true}
	o := otlpV5Options()
	o.ClockDomain = "mono"
	o.HasClockCalibration = true
	o.ClockCalibrationBoundsNanos = 1
	var w otlpCountingWriter
	if err := n.WriteOTLPMetrics(&w, []n.Reading{a}, o); err == nil || w.calls != 0 {
		t.Fatal("conflicting explicit calibration accepted")
	}
}

func TestOTLPHistogramAttributeCap(t *testing.T) {
	r, o := otlpV5Local()
	r.Observation.Producer = "producer"
	r.Observation.ProducerVersion = "version"
	r.Observation.ProducerSchema = "schema"
	r.Observation.Transform = "transform"
	raw := otlpV5Write(t, []n.Reading{r}, o)
	attrs := otlpV5Point(otlpV5Wire(t, raw))["attributes"].([]any)
	if len(attrs) != 32 {
		t.Fatalf("fixture needs exact32 attrs, got %d", len(attrs))
	}
	result := otlpV5Read(t, raw)
	if result.Readings[0].Observation.Producer != "producer" || result.Readings[0].Observation.OTLPSourceTime == nil {
		t.Fatal("cap dropped fields")
	}
	r.Observation.TransformSHA256 = strings.Repeat("a", 64)
	var w otlpCountingWriter
	if err := n.WriteOTLPMetrics(&w, []n.Reading{r}, o); err == nil || w.calls != 0 {
		t.Fatal("33 attrs dropped fields or emitted")
	}
	wire := otlpV5Wire(t, raw)
	p := otlpV5Point(wire)
	p["attributes"] = append(attrs, map[string]any{"key": "nvtelemetry.observation.transform_sha256", "value": map[string]any{"stringValue": strings.Repeat("a", 64)}})
	result, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(otlpV5JSON(t, wire)), otlpV5ReadOptions())
	if err == nil || len(result.Readings) != 0 {
		t.Fatal("reader attr cap admitted partial result")
	}
}

func otlpV5ID(index int, alphabet string) string {
	base := len(alphabet)
	var s []byte
	for {
		s = append(s, alphabet[index%base])
		index /= base
		if index == 0 {
			break
		}
	}
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
	return string(s)
}
func otlpV5Minimal() n.Reading {
	r := v5Reading()
	r.TimestampUnix = 1
	r.Observation.Family = ""
	r.Observation.OTLPScopeName = v5String("")
	r.Observation.Histogram = &n.Histogram{Count: v5Uint(0), Temporality: "delta"}
	return r
}
func TestOTLPHistogramReadingBoundaries(t *testing.T) {
	const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	for _, size := range []int{4095, 4096, 4097, 65535, 65536, 65537} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			readings := make([]n.Reading, size)
			for i := range readings {
				readings[i] = otlpV5Minimal()
				readings[i].DeviceUUID = otlpV5ID(i, alphabet)
			}
			o := otlpV5Options()
			o.MaxBytes = 64 << 20
			o.MaxReadings = 65536
			var w otlpCountingWriter
			err := n.WriteOTLPMetrics(&w, readings, o)
			if size > 65536 {
				if err == nil || !strings.Contains(err.Error(), "reading limit") || w.calls != 0 {
					t.Fatalf("writer max readings: %v calls%d", err, w.calls)
				}
				return
			}
			if err != nil {
				t.Fatalf("representable writer fixture failed: %v", err)
			}
			raw := w.b.Bytes()
			t.Logf("readings=%d encoded_bytes=%d reader/writerMaxBytes=%d MaxMetrics=4096 MaxExplicitBounds=256", size, len(raw), o.MaxBytes)
			ro := otlpV5ReadOptions()
			ro.MaxBytes = 64 << 20
			ro.MaxReadings = 65536
			result, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), ro)
			if err != nil || len(result.Readings) != size {
				t.Fatalf("reader max readings: %v count%d", err, len(result.Readings))
			}
			// Default and explicit point caps are separate from the recorded byte allowance.
			for _, limit := range []int{0, 4096, 65536} {
				ro.MaxReadings = limit
				result, err = n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), ro)
				maximum := limit
				if maximum == 0 {
					maximum = 4096
				}
				if size > maximum {
					if err == nil || !strings.Contains(err.Error(), "reading limit") || len(result.Readings) != 0 {
						t.Fatalf("reader point guard masked: limit%d %v", limit, err)
					}
				} else if err != nil {
					t.Fatal(err)
				}
			}
			for _, limit := range []int{0, 4096, 65536} {
				o.MaxReadings = limit
				var out otlpCountingWriter
				err = n.WriteOTLPMetrics(&out, readings, o)
				maximum := limit
				if maximum == 0 {
					maximum = 4096
				}
				if size > maximum {
					if err == nil || !strings.Contains(err.Error(), "reading limit") || out.calls != 0 {
						t.Fatalf("writer point guard masked: limit%d %v", limit, err)
					}
				} else if err != nil {
					t.Fatal(err)
				}
			}
			if size == 65536 {
				// Add one distinct valid point to the admitted body; the byte cap still permits it.
				wire := otlpV5Wire(t, raw)
				hist := otlpV5Metric(wire)["histogram"].(map[string]any)
				last := otlpV5Point(otlpV5Wire(t, otlpV5Write(t, []n.Reading{func() n.Reading { r := otlpV5Minimal(); r.DeviceUUID = otlpV5ID(65536, alphabet); return r }()}, otlpV5Options())))
				hist["dataPoints"] = append(hist["dataPoints"].([]any), last)
				over := otlpV5JSON(t, wire)
				if len(over) > 64<<20 {
					t.Fatalf("65537 reader fixture byte-masked: %d", len(over))
				}
				ro.MaxReadings = 65536
				result, err = n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(over), ro)
				t.Logf("reader65537 encoded_bytes=%d", len(over))
				if err == nil || !strings.Contains(err.Error(), "reading limit") || len(result.Readings) != 0 {
					t.Fatalf("reader upper point guard: %v", err)
				}
			}
		})
	}
}

func TestOTLPHistogramMetricBoundaries(t *testing.T) {
	prototype := otlpV5Wire(t, otlpV5Write(t, []n.Reading{otlpV5Minimal()}, otlpV5Options()))
	empty := otlpV5Metric(prototype)
	empty["histogram"].(map[string]any)["dataPoints"] = []any{}
	for _, size := range []int{4095, 4096, 4097, 65535, 65536, 65537} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			groups := make([]any, size)
			for i := range groups {
				groups[i] = empty
			}
			wire := otlpV5Wire(t, otlpV5Write(t, []n.Reading{otlpV5Minimal()}, otlpV5Options()))
			otlpV5Scope(wire)["metrics"] = groups
			raw := otlpV5JSON(t, wire)
			for _, limit := range []int{0, 4096, 65536} {
				o := otlpV5ReadOptions()
				o.MaxBytes = 64 << 20
				o.MaxMetrics = limit
				result, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), o)
				max := limit
				if max == 0 {
					max = 4096
				}
				if size > max {
					if err == nil || !strings.Contains(err.Error(), "metric limit") || len(result.Readings) != 0 {
						t.Fatalf("reader metric guard: limit%d %v", limit, err)
					}
				} else if err != nil || len(result.Readings) != 0 {
					t.Fatalf("valid empty metrics: %v", err)
				}
			}
			t.Logf("input_empty_metrics=%d encoded_bytes=%d MaxBytes=%d; no fabricated readings", size, len(raw), 64<<20)
		})
	}
	// Writer uses nonempty groups. Actual high-count calls hit inherited byte/point
	// caps; do not present those refusals as detection of its independent metric guard.
	prototypeReading := otlpV5Minimal()
	prototypeReading.DeviceUUID = "a"
	prototypeRaw := otlpV5Write(t, []n.Reading{prototypeReading}, otlpV5Options())
	point := otlpV5Metric(otlpV5Wire(t, prototypeRaw))
	for _, size := range []int{4095, 4096, 4097, 65535, 65536, 65537} {
		t.Run("writer/"+fmt.Sprint(size), func(t *testing.T) {
			readings := make([]n.Reading, size)
			bindings := make([]n.HistogramBinding, size)
			groups := make([]any, size)
			for i := range readings {
				name := "h_" + otlpV5ID(i, "abcdefghijklmnopqrstuvwxyz")
				readings[i] = otlpV5Minimal()
				readings[i].Name = name
				readings[i].DeviceUUID = "a"
				bindings[i] = n.HistogramBinding{Name: name, Unit: n.UnitSeconds}
				metric := map[string]any{}
				for k, v := range point {
					metric[k] = v
				}
				metric["name"] = "nvtelemetry." + name
				groups[i] = metric
			}
			// Field-complete encoded reference derives its point from one actual public
			// export. It records geometry, not acceptance of the unrepresentable batch.
			reference := otlpV5Wire(t, prototypeRaw)
			otlpV5Scope(reference)["metrics"] = groups
			referenceBytes := otlpV5JSON(t, reference)
			o := n.ExportOptions{HistogramBindings: bindings, MaxReadings: 65536, MaxMetrics: 65536, MaxBytes: 64 << 20}
			var w otlpCountingWriter
			err := n.WriteOTLPMetrics(&w, readings, o)
			t.Logf("writer_groups=%d encoded_reference_bytes=%d actual_bytes=%d MaxBytes=%d error=%v", size, len(referenceBytes), w.b.Len(), o.MaxBytes, err)
			if size > 65536 {
				if err == nil || !strings.Contains(err.Error(), "reading limit") || w.calls != 0 {
					t.Fatalf("actual competing reading cap: %v", err)
				}
			} else if len(referenceBytes) > 64<<20 {
				if err == nil || !strings.Contains(err.Error(), "byte limit") || w.calls != 0 {
					t.Fatalf("actual competing byte cap: %v", err)
				}
			} else if err != nil {
				t.Fatal(err)
			} else if !bytes.Equal(w.b.Bytes(), referenceBytes) {
				t.Fatal("encoded reference differs from actual writer bytes")
			}
			if size <= 4097 {
				for _, limit := range []int{0, 4096, 65536} {
					o.MaxMetrics = limit
					var out otlpCountingWriter
					err = n.WriteOTLPMetrics(&out, readings, o)
					max := limit
					if max == 0 {
						max = 4096
					}
					if size > max {
						if err == nil || !strings.Contains(err.Error(), "metric limit") || out.calls != 0 {
							t.Fatalf("reachable writer metric guard: %v", err)
						}
					} else if err != nil {
						t.Fatal(err)
					}
				}
			}
		})
	}
	a := v5Reading()
	b := v5Reading()
	b.Name = "other"
	o := otlpV5Options()
	o.HistogramBindings = append(o.HistogramBindings, n.HistogramBinding{Name: b.Name, Unit: n.UnitSeconds})
	o.MaxMetrics = 1
	var w otlpCountingWriter
	if err := n.WriteOTLPMetrics(&w, []n.Reading{a, b}, o); err == nil || !strings.Contains(err.Error(), "metric limit") || w.calls != 0 {
		t.Fatalf("custom writer metric guard: %v", err)
	}
}

func TestOTLPHistogramExplicitBoundaries(t *testing.T) {
	for _, size := range []int{255, 256, 257, 4095, 4096, 4097} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			r := v5Reading()
			h := r.Observation.Histogram
			h.Bounds = nil
			h.Counts = nil
			for i := 0; i < size; i++ {
				h.Bounds = append(h.Bounds, fmt.Sprint(i))
				h.Counts = append(h.Counts, 3)
			}
			h.Bounds = append(h.Bounds, "+Inf")
			h.Counts = append(h.Counts, 3)
			o := otlpV5Options()
			o.MaxBytes = 64 << 20
			o.MaxExplicitBounds = 4096
			var w otlpCountingWriter
			err := n.WriteOTLPMetrics(&w, []n.Reading{r}, o)
			if size > 4096 {
				if err == nil || w.calls != 0 {
					t.Fatal("over maximum bounds emitted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			raw := w.b.Bytes()
			t.Logf("explicit_bounds=%d internal_entries=%d encoded_bytes=%d MaxBytes=%d", size, len(h.Bounds), len(raw), o.MaxBytes)
			for _, limit := range []int{0, 256, 4096} {
				ro := otlpV5ReadOptions()
				ro.MaxBytes = 64 << 20
				ro.MaxExplicitBounds = limit
				result, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), ro)
				max := limit
				if max == 0 {
					max = 256
				}
				if size > max {
					if err == nil || !strings.Contains(err.Error(), "bound limit") || len(result.Readings) != 0 {
						t.Fatalf("reader bound guard: %v", err)
					}
				} else if err != nil || len(result.Readings[0].Observation.Histogram.Bounds) != size+1 {
					t.Fatalf("reader maximum bound: %v", err)
				}
				o.MaxExplicitBounds = limit
				var out otlpCountingWriter
				err = n.WriteOTLPMetrics(&out, []n.Reading{r}, o)
				if size > max {
					if err == nil || !strings.Contains(err.Error(), "bound limit") || out.calls != 0 {
						t.Fatalf("writer bound guard: %v", err)
					}
				} else if err != nil {
					t.Fatal(err)
				}
			}
			if size == 4096 {
				wire := otlpV5Wire(t, raw)
				point := otlpV5Point(wire)
				point["explicitBounds"] = append(point["explicitBounds"].([]any), 4096)
				point["bucketCounts"] = append(point["bucketCounts"].([]any), "0")
				ro := otlpV5ReadOptions()
				ro.MaxBytes = 64 << 20
				ro.MaxExplicitBounds = 4096
				if _, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(otlpV5JSON(t, wire)), ro); err == nil || !strings.Contains(err.Error(), "bound limit") {
					t.Fatalf("reader4097 bound guard: %v", err)
				}
			}
		})
	}
}

func TestOTLPHistogramReachableCaps(t *testing.T) {
	a, b := v5Reading(), v5Reading()
	b.DeviceUUID = "GPU-other"
	raw := otlpV5Write(t, []n.Reading{a, b}, otlpV5Options())
	o := otlpV5Options()
	o.MaxReadings = 1
	var w otlpCountingWriter
	if err := n.WriteOTLPMetrics(&w, []n.Reading{a}, o); err != nil {
		t.Fatal(err)
	}
	w = otlpCountingWriter{}
	if err := n.WriteOTLPMetrics(&w, []n.Reading{a, b}, o); err == nil || !strings.Contains(err.Error(), "reading limit") || w.calls != 0 {
		t.Fatalf("reachable writer point guard: %v", err)
	}
	ro := otlpV5ReadOptions()
	ro.MaxReadings = 1
	result, err := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), ro)
	if err == nil || !strings.Contains(err.Error(), "reading limit") || len(result.Readings) != 0 || result.TransportSHA256 != "" {
		t.Fatalf("reachable reader point guard: %v", err)
	}
	b.Name = "other"
	o = otlpV5Options()
	o.HistogramBindings = append(o.HistogramBindings, n.HistogramBinding{Name: b.Name, Unit: b.Unit})
	raw = otlpV5Write(t, []n.Reading{a, b}, o)
	o.MaxMetrics = 1
	w = otlpCountingWriter{}
	if err := n.WriteOTLPMetrics(&w, []n.Reading{a}, o); err != nil {
		t.Fatal(err)
	}
	w = otlpCountingWriter{}
	if err := n.WriteOTLPMetrics(&w, []n.Reading{a, b}, o); err == nil || !strings.Contains(err.Error(), "metric limit") || w.calls != 0 {
		t.Fatalf("reachable writer metric guard: %v", err)
	}
	ro = otlpV5ReadOptions()
	ro.HistogramBindings = o.HistogramBindings
	ro.MaxMetrics = 1
	result, err = n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw), ro)
	if err == nil || !strings.Contains(err.Error(), "metric limit") || len(result.Readings) != 0 || result.TransportSHA256 != "" {
		t.Fatalf("reachable reader metric guard: %v", err)
	}
}
