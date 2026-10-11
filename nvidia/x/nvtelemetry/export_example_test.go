package nvtelemetry_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	telemetry "github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestExportEvidenceReceiver(t *testing.T) {
	for _, evidence := range []string{telemetry.EvidenceMeasured, telemetry.EvidenceDerived} {
		for _, known := range []bool{false, true} {
			r := telemetry.Reading{Name: "power_milliwatts", Unit: "milliwatts", DeviceUUID: "GPU-a", Evidence: evidence, Value: 12500, TimestampUnix: 100, HasWallClockUncertainty: known}
			opts := telemetry.ExportOptions{ClockDomain: "collector", HasClockCalibration: known}
			var b bytes.Buffer
			if err := telemetry.WriteOTLPMetrics(&b, []telemetry.Reading{r}, opts); err != nil {
				t.Fatal(err)
			}
			var request struct {
				ResourceMetrics []struct {
					ScopeMetrics []struct {
						Metrics []struct {
							Gauge struct {
								DataPoints []struct {
									Attributes []struct {
										Key   string
										Value struct {
											StringValue string
											IntValue    string
										}
									}
								}
							}
						}
					}
				}
			}
			if err := json.Unmarshal(b.Bytes(), &request); err != nil {
				t.Fatal(err)
			}
			attrs := map[string]string{}
			for _, a := range request.ResourceMetrics[0].ScopeMetrics[0].Metrics[0].Gauge.DataPoints[0].Attributes {
				attrs[a.Key] = a.Value.StringValue + a.Value.IntValue
			}
			if attrs["nvtelemetry.evidence"] != string(evidence) {
				t.Fatal(attrs)
			}
			want := "false"
			if known {
				want = "true"
			}
			if attrs["nvtelemetry.wall_uncertainty_known"] != want || attrs["nvtelemetry.calibration_known"] != want {
				t.Fatal(attrs)
			}
			_, present := attrs["nvtelemetry.wall_uncertainty_nanos"]
			if present != known {
				t.Fatal(attrs)
			}
			b.Reset()
			if err := telemetry.WriteExposition(&b, []telemetry.Reading{r}, opts); err != nil {
				t.Fatal(err)
			}
			samples, err := telemetry.ReadExposition(t.Context(), &b, telemetry.ExpositionOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if samples[0].Labels["evidence"] != string(evidence) || samples[0].Labels["wall_uncertainty_known"] != want {
				t.Fatal(samples)
			}
		}
	}
}

func TestExportClockRefusal(t *testing.T) {
	r := telemetry.Reading{Name: "power_milliwatts", Unit: "milliwatts", DeviceUUID: "GPU-a", Evidence: telemetry.EvidenceMeasured, TimestampUnix: 100}
	for _, mutate := range []func(*telemetry.Reading){func(r *telemetry.Reading) { r.SourceTimestamp = 1 }, func(r *telemetry.Reading) { r.HasSourceTimestamp = true }, func(r *telemetry.Reading) { r.WallClockUncertaintyNanos = -1 }, func(r *telemetry.Reading) { r.Backend = "bad\nlabel" }} {
		bad := r
		mutate(&bad)
		for _, write := range []func(*bytes.Buffer) error{func(b *bytes.Buffer) error {
			return telemetry.WriteExposition(b, []telemetry.Reading{bad}, telemetry.ExportOptions{})
		}, func(b *bytes.Buffer) error {
			return telemetry.WriteOTLPMetrics(b, []telemetry.Reading{bad}, telemetry.ExportOptions{})
		}} {
			var b bytes.Buffer
			if write(&b) == nil || b.Len() != 0 {
				t.Fatal("invalid metadata admitted")
			}
		}
	}
	var b bytes.Buffer
	r.HasSourceTimestamp = true
	r.SourceTimestampDomain = "sensor"
	r.HasWallClockUncertainty = true
	if err := telemetry.WriteOTLPMetrics(&b, []telemetry.Reading{r}, telemetry.ExportOptions{ClockDomain: "collector", HasClockCalibration: true}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(b.String(), `"intValue":"0"`) {
		t.Fatal(b.String())
	}
}

func TestExportManagedSourceReceiver(t *testing.T) {
	sample, err := telemetry.ReadManaged(t.Context(), strings.NewReader("# TYPE DCGM_FI_DEV_POWER_USAGE gauge\nDCGM_FI_DEV_POWER_USAGE{UUID=\"GPU-a\"} 12.5\n"), telemetry.ManagedImport{SourceID: "secret-incarnation", Producer: "dcgm", ProducerVersion: "exact-version", NativeOrigin: "shared-engine", Fields: telemetry.DCGMExporterFields(), TimestampUnix: 100})
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	if err := telemetry.WriteExposition(&b, sample.Readings, telemetry.ExportOptions{}); err != nil {
		t.Fatal(err)
	}
	samples, err := telemetry.ReadExposition(t.Context(), &b, telemetry.ExpositionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	labels := samples[0].Labels
	if labels["source_status"] != "reported" || labels["source_producer"] != "dcgm" || labels["source_version"] != "exact-version" || labels["source_native_origin"] != "shared-engine" {
		t.Fatal(labels)
	}
	b.Reset()
	if err := telemetry.WriteOTLPMetrics(&b, sample.Readings, telemetry.ExportOptions{}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(b.String(), "secret-incarnation") || !strings.Contains(b.String(), "shared-engine") {
		t.Fatal(b.String())
	}
	sample.Readings[0].Source.MappingDigest = "invalid"
	b.Reset()
	if telemetry.WriteExposition(&b, sample.Readings, telemetry.ExportOptions{}) == nil || b.Len() != 0 {
		t.Fatal("invalid source accepted")
	}
}

func TestExportCalibrationRequiresDomain(t *testing.T) {
	r := telemetry.Reading{Name: "power_milliwatts", Unit: "milliwatts", DeviceUUID: "GPU-a", Evidence: telemetry.EvidenceMeasured, TimestampUnix: 100}
	for _, write := range []func(*bytes.Buffer) error{
		func(b *bytes.Buffer) error {
			return telemetry.WriteExposition(b, []telemetry.Reading{r}, telemetry.ExportOptions{HasClockCalibration: true})
		},
		func(b *bytes.Buffer) error {
			return telemetry.WriteOTLPMetrics(b, []telemetry.Reading{r}, telemetry.ExportOptions{HasClockCalibration: true})
		},
	} {
		var b bytes.Buffer
		if write(&b) == nil || b.Len() != 0 {
			t.Fatal("domainless calibration admitted")
		}
	}
	r.HasSourceTimestamp = true
	r.SourceTimestampDomain = "opaque-sensor"
	r.SourceTimestamp = 7
	var b bytes.Buffer
	if err := telemetry.WriteOTLPMetrics(&b, []telemetry.Reading{r}, telemetry.ExportOptions{ClockDomain: "collector", HasClockCalibration: true}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(b.String(), `"key":"nvtelemetry.sensor_timestamp"`) || strings.Contains(b.String(), "sensor_timestamp_nanos") {
		t.Fatal("invented sensor unit", b.String())
	}
}

func ExampleWriteOTLPLogs() {
	var b bytes.Buffer
	err := telemetry.WriteOTLPLogs(&b, []telemetry.Reading{{Name: telemetry.MetricXIDEvent, Unit: "code", DeviceUUID: "GPU-a", Evidence: telemetry.EvidenceMeasured, IsInt: true, IntValue: 79}}, telemetry.ExportOptions{})
	fmt.Println(json.Valid(b.Bytes()), err)
	// Output: true <nil>
}

func TestOTLPLogsReceiver(t *testing.T) {
	readings := []telemetry.Reading{
		{Name: telemetry.MetricThrottleReasons, Unit: "bitmask", DeviceUUID: "GPU-a", Evidence: telemetry.EvidenceMeasured, IsInt: true, IntValue: ^uint64(0)},
		{Name: telemetry.MetricXIDEvent, Unit: "code", DeviceUUID: "GPU-a", Evidence: telemetry.EvidenceMeasured, IsInt: true, IntValue: 79, TimestampUnix: 100},
		{Name: "power_milliwatts", Unit: "milliwatts", DeviceUUID: "GPU-a", Evidence: telemetry.EvidenceUnavailable, UnavailableReason: "sensitive-driver-message"},
	}
	var b bytes.Buffer
	if err := telemetry.WriteOTLPLogs(&b, readings, telemetry.ExportOptions{}); err != nil {
		t.Fatal(err)
	}
	var request struct {
		ResourceLogs []struct {
			ScopeLogs []struct {
				LogRecords []struct {
					Time     string `json:"timeUnixNano"`
					Severity string `json:"severityText"`
					Body     struct {
						KV struct {
							Values []struct {
								Key   string
								Value struct {
									String string `json:"stringValue"`
								}
							}
						} `json:"kvlistValue"`
					}
				}
			}
		}
	}
	if err := json.Unmarshal(b.Bytes(), &request); err != nil {
		t.Fatal(err)
	}
	records := request.ResourceLogs[0].ScopeLogs[0].LogRecords
	if len(records) != 3 || records[0].Time != "" || records[1].Time != "100" || records[2].Severity != "" {
		t.Fatal(records)
	}
	fields := func(i int) map[string]string {
		m := map[string]string{}
		for _, v := range records[i].Body.KV.Values {
			m[v.Key] = v.Value.String
		}
		return m
	}
	if fields(0)["value_uint64"] != "18446744073709551615" || fields(0)["kind"] != "bitmask" || fields(1)["value_uint64"] != "79" || fields(2)["status"] != "unavailable" || fields(2)["value_uint64"] != "" {
		t.Fatal(records)
	}
	if strings.Contains(b.String(), "sensitive-driver-message") || strings.Contains(b.String(), "private-error") {
		t.Fatal("diagnostic payload leaked")
	}
	for _, bad := range []telemetry.Reading{{Name: telemetry.MetricXIDEvent, Unit: "code", DeviceUUID: "GPU-a", Evidence: telemetry.EvidenceMeasured, Value: 79}, {Name: telemetry.MetricXIDEvent, Unit: "code", DeviceUUID: "GPU-a", Evidence: telemetry.EvidenceMeasured, IsInt: true, TimestampUnix: -1}} {
		b.Reset()
		if telemetry.WriteOTLPLogs(&b, []telemetry.Reading{bad}, telemetry.ExportOptions{}) == nil || b.Len() != 0 {
			t.Fatal("invalid diagnostic admitted")
		}
	}
}

func TestOTLPLogReasonClasses(t *testing.T) {
	var previous []byte
	for _, reason := range []string{telemetry.ReasonUnsupported, telemetry.ReasonPermissionDenied, telemetry.ReasonStale, telemetry.ReasonGPULost, "unsupported_field_type", "redfish_disabled", "SECRET_API_KEY=private"} {
		r := telemetry.Reading{Name: "power_milliwatts", Unit: "milliwatts", DeviceUUID: "GPU-a", Evidence: telemetry.EvidenceUnavailable, UnavailableReason: reason}
		var b bytes.Buffer
		if err := telemetry.WriteOTLPLogs(&b, []telemetry.Reading{r}, telemetry.ExportOptions{}); err != nil {
			t.Fatal(err)
		}
		want := reason
		if strings.HasPrefix(reason, "SECRET_") {
			want = "unknown"
		}
		var request struct {
			ResourceLogs []struct {
				ScopeLogs []struct {
					LogRecords []struct {
						Body struct {
							KV struct {
								Values []struct {
									Key   string
									Value struct {
										String string `json:"stringValue"`
									}
								}
							} `json:"kvlistValue"`
						}
					}
				}
			}
		}
		if err := json.Unmarshal(b.Bytes(), &request); err != nil {
			t.Fatal(err)
		}
		got := ""
		for _, field := range request.ResourceLogs[0].ScopeLogs[0].LogRecords[0].Body.KV.Values {
			if field.Key == "unavailable_reason" {
				got = field.Value.String
			}
		}
		if got != want || bytes.Equal(previous, b.Bytes()) || strings.Contains(b.String(), "SECRET_") {
			t.Fatal("reason collapsed or leaked", got, want)
		}
		previous = append(previous[:0], b.Bytes()...)
	}
}
