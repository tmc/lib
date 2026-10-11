package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func ExampleWriteExposition() {
	var b bytes.Buffer
	e := WriteExposition(&b, []Reading{{Name: "power_milliwatts", Unit: "milliwatts", DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, IsInt: true, IntValue: 12500}}, ExportOptions{})
	fmt.Print(b.String())
	fmt.Println(e)
	// Output:
	// # TYPE nvtelemetry_power_milliwatts gauge
	// # TYPE nvtelemetry_power_milliwatts_available gauge
	// nvtelemetry_power_milliwatts{device_uuid="GPU-a",evidence="measured",export_schema="metrics.v1",backend_status="unknown",source_status="unknown",native_origin_status="unknown",clock_domain_known="false",sensor_timestamp_known="false",wall_uncertainty_known="false",calibration_known="false"} 12500
	// nvtelemetry_power_milliwatts_available{device_uuid="GPU-a",evidence="measured",export_schema="metrics.v1",backend_status="unknown",source_status="unknown",native_origin_status="unknown",clock_domain_known="false",sensor_timestamp_known="false",wall_uncertainty_known="false",calibration_known="false"} 1
	// <nil>
}

func ExampleWriteOTLPMetrics() {
	var b bytes.Buffer
	e := WriteOTLPMetrics(&b, []Reading{{Name: "power_milliwatts", Unit: "milliwatts", DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, IsInt: true, IntValue: 12500, TimestampUnix: 100}}, ExportOptions{})
	fmt.Println(json.Valid(b.Bytes()), e)
	// Output: true <nil>
}

func TestExport(t *testing.T) {
	readings := []Reading{{Name: "total_energy_millijoules", Unit: "millijoules", DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, IsInt: true, IntValue: 9007199254740993, TimestampUnix: 100}}
	var b bytes.Buffer
	if e := WriteExposition(&b, readings, ExportOptions{OpenMetrics: true}); e != nil {
		t.Fatal(e)
	}
	s, e := ReadExposition(t.Context(), &b, ExpositionOptions{Format: OpenMetrics100})
	if e != nil || s[0].RawValue != "9007199254740993" {
		t.Fatal(s, e)
	}
	opts := ExportOptions{HasCounterStart: true, CounterStartUnixNanos: 50}
	if e := WriteOTLPMetrics(&b, readings, opts); e != nil {
		t.Fatal(e)
	}
	if !strings.Contains(b.String(), `"asInt":"9007199254740993"`) || !strings.Contains(b.String(), `"aggregationTemporality":2`) {
		t.Fatal(b.String())
	}
	for _, tc := range []string{"epoch", "overflow", "unavailable", "unit", "duplicate", "limit", "negative", "control", "status"} {
		t.Run(tc, func(t *testing.T) {
			r := append([]Reading(nil), readings...)
			o := opts
			switch tc {
			case "negative":
				r[0].IsInt = false
				r[0].Value = -1
			case "control":
				r[0].DeviceUUID = "GPU\ta"
			case "status":
				r[0].UnavailableReason = "stale"
			case "epoch":
				o.HasCounterStart = false
			case "overflow":
				r[0].IntValue = 1 << 63
			case "unavailable":
				r[0].Evidence = EvidenceUnavailable
			case "unit":
				r[0].Unit = "watts"
			case "duplicate":
				r = append(r, r[0])
			case "limit":
				o.MaxBytes = 2
			}
			if e := WriteOTLPMetrics(&bytes.Buffer{}, r, o); e == nil {
				t.Fatal("accepted")
			}
		})
	}
	for _, id := range []string{"GPU\ta", "GPU\u2028a", "GPU\xffa"} {
		r := readings
		r[0].DeviceUUID = id
		if e := WriteExposition(&bytes.Buffer{}, r, ExportOptions{}); e == nil {
			t.Fatal("invalid label identity accepted")
		}
	}
}
