package nvtelemetry

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
)

func TestOriginal31DeviceMetricNamespace(t *testing.T) {
	for _, name := range []string{
		MetricGPUUtilization, MetricMemoryUtilization, MetricMemoryUsedBytes,
		MetricMemoryTotalBytes, MetricPowerUsageMW, MetricTotalEnergyMJ,
		MetricTemperatureC, MetricSMClockMHz, MetricMemoryClockMHz,
		MetricSMActivity, MetricTensorActivity, MetricDRAMActivity,
		MetricPCIeRxBytesPerSec, MetricPCIeTxBytesPerSec,
		MetricNVLinkRxBytesPerSec, MetricNVLinkTxBytesPerSec,
		MetricPCIERxBytes, MetricPCIETxBytes,
		"nvlink_dl_replay_errors", "nvlink_dl_recovery_errors",
		"nvlink_dl_crc_flit_errors", "nvlink_dl_crc_data_errors", "nvlink_dl_ecc_data_errors",
		"ecc_corrected_errors", "ecc_uncorrected_errors",
	} {
		t.Run(name, func(t *testing.T) {
			unit, ok := LookupMetricUnit(name)
			if !ok {
				t.Fatal("missing registered unit")
			}
			r := Reading{Name: name, Unit: unit, DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, IsInt: true, IntValue: 1, TimestampUnix: 100}
			var out bytes.Buffer
			if err := WriteOTLPMetrics(&out, []Reading{r}, ExportOptions{HasCounterStart: true, CounterStartUnixNanos: 50}); err != nil {
				t.Fatal(err)
			}
			groups := decodeEntityOTLP(t, out.Bytes(), "resourceMetrics")
			metrics := groups[0].(map[string]any)["scopeMetrics"].([]any)[0].(map[string]any)["metrics"].([]any)
			if len(metrics) != 1 || metrics[0].(map[string]any)["name"] != "gpu."+name {
				t.Fatalf("wrong device namespace: %s", out.Bytes())
			}
			attrs := entityResourceAttrs(t, groups[0])
			if entityAttrText(attrs, "device.uuid") != r.DeviceUUID || entityAttrText(attrs, "assignment.status") != "absent" {
				t.Fatal("resource attributes changed", attrs)
			}
		})
	}
	// A process reading uses the same wire name and retains its own resource.
	r := entityExportReading(12, "job-a", 7)
	var out bytes.Buffer
	if err := WriteOTLPMetrics(&out, []Reading{r}, ExportOptions{}); err != nil {
		t.Fatal(err)
	}
	groups := decodeEntityOTLP(t, out.Bytes(), "resourceMetrics")
	attrs := entityResourceAttrs(t, groups[0])
	if !strings.Contains(out.String(), `"name":"gpu.memory_used_bytes"`) || entityAttrText(attrs, "owner.job_id") != "job-a" || entityAttrText(attrs, "assignment.status") != "assigned" {
		t.Fatal("lost process scope", out.String())
	}
	// Renaming the metric does not rename the Prometheus profile or attributes.
	out.Reset()
	if err := WriteExposition(&out, []Reading{{Name: MetricPowerUsageMW, Unit: UnitMilliwatts, DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, Value: 2}}, ExportOptions{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "nvtelemetry_power_milliwatts{") {
		t.Fatal("changed Prometheus profile", out.String())
	}
}

func original31ECCReadings(t *testing.T, value uint64, queryErr error) []Reading {
	t.Helper()
	d := &operationNVMLDevice{uuid: "GPU-a", ecc: value, err: queryErr}
	r, err := newNVMLHealthReader(d, NewFakeClock(1, 100), "node", "epoch", MIGScopeUnknown, nil, "fixture-v1")
	if err != nil {
		t.Fatal(err)
	}
	s, err := r.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	limits := HealthLimits{MaxBytes: 64 << 10, MaxDevices: 1, MaxSources: 1, MaxObservations: 32, MaxEdges: 1}
	var frame bytes.Buffer
	if err := WriteHealthSnapshot(&frame, *s, limits); err != nil {
		t.Fatal(err)
	}
	got, err := ReadHealthSnapshot(&frame, limits)
	if err != nil {
		t.Fatal(err)
	}
	var readings []Reading
	for _, o := range got.Observations {
		if o.Reading.Name == "ecc_corrected_errors" || o.Reading.Name == "ecc_uncorrected_errors" {
			readings = append(readings, o.Reading)
		}
	}
	if len(readings) != 2 || len(d.scopes) != 2 || d.scopes[0] != nvml.VolatileEcc || d.scopes[1] != nvml.VolatileEcc {
		t.Fatal("lost actual volatile ECC queries")
	}
	return readings
}

func TestOriginal31ECCProducerExport(t *testing.T) {
	for _, value := range []uint64{0, 9007199254740993, math.MaxUint64} {
		t.Run(strconv.FormatUint(value, 10), func(t *testing.T) {
			readings := original31ECCReadings(t, value, nil)
			var metrics, logs bytes.Buffer
			opts := ExportOptions{HasCounterStart: true, CounterStartUnixNanos: 50}
			err := WriteOTLPMetrics(&metrics, readings, opts)
			if value > math.MaxInt64 {
				if err == nil || metrics.Len() != 0 {
					t.Fatal("accepted OTLP signed overflow or wrote a prefix")
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				groups := decodeEntityOTLP(t, metrics.Bytes(), "resourceMetrics")
				items := groups[0].(map[string]any)["scopeMetrics"].([]any)[0].(map[string]any)["metrics"].([]any)
				if len(items) != 2 {
					t.Fatal("lost ECC series")
				}
				for i, item := range items {
					m := item.(map[string]any)
					sum := m["sum"].(map[string]any)
					point := sum["dataPoints"].([]any)[0].(map[string]any)
					if m["name"] != "gpu."+readings[i].Name || m["unit"] != "1" || point["asInt"] != strconv.FormatUint(value, 10) || point["startTimeUnixNano"] != "50" || sum["isMonotonic"] != true || fmt.Sprint(sum["aggregationTemporality"]) != "2" {
						t.Fatal("changed counter semantics", m)
					}
				}
			}
			if err := WriteOTLPLogs(&logs, readings, opts); err != nil {
				t.Fatal(err)
			}
			groups := decodeEntityOTLP(t, logs.Bytes(), "resourceLogs")
			for i := range readings {
				fields := entityLogFields(t, groups[0], i)
				if fields["name"] != readings[i].Name || fields["kind"] != KindCounter || fields["unit"] != "count" || fields["value_uint64"] != strconv.FormatUint(value, 10) || fields["status"] != "available" {
					t.Fatal("lost exact diagnostic counter", fields)
				}
			}
			if value <= math.MaxInt64 {
				mg := decodeEntityOTLP(t, metrics.Bytes(), "resourceMetrics")
				if !reflect.DeepEqual(entityResourceAttrs(t, mg[0]), entityResourceAttrs(t, groups[0])) {
					t.Fatal("signals disagree on resource")
				}
			}
		})
	}
}

func TestOriginal31ECCEpochAndUnavailable(t *testing.T) {
	readings := original31ECCReadings(t, 0, nil)
	for _, opts := range []ExportOptions{{}, {HasCounterStart: true}, {HasCounterStart: true, CounterStartUnixNanos: 101}, {MaxBytes: 1, HasCounterStart: true, CounterStartUnixNanos: 50}} {
		var out bytes.Buffer
		if err := WriteOTLPMetrics(&out, readings, opts); err == nil || out.Len() != 0 {
			t.Fatal("invented epoch or wrote rejected prefix")
		}
	}
	for _, queryErr := range []error{nvml.ErrNotSupported, nvml.ErrNoPermission, nvml.ErrGpuIsLost} {
		readings := original31ECCReadings(t, 0, queryErr)
		var out bytes.Buffer
		if err := WriteOTLPMetrics(&out, readings, ExportOptions{HasCounterStart: true, CounterStartUnixNanos: 50}); err == nil || out.Len() != 0 {
			t.Fatal("unavailable ECC became numeric zero")
		}
		if err := WriteOTLPLogs(&out, readings, ExportOptions{}); err != nil {
			t.Fatal(err)
		}
		groups := decodeEntityOTLP(t, out.Bytes(), "resourceLogs")
		for i, r := range readings {
			fields := entityLogFields(t, groups[0], i)
			if fields["status"] != "unavailable" || fields["unavailable_reason"] != r.UnavailableReason || fields["value_uint64"] != "" {
				t.Fatal("lost typed absence", fields)
			}
		}
	}
	// The producer does not know a driver epoch. The caller must declare a
	// fresh cumulative epoch after reset; no timestamp is inferred here.
	for _, value := range []uint64{9, 1} {
		var out bytes.Buffer
		if err := WriteOTLPMetrics(&out, original31ECCReadings(t, value, nil), ExportOptions{}); err == nil || out.Len() != 0 {
			t.Fatal("counter reset invented a start epoch")
		}
	}
}

func TestOriginal31ECCUnregisteredRefusal(t *testing.T) {
	r := original31ECCReadings(t, 1, nil)[0]
	r.Name = "ecc_unregistered_errors"
	for _, write := range []func(*bytes.Buffer) error{
		func(b *bytes.Buffer) error {
			return WriteOTLPMetrics(b, []Reading{r}, ExportOptions{HasCounterStart: true, CounterStartUnixNanos: 50})
		},
		func(b *bytes.Buffer) error { return WriteOTLPLogs(b, []Reading{r}, ExportOptions{}) },
	} {
		var out bytes.Buffer
		if err := write(&out); err == nil || out.Len() != 0 {
			t.Fatal("admitted unregistered scalar")
		}
	}
}

func TestOriginal31MixedHistogramNamespace(t *testing.T) {
	count := uint64(3)
	scope := "github.com/tmc/lib/nvidia/x/nvtelemetry"
	histogram := Reading{Name: "latency", Unit: UnitSeconds, DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, TimestampUnix: 100,
		Observation: &Observation{Family: "histograms", SourceClass: "imported", ClockEpoch: "unix", Kind: "histogram",
			Observer: Observer{SamplingCoverage: &SamplingCoverage{Status: "unknown"}}, OTLPScopeName: &scope, OTLPClock: &OTLPClockMetadata{},
			Histogram: &Histogram{Bounds: []string{"1", "+Inf"}, Counts: []uint64{2, 3}, Count: &count, Temporality: "delta"}}}
	readings := []Reading{histogram, {Name: MetricPowerUsageMW, Unit: UnitMilliwatts, DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, Value: 2, TimestampUnix: 100}}
	var out bytes.Buffer
	if err := WriteOTLPMetrics(&out, readings, ExportOptions{HistogramBindings: []HistogramBinding{{Name: "latency", Unit: UnitSeconds}}}); err != nil {
		t.Fatal(err)
	}
	groups := decodeEntityOTLP(t, out.Bytes(), "resourceMetrics")
	metrics := groups[0].(map[string]any)["scopeMetrics"].([]any)[0].(map[string]any)["metrics"].([]any)
	if len(metrics) != 2 || metrics[0].(map[string]any)["name"] != "nvtelemetry.latency" || metrics[1].(map[string]any)["name"] != "gpu.power_milliwatts" {
		t.Fatal("mixed namespace changed", out.String())
	}
}
