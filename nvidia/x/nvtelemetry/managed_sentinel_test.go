package nvtelemetry_test

import (
	"bytes"
	"context"
	"fmt"
	"strconv"
	"strings"
	"testing"

	telemetry "github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func managedSentinelSample(t *testing.T, producer, raw string) telemetry.Sample {
	t.Helper()
	text := "# TYPE DCGM_FI_DEV_POWER_USAGE gauge\nDCGM_FI_DEV_POWER_USAGE{UUID=\"GPU-a\"} " + raw + "\n"
	sample, err := telemetry.ReadManaged(context.Background(), strings.NewReader(text), telemetry.ManagedImport{SourceID: "protocol-control", Producer: producer, ProducerVersion: "control", NativeOrigin: "control", QueryStartNanos: 10, QueryEndNanos: 20, Fields: telemetry.DCGMExporterFields()})
	if err != nil {
		t.Fatal(err)
	}
	if len(sample.Readings) != 1 {
		t.Fatalf("readings: %d", len(sample.Readings))
	}
	sample.Seq = 1
	return sample
}

func managedSentinelStream(t *testing.T, sample telemetry.Sample) []byte {
	t.Helper()
	var b bytes.Buffer
	w := telemetry.NewWriter(&b, telemetry.CollectorConfig{})
	m := &telemetry.Manifest{RunID: "control", SchemaVersion: 2, Devices: []telemetry.DeviceMetadata{{UUID: "GPU-a"}}, StartTimeMonotonic: 1}
	if err := w.WriteManifest(m); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteSample(&sample); err != nil {
		t.Fatal(err)
	}
	m.IsComplete = true
	m.TotalSamples = 1
	m.EndTimeMonotonic = 30
	if err := w.WriteManifest(m); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestManagedFloatingSentinelPublicRoundTrip(t *testing.T) {
	for _, tc := range []struct{ raw, reason string }{
		{"12.5", ""},
		{"140737488355328", telemetry.ReasonBlank},
		{"140737488355329", telemetry.ReasonQueryFailure},
		{"140737488355330", telemetry.ReasonUnsupported},
		{"140737488355331", telemetry.ReasonPermissionDenied},
		{"140737488355343", telemetry.ReasonBlank},
		{"140737488355344", telemetry.ReasonBlank},
		{"1e100", telemetry.ReasonBlank},
		{"140737488355329.5", telemetry.ReasonBlank},
		{"140737488355331.5", telemetry.ReasonBlank},
		{"NaN", "non_finite"}, {"+Inf", "non_finite"},
	} {
		t.Run(tc.raw, func(t *testing.T) {
			sample := managedSentinelSample(t, "dcgm-exporter", tc.raw)
			r := sample.Readings[0]
			if r.IsAvailable() != (tc.reason == "") || r.UnavailableReason != tc.reason || r.Source.RawValue != tc.raw {
				t.Fatalf("reading: %+v raw=%s", r, r.Source.RawValue)
			}
			if tc.reason == "" && r.Value != 12500 {
				t.Fatalf("converted value %g", r.Value)
			}
			stream := managedSentinelStream(t, sample)
			_, _, samples, err := telemetry.ReadStreamStrict(bytes.NewReader(stream), 0)
			if err != nil {
				t.Fatal(err)
			}
			if samples[0].Readings[0].IsAvailable() != (tc.reason == "") || samples[0].Readings[0].UnavailableReason != tc.reason {
				t.Fatal("replay changed status")
			}
			_, windows, _, err := telemetry.Replay(bytes.NewReader(stream), 0, telemetry.WindowOptions{})
			if err != nil {
				t.Fatal(err)
			}
			summaries := windows["full_run"]
			if tc.reason != "" && len(summaries) != 0 {
				t.Fatalf("blank became a numeric window: %+v", summaries)
			}
			if tc.reason == "" && len(summaries) != 1 {
				t.Fatalf("ordinary value missing window: %+v", summaries)
			}
		})
	}
}

func TestManagedFloatingSentinelReplayRefusesMeasuredForgery(t *testing.T) {
	for _, raw := range []string{"140737488355328", "140737488355329", "140737488355344", "1e100", "140737488355329.5"} {
		for _, evidence := range []string{telemetry.EvidenceMeasured, telemetry.EvidenceDerived} {
			t.Run(fmt.Sprint(raw, "/", evidence), func(t *testing.T) {
				sample := managedSentinelSample(t, "dcgm-exporter", raw)
				r := &sample.Readings[0]
				r.Evidence = evidence
				r.UnavailableReason = ""
				x, err := strconv.ParseFloat(raw, 64)
				if err != nil {
					t.Fatal(err)
				}
				r.Value = x * 1000
				stream := managedSentinelStream(t, sample)
				if _, _, _, err := telemetry.ReadStreamStrict(bytes.NewReader(stream), 0); err == nil {
					t.Fatal("accepted measured blank")
				}
				if _, _, _, err := telemetry.Replay(bytes.NewReader(stream), 0, telemetry.WindowOptions{}); err == nil {
					t.Fatal("replay accepted measured blank")
				}
			})
		}
	}
}

func TestManagedFloatingSentinelOtherProducer(t *testing.T) {
	sample := managedSentinelSample(t, "other-producer", "140737488355344")
	if !sample.Readings[0].IsAvailable() {
		t.Fatal("DCGM sentinel policy applied to unrelated producer")
	}
	if _, _, _, err := telemetry.ReadStreamStrict(bytes.NewReader(managedSentinelStream(t, sample)), 0); err != nil {
		t.Fatal(err)
	}
}
