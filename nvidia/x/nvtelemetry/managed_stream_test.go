package nvtelemetry

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
)

func ExampleReadManaged() {
	c := ManagedImport{SourceID: "scrape-1", Producer: "dcgm-exporter", ProducerVersion: "explicit-fixture-v1", NativeOrigin: "dcgm/nvml/GPU-a", Fields: []ManagedField{{Metric: "power", Name: "power_milliwatts", Numerator: 1000, Denominator: 1, Integer: true}}}
	s, e := ReadManaged(context.Background(), strings.NewReader("power{UUID=\"GPU-a\"} 12.5\n"), c)
	fmt.Println(s.Readings[0].IntValue, s.Readings[0].HasSourceTimestamp, e)
	// Output: 12500 false <nil>
}

func ExampleDCGMExporterFields() {
	fmt.Println(len(DCGMExporterFields())) // Output: 8
}

func ExampleValidateReadingSource() {
	fmt.Println(ValidateReadingSource(nil)) // Output: <nil>
}

func TestManagedStream(t *testing.T) {
	c := ManagedImport{SourceID: "snapshot", Producer: "dcgm-exporter", ProducerVersion: "fixture-v1", NativeOrigin: "dcgm/nvml/GPU-a", QueryStartNanos: 10, QueryEndNanos: 20, Fields: []ManagedField{{Metric: "energy", Name: "total_energy_millijoules", Numerator: 1, Denominator: 1, Integer: true}}}
	s, e := ReadManaged(context.Background(), strings.NewReader("energy{UUID=\"GPU-a\",secret=\"drop\"} 9007199254740993 50\n"), c)
	if e != nil {
		t.Fatal(e)
	}
	s.Seq = 1
	if s.Readings[0].HasSourceTimestamp || s.Readings[0].IntValue != 9007199254740993 || s.Readings[0].Source.Labels["secret"] != "" {
		t.Fatal(s)
	}
	for _, version := range []int{1, 2} {
		for _, mut := range []string{"valid", "absent", "identity", "unknown timestamp", "label", "colon label", "sensor", "raw timestamp", "schema", "evidence", "backend", "status", "mapping", "integer", "raw", "digest", "mapped type", "float counter"} {
			t.Run(fmt.Sprintf("%d/%s", version, mut), func(t *testing.T) {
				sample := s
				sample.Readings = append([]Reading(nil), s.Readings...)
				source := *s.Readings[0].Source
				sample.Readings[0].Source = &source
				switch mut {
				case "mapped type":
					source.Mapping.Type = "histogram"
					source.MappingDigest = managedMappingDigest(source.Mapping)
				case "float counter":
					source.Mapping.Integer = false
					source.MappingDigest = managedMappingDigest(source.Mapping)
					sample.Readings[0].IsInt = false
				case "mapping":
					source.Mapping.Numerator = 2
				case "integer":
					sample.Readings[0].IntValue++
				case "raw":
					source.RawValue = "9007199254740995"
				case "digest":
					source.MappingDigest = "sha256:unknown"
				case "schema":
					source.Schema = "unknown"
				case "evidence":
					sample.Readings[0].Evidence = "unknown"
				case "backend":
					source.Producer = "other"
				case "status":
					sample.Readings[0].UnavailableReason = "stale"
				case "sensor":
					sample.Readings[0].HasSourceTimestamp = true
					sample.Readings[0].SourceTimestamp = 1
					sample.Readings[0].SourceTimestampDomain = "unix"
				case "raw timestamp":
					source.RawTimestamp = "51"
				case "absent":
					sample.Readings[0].Source = nil
				case "identity":
					source.NativeOrigin = ""
				case "unknown timestamp":
					source.HasExpositionTimestamp = false
				case "label":
					source.Labels = map[string]string{"bad-label": "x"}
				case "colon label":
					source.Labels = map[string]string{"bad:label": "x"}
				}
				var b bytes.Buffer
				w := NewWriter(&b, CollectorConfig{})
				m := &Manifest{RunID: "managed", SchemaVersion: version, Devices: []DeviceMetadata{{UUID: "GPU-a"}}}
				if e := w.WriteManifest(m); e != nil {
					t.Fatal(e)
				}
				if e := w.WriteSample(&sample); e != nil {
					t.Fatal(e)
				}
				m.IsComplete = true
				m.TotalSamples = 1
				m.EndTimeMonotonic = 20
				if e := w.WriteManifest(m); e != nil {
					t.Fatal(e)
				}
				_, _, _, e := ReadStreamStrict(&b, 0)
				bad := mut != "valid" && mut != "absent"
				if (e != nil) != bad {
					t.Fatalf("strict: %v", e)
				}
			})
		}
	}
	for _, raw := range []string{"-1", "1.1", "18446744073709551616"} {
		if _, e := ReadManaged(context.Background(), strings.NewReader("energy{UUID=\"GPU-a\"} "+raw+"\n"), c); e == nil {
			t.Fatal(raw)
		}
	}
	c.QueryStartNanos = -1
	if _, e := ReadManaged(context.Background(), strings.NewReader("energy{UUID=\"GPU-a\"} 1\n"), c); e == nil {
		t.Fatal("negative import query accepted")
	}
	c.QueryStartNanos = 10
	c.AllowedLabels = []string{"bad:label"}
	if _, err := ReadManaged(context.Background(), strings.NewReader("energy{UUID=\"GPU-a\"} 1\n"), c); err == nil {
		t.Fatal("colon label allowlist accepted")
	}
}
