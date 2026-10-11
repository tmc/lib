package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"os"
	"strings"
	"testing"
)

func operationHealthLimits() HealthLimits {
	return HealthLimits{MaxBytes: 32 << 10, MaxDevices: 4, MaxSources: 8, MaxObservations: 32, MaxEdges: 8}
}

func operationHealthFixture(t *testing.T) HealthSnapshot {
	t.Helper()
	f, err := os.Open("testdata/health_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s, err := ReadHealthSnapshot(f, operationHealthLimits())
	if err != nil {
		t.Fatal(err)
	}
	return *s
}

func TestHealthPublicConsumer(t *testing.T) {
	s := operationHealthFixture(t)
	var b bytes.Buffer
	if err := WriteHealthSnapshot(&b, s, operationHealthLimits()); err != nil {
		t.Fatal(err)
	}
	got, err := ReadHealthSnapshot(&b, operationHealthLimits())
	if err != nil {
		t.Fatal(err)
	}
	if got.Observations[0].Reading.IntValue != 9007199254740993 {
		t.Fatal("exact uint64 lost")
	}
	if got.DeviceStates["GPU-0"] != "present" || got.Sources[1].State != "failed" {
		t.Fatal("source outage changed device health")
	}
	if !got.Sources[1].QueryStale(100, 10) || got.Sources[0].QueryStale(100, 10) {
		t.Fatal("incorrect query freshness")
	}
}

func TestHealthIndependentNegativeControls(t *testing.T) {
	for _, mutate := range []func(*HealthSnapshot){func(s *HealthSnapshot) { s.Observations[0].Reading.Value = 123456789 }, func(s *HealthSnapshot) {
		r := &s.Observations[0].Reading
		r.HasSourceTimestamp = true
		r.SourceTimestamp = -1
		r.SourceTimestampDomain = s.ClockDomain
	}} {
		s := operationHealthFixture(t)
		mutate(&s)
		b, err := json.Marshal(s)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := ReadHealthSnapshot(bytes.NewReader(b), operationHealthLimits()); err == nil {
			t.Fatal("contradictory reading admitted")
		}
	}
	if !(HealthSource{HasLastSuccess: true, LastSuccessNanos: math.MinInt64}).QueryStale(math.MaxInt64, 0) {
		t.Fatal("freshness subtraction overflow")
	}
}

func TestHealthConsumerMutations(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*HealthSnapshot)
	}{
		{"identity", func(s *HealthSnapshot) { s.Observations[0].Reading.DeviceUUID = "GPU-evil" }},
		{"unit", func(s *HealthSnapshot) { s.Observations[0].Reading.Unit = "bytes" }},
		{"source_outage", func(s *HealthSnapshot) { s.Sources[0].State = "failed" }},
		{"lost_device", func(s *HealthSnapshot) { s.DeviceStates["GPU-0"] = "lost" }},
		{"duplicated_event", func(s *HealthSnapshot) { s.Observations = append(s.Observations, s.Observations[0]) }},
		{"topology_reconfigured", func(s *HealthSnapshot) { delete(s.DeviceStates, "GPU-1") }},
		{"edge_expired", func(s *HealthSnapshot) { s.Edges[0].ValidUntilNanos = 100 }},
		{"edge_peer_unknown", func(s *HealthSnapshot) { s.Edges[0].To = "GPU-other" }},
		{"loss_complete", func(s *HealthSnapshot) { s.LostRecords = 1 }},
		{"hidden_time", func(s *HealthSnapshot) { s.Observations[0].Reading.SourceTimestamp = 1 }},
		{"judgment_as_raw", func(s *HealthSnapshot) { s.Observations[0].Judgment = "healthy" }},
		{"missing_source_version", func(s *HealthSnapshot) { s.Sources[0].Version = "" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := operationHealthFixture(t)
			tc.mutate(&s)
			b, err := json.Marshal(s)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ReadHealthSnapshot(bytes.NewReader(b), operationHealthLimits()); err == nil {
				t.Fatal("mutated bundle admitted")
			}
		})
	}
}

func TestHealthBoundsAndStrictJSON(t *testing.T) {
	b, err := os.ReadFile("testdata/health_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range []string{string(b) + "{}", strings.Replace(string(b), `"version": 1`, `"version": 1, "version": 1`, 1), strings.Replace(string(b), `"version": 1`, `"version": 1, "Version": 2`, 1), strings.Replace(string(b), `"version": 1`, `"version": 1, "unknown": 1`, 1), strings.Repeat("[", 65) + "0" + strings.Repeat("]", 65)} {
		if _, err := ReadHealthSnapshot(strings.NewReader(input), operationHealthLimits()); err == nil {
			t.Fatal("invalid json admitted")
		}
	}
	l := operationHealthLimits()
	l.MaxBytes = 10
	if _, err := ReadHealthSnapshot(bytes.NewReader(b), l); err == nil {
		t.Fatal("oversized bundle admitted")
	}
	l = operationHealthLimits()
	l.MaxSources = 1
	if _, err := ReadHealthSnapshot(bytes.NewReader(b), l); err == nil {
		t.Fatal("source limit not enforced")
	}
	if err := WriteHealthSnapshot(shortHealthWriter{}, operationHealthFixture(t), operationHealthLimits()); err != io.ErrShortWrite {
		t.Fatalf("short write=%v", err)
	}
}

type shortHealthWriter struct{}

func (shortHealthWriter) Write(b []byte) (int, error) { return len(b) - 1, nil }

func FuzzReadHealthSnapshot(f *testing.F) {
	b, err := os.ReadFile("testdata/health_v1.json")
	if err != nil {
		f.Fatal(err)
	}
	f.Add(b)
	f.Fuzz(func(t *testing.T, b []byte) {
		s, err := ReadHealthSnapshot(bytes.NewReader(b), operationHealthLimits())
		if err == nil {
			var out bytes.Buffer
			if err := WriteHealthSnapshot(&out, *s, operationHealthLimits()); err != nil {
				t.Fatal(err)
			}
		}
	})
}

func TestHealthManagedSourceConsumer(t *testing.T) {
	s := operationHealthFixture(t)
	reading := operationManagedReading(t)
	s.Observations = []HealthObservation{{SourceID: "fixture-nvml", Class: "raw", Reading: reading}}
	s.Sources = []HealthSource{{ID: "fixture-nvml", Kind: "dcgm", Version: "fixture-v1", State: "available", HasLastSuccess: true, LastSuccessNanos: 1}}
	s.Edges = nil
	var raw bytes.Buffer
	if err := WriteHealthSnapshot(&raw, s, operationHealthLimits()); err != nil {
		t.Fatal(err)
	}
	got, err := ReadHealthSnapshot(&raw, operationHealthLimits())
	if err != nil {
		t.Fatal(err)
	}
	if got.Observations[0].Reading.Source.Mapping.Name != MetricTemperatureC {
		t.Fatal("default-tag mapping contract not preserved")
	}
	reading.Value = 32
	s.Observations[0].Reading = reading
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ReadHealthSnapshot(bytes.NewReader(b), operationHealthLimits()); err == nil {
		t.Fatal("health consumer accepted contradictory source conversion")
	}
}
